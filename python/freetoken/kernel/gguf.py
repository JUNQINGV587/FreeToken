"""Borrowed llama.cpp GGUF dequant/GEMM CUDA kernels, JIT-compiled on first use.

The ``.cu``/``.cuh`` under ``csrc/gguf/`` are vendored verbatim from sgl-kernel
(``csrc/quantization/gguf/``), which are themselves ports of llama.cpp. We compile
them through ``torch.utils.cpp_extension.load`` (the same toolchain sglang/vllm use)
into a torch-op module and expose the handful of ops the GGUF path needs. This is a
separate, torch-native extension that sits alongside FreeToken's tvm-ffi kernels.

All ops keep the weight in its native GGUF block layout (packed ``uint8`` rows) and
dequantize *inside* the kernel -- no bf16 copy of the weight is ever materialized.
"""

from __future__ import annotations

import fcntl
import functools
import logging
import os
import pathlib
import shutil
import sys
from contextlib import contextmanager

import torch

_CSRC = pathlib.Path(__file__).parent / "csrc" / "gguf"

logger = logging.getLogger(__name__)


# --- stale build-lock / corrupt-.so recovery (semantic port of sglang #40989) ----


def _build_directory(name: str) -> pathlib.Path:
    """The directory torch.utils.cpp_extension.load would build ``name`` into."""
    try:
        from torch.utils.cpp_extension import _get_build_directory

        return pathlib.Path(_get_build_directory(name, False))
    except (ImportError, AttributeError):
        from torch.utils.cpp_extension import get_default_build_root

        root = os.environ.get("TORCH_EXTENSIONS_DIR") or get_default_build_root()
        if "TORCH_EXTENSIONS_DIR" not in os.environ:
            cu_str = (
                "cpu" if torch.version.cuda is None else f"cu{torch.version.cuda.replace('.', '')}"
            )
            py_str = f"py{sys.version_info.major}{sys.version_info.minor}{getattr(sys, 'abiflags', '')}"
            root = os.path.join(root, f"{py_str}_{cu_str}")
        return pathlib.Path(root) / name


def _is_recoverable_load_error(exc: BaseException, name: str, build_directory: pathlib.Path) -> bool:
    """True when the failure is a stale/corrupt cached .so, not a compile error."""
    message = str(exc).lower()
    current = exc.__cause__ or exc.__context__
    while current is not None:
        message += f"\n{current}".lower()
        current = current.__cause__ or current.__context__

    if any(
        marker in message
        for marker in (
            "error building extension",
            "error compiling objects for extension",
            "ninja",
            "nvcc",
            "gcc",
            "g++",
            "fatal error:",
            "compilation terminated",
        )
    ):
        return False
    if not any(
        marker in message
        for marker in (str(build_directory / f"{name}.so").lower(), f"{name}.so")
    ):
        return False
    return any(
        marker in message
        for marker in (
            "undefined symbol",
            "cannot open shared object file",
            "no such file or directory",
            "file too short",
            "invalid elf header",
            "wrong elf class",
            "elf load command",
            "dlopen",
            "version `glibcxx",
        )
    )


@contextmanager
def _extension_build_lock(build_directory: pathlib.Path):
    """Serialize builds and discard PyTorch lock files left by dead processes.

    torch's FileBaton spins forever on a ``lock`` file whose owner died; a crashed
    build therefore hangs every later start of this box. An outer flock serializes
    concurrent first-builds, and inside it any pre-existing torch lock is stale by
    construction (nobody else can be building).
    """
    build_directory.parent.mkdir(parents=True, exist_ok=True)
    lock_path = build_directory.parent / f".{build_directory.name}.freetoken.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            torch_lock_path = build_directory / "lock"
            if torch_lock_path.exists():
                logger.warning(
                    "removing stale torch extension lock for %s at %s",
                    build_directory.name,
                    torch_lock_path,
                )
                torch_lock_path.unlink(missing_ok=True)
            build_directory.mkdir(parents=True, exist_ok=True)
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _host_compiler() -> str | None:
    """A host compiler nvcc + libtorch headers accept.

    The system default gcc can be too new for the torch headers (gcc 16 hard-errors),
    and on this toolchain even nvcc+gcc-13 trips a non-conformant ``typename
    decltype`` in ``List_inl.h`` once ``torch::Tensor`` is instantiated -- but nvcc
    with ``clang++`` as host compiles it cleanly. So prefer clang++, then fall back
    to an older gcc. Override with ``FREETOKEN_GGUF_HOST_CXX``.

    The gcc-14/15 fallbacks are last on purpose: they are the versions measured to trip
    that header error, so a hit here means the build will fail and the caller should say
    so plainly (see ``_module``) rather than let ninja's output stand as the diagnosis.
    """
    override = os.environ.get("FREETOKEN_GGUF_HOST_CXX")
    if override:
        return override
    for cxx in ("clang++", "g++-12", "g++-13", "g++-14", "g++-15"):
        if shutil.which(cxx):
            return cxx
    return None


def _c_compiler_for(cxx: str) -> str:
    base = os.path.basename(cxx)
    if "clang" in base:
        return shutil.which("clang") or "clang"
    cc = base.replace("g++", "gcc")
    return shutil.which(cc) or cc

def _toolchain_error(host_cxx: str | None, exc: Exception) -> RuntimeError:
    # A compile failure here is almost always the toolchain, not this source: nvcc's
    # host pass inherits a gcc the torch headers reject (ATen/core/List_inl.h wants a
    # `typename` gcc 14+ does not require). Say that, and how to fix it, instead of
    # letting a raw ninja error read as a bug in the GGUF kernels.
    return RuntimeError(
        "could not build the GGUF kernels"
        f" (host compiler: {host_cxx or 'system default'}); nvcc's host pass needs a"
        " compiler the installed torch headers accept. Install clang++ (preferred) or"
        " gcc 12/13, or point FREETOKEN_GGUF_HOST_CXX at one."
        f" Underlying failure: {type(exc).__name__}: {exc}"
    )


@functools.cache
def _module():
    from torch.utils.cpp_extension import load

    extra_cuda_cflags = ["-O3", "--expt-relaxed-constexpr"]
    host_cxx = _host_compiler()
    if host_cxx is not None:
        # Point both nvcc's host pass (-ccbin) and torch's C++ compile (CXX) at a
        # libtorch/nvcc-compatible compiler. Force (not setdefault): the system
        # default (CXX unset -> g++) can be a gcc too new for the torch headers.
        cxx_path = shutil.which(host_cxx) or host_cxx
        extra_cuda_cflags += ["-ccbin", cxx_path]
        os.environ["CXX"] = cxx_path
        os.environ["CC"] = _c_compiler_for(cxx_path)

    # gguf_kernel.cu carries its own PYBIND11_MODULE (appended at the end), so a
    # plain `load` of the single source compiles + binds the ggml_* ops. The build
    # lock + single rebuild recover from a crashed earlier build (stale torch lock,
    # corrupt cached .so) instead of hanging or failing on the leftover state.
    name = "freetoken_gguf_kernels"
    build_directory = _build_directory(name)
    load_kwargs = dict(
        name=name,
        sources=[str(_CSRC / "gguf_kernel.cu")],
        extra_include_paths=[str(_CSRC)],
        extra_cuda_cflags=extra_cuda_cflags,
        build_directory=str(build_directory),
        verbose=True,
    )
    with _extension_build_lock(build_directory):
        try:
            return load(**load_kwargs)
        except Exception as exc:  # noqa: BLE001
            if not _is_recoverable_load_error(exc, name, build_directory):
                raise _toolchain_error(host_cxx, exc) from exc
            logger.warning(
                "detected a stale or broken JIT extension for %s at %s; clearing "
                "its cache and retrying once",
                name,
                build_directory,
            )
            sys.modules.pop(name, None)
            shutil.rmtree(build_directory, ignore_errors=True)
            build_directory.mkdir(parents=True, exist_ok=True)
            try:
                return load(**load_kwargs)
            except Exception as exc2:  # noqa: BLE001
                raise _toolchain_error(host_cxx, exc2) from exc2


# ---- thin typed wrappers (signatures mirror sgl_kernel.quantization.gguf) ----


def ggml_dequantize(
    weight: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Dequantize a packed GGUF weight ``[m, row_bytes]`` to a dense ``[m, n]`` tensor."""
    return _module().ggml_dequantize(weight, quant_type, m, n, dtype)


def ggml_mul_mat_vec_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMVQ: small-batch GEMV with on-the-fly dequant. ``row`` = output features."""
    return _module().ggml_mul_mat_vec_a8(weight, x, quant_type, row)


def ggml_mul_mat_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMQ: large-batch quantized matmul. ``row`` = output features."""
    return _module().ggml_mul_mat_a8(weight, x, quant_type, row)


def ggml_moe_a8(
    x: torch.Tensor,
    weight: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """MMQ grouped expert matmul over stacked experts ``weight[E, row, *]``."""
    return _module().ggml_moe_a8(
        x, weight, sorted_token_ids, expert_ids, num_tokens_post_padded,
        quant_type, row, top_k, tokens,
    )


def ggml_moe_a8_vec(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
) -> torch.Tensor:
    """MMVQ grouped expert GEMV over stacked experts ``weight[E, row, *]``."""
    return _module().ggml_moe_a8_vec(x, weight, topk_ids, top_k, quant_type, row, tokens)


def ggml_moe_get_block_size(quant_type: int) -> int:
    return _module().ggml_moe_get_block_size(quant_type)


__all__ = [
    "ggml_dequantize",
    "ggml_mul_mat_vec_a8",
    "ggml_mul_mat_a8",
    "ggml_moe_a8",
    "ggml_moe_a8_vec",
    "ggml_moe_get_block_size",
]

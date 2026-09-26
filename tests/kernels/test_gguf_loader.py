"""Stale build-lock / corrupt-.so recovery for the GGUF JIT loader (sglang #40989 port)."""
from __future__ import annotations

import pathlib

import pytest

torch = pytest.importorskip("torch")

from freetoken.kernel.gguf import (  # noqa: E402
    _build_directory,
    _extension_build_lock,
    _is_recoverable_load_error,
)


def test_build_directory_matches_torch():
    from torch.utils.cpp_extension import _get_build_directory

    assert _build_directory("freetoken_gguf_kernels") == pathlib.Path(
        _get_build_directory("freetoken_gguf_kernels", False)
    )


def test_build_lock_removes_stale_torch_lock(tmp_path):
    build_dir = tmp_path / "ext_build" / "freetoken_gguf_kernels"
    build_dir.mkdir(parents=True)
    (build_dir / "lock").write_text("stale")
    with _extension_build_lock(build_dir):
        assert not (build_dir / "lock").exists()
        assert (tmp_path / "ext_build" / ".freetoken_gguf_kernels.freetoken.lock").exists()


def test_recoverable_load_error_markers(tmp_path):
    build_dir = tmp_path / "freetoken_gguf_kernels"
    so = build_dir / "freetoken_gguf_kernels.so"
    stale = RuntimeError(
        f"Error importing extension: {so}: cannot open shared object file: "
        "No such file or directory"
    )
    assert _is_recoverable_load_error(stale, "freetoken_gguf_kernels", build_dir)
    corrupt = ImportError(f"{so}: file too short")
    assert _is_recoverable_load_error(corrupt, "freetoken_gguf_kernels", build_dir)


def test_compile_errors_are_not_recoverable(tmp_path):
    build_dir = tmp_path / "freetoken_gguf_kernels"
    compile_err = RuntimeError(
        "Error building extension 'freetoken_gguf_kernels': ninja: build stopped"
    )
    assert not _is_recoverable_load_error(compile_err, "freetoken_gguf_kernels", build_dir)
    nvcc_err = RuntimeError("nvcc fatal: unsupported gpu architecture")
    assert not _is_recoverable_load_error(nvcc_err, "freetoken_gguf_kernels", build_dir)
    unrelated = RuntimeError("cannot open shared object file: libother.so")
    assert not _is_recoverable_load_error(unrelated, "freetoken_gguf_kernels", build_dir)


def test_error_chain_is_inspected(tmp_path):
    build_dir = tmp_path / "freetoken_gguf_kernels"
    cause = OSError(
        f"{build_dir}/freetoken_gguf_kernels.so: undefined symbol: _ZN2at4fooEv"
    )
    wrapper = RuntimeError("extension import failed")
    wrapper.__cause__ = cause
    assert _is_recoverable_load_error(wrapper, "freetoken_gguf_kernels", build_dir)

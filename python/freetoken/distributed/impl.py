from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from freetoken.distributed import DistributedInfo
    from freetoken.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


@dataclass
class CustomAllReduceImpl(DistributedImpl):
    """vLLM custom-allreduce donor for small decode-time tensors (2x PCIe GPUs).

    Measured on 2xL20 (PCIe P2P, no NVLink): pynccl/NCCL costs 22-95 us per
    [bs<=8, 2560] all-reduce; the donor's one-stage kernel does it in ~4-9 us with
    fp32 accumulation in rank order -- bit-identical to NCCL's sum. Tensors the donor
    declines (oversize / dtype / non-contiguous) fall back to the inner plugin.

    expandable_segments note: the donor's zero-copy capture path registers graph-pool
    addresses via cudaIpcGetMemHandle, which VMM (expandable segments) pointers do not
    support. ``_CopyCaptureCAR`` therefore captures the copy-in path instead: the
    captured kernel reads the donor's own cudaMalloc'd IPC buffer (IPC-able), at the
    price of one ~2 us D2D copy per call -- still far cheaper than pynccl. The
    ``capture()`` ctx is still required around graph capture so the donor exchanges
    IPC handles for its own buffers at capture exit.
    """

    inner: DistributedImpl
    ca: Any  # vllm CustomAllreduce
    calls: int = 0
    calls_custom: int = 0

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        # The counters feed verify_ar_sequence_boot: the donor pairs spin barriers by
        # call order, so per-rank (total, custom-served) counts must match after boot.
        self.calls += 1
        out = self.ca.custom_all_reduce(x) if not self.ca.disabled else None
        if out is not None:
            self.calls_custom += 1
            return out
        return self.inner.all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner.all_gather(x)


class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)

    @staticmethod
    def graph_capture_ctx():
        """The custom-AR donor must exchange IPC handles for captured buffers."""
        for p in reversed(DistributedCommunicator.plugins):
            ca = getattr(p, "ca", None)
            if ca is not None and not ca.disabled:
                return ca.capture()
        import contextlib

        return contextlib.nullcontext()


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    Enable PyNCCL-based distributed communication for tensor parallelism.
    """
    if tp_info.size == 1:
        return
    from freetoken.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))
    enable_custom_all_reduce(tp_info, tp_cpu_group, max_bytes)


def enable_custom_all_reduce(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> bool:
    """Attach the vLLM custom-allreduce donor ahead of the current plugin, if usable.

    The donor self-disables when P2P is unavailable; any failure leaves the pynccl
    plugin as the active one. The donor is OPT-IN: its 1-stage spin-read kernel has
    produced Xid 31 MMU faults under PCIe P2P+H2D contention twice on this fleet
    (2026-09-16 reproduced, 2026-09-23 suspected), so it only activates with
    FREETOKEN_CUSTOM_ALL_REDUCE=1.
    """
    if tp_info.size == 1 or os.getenv("FREETOKEN_CUSTOM_ALL_REDUCE", "0") != "1":
        return False
    global _boot_tp_cpu_group
    _boot_tp_cpu_group = tp_cpu_group
    try:
        from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

        class _CopyCaptureCAR(CustomAllreduce):
            """Under capture, use the copy-in path: VMM (expandable_segments) input
            pointers cannot be IPC-registered, but the donor's own buffer can."""

            def custom_all_reduce(self, input):
                if self.disabled or not self.should_custom_ar(input):
                    return None
                if self._IS_CAPTURING:
                    if torch.cuda.is_current_stream_capturing():
                        return self.all_reduce(input, registered=False)
                    # The base class would return uninitialized memory here; fail loudly.
                    raise RuntimeError(
                        "custom AR called inside the capture ctx but off the capturing stream"
                    )
                return super().custom_all_reduce(input)

        ca = _CopyCaptureCAR(
            group=tp_cpu_group,
            device=tp_info.rank,
            # decode ARs are [bs, hidden] (a few KB); 2 MiB leaves headroom without
            # growing the registered IPC pool pointlessly.
            max_size=max(2 * max_bytes, 2 << 20),
        )
    except Exception as exc:
        from freetoken.utils import init_logger

        init_logger(__name__).warning(
            f"custom all-reduce donor unusable ({exc!r}); staying on pynccl"
        )
        return False
    # The donor's P2P self-check is a per-rank decision: a split verdict would make one
    # rank spin-wait for a peer that stayed on pynccl (startup hang). Take consensus.
    # The prefetch env gates ride the same gather: they are pure scheduling knobs (no AR
    # call-order or numerics effect), but a drifted pair would silently desync the two
    # ranks' prefill timing, so refuse the boot instead of serving a skewed TP pair.
    from freetoken.moe.offload_cache import PREFILL_PREFETCH_EARLY
    from freetoken.moe.ownership import PREFILL_PREFETCH_DEPTH

    verdicts = [None] * tp_info.size
    dist.all_gather_object(
        verdicts, (ca.disabled, PREFILL_PREFETCH_DEPTH, PREFILL_PREFETCH_EARLY), group=tp_cpu_group
    )
    if any(disabled for disabled, _, _ in verdicts):
        ca.close()
        return False
    if len({(depth, early) for _, depth, early in verdicts}) != 1:
        ca.close()
        raise RuntimeError(
            "FREETOKEN_PREFILL_PREFETCH_DEPTH/EARLY diverged across TP ranks: "
            f"{verdicts} (set them identically on every rank)"
        )
    DistributedCommunicator.plugins.append(
        CustomAllReduceImpl(DistributedCommunicator.plugins[-1], ca)
    )
    from freetoken.utils import init_logger

    init_logger(__name__).info_rank0("custom all-reduce donor active (vLLM one-stage kernel)")
    return True


_boot_tp_cpu_group = None


def verify_ar_sequence_boot() -> None:
    """Boot-time guard for the donor's sequence-paired spin barriers (F1).

    The 1-stage kernel pairs ARs across ranks purely by call order; a single divergent
    call would silently misalign every later AR (no hang, no error). The sequence is
    structural (identical on both ranks), so comparing call counts after warmup+capture
    catches any divergence before serving. Runtime divergence remains a documented
    latent risk (per-step checking would cost a host sync per step)."""
    active = DistributedCommunicator.plugins[-1] if DistributedCommunicator.plugins else None
    if not isinstance(active, CustomAllReduceImpl) or _boot_tp_cpu_group is None:
        return
    allc = [None] * dist.get_world_size(_boot_tp_cpu_group)
    dist.all_gather_object(allc, (active.calls, active.calls_custom), group=_boot_tp_cpu_group)
    if len(set(allc)) != 1:
        raise RuntimeError(
            f"custom AR call sequence diverged across ranks during boot: {allc}"
        )


def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    DistributedCommunicator.plugins = []

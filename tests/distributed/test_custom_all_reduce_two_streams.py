"""Two-stream concurrent issue on one custom-AR communicator (sglang #31135 scenario).

The donor's one-stage kernel is single-in-flight-per-communicator: two ARs issued
from two CUDA streams without serialization alias the rendezvous state and hang
permanently (unkillable spin loops). CustomAllReduceImpl._serialize_ar_stream must
make concurrent issue terminate and stay bit-identical to the NCCL reference.
Only the GUARDED variant runs here -- deliberately deadlocking shared fleet GPUs
is not an acceptable test.

Run: torchrun --nproc_per_node=2 tests/distributed/test_custom_all_reduce_two_streams.py
"""
from __future__ import annotations

import torch
import torch.distributed as dist

HIDDEN = 2560
ITERS = 200
MAX_BYTES = 2 * 8 * HIDDEN * 4  # 2x headroom: donor declines inp_size >= max_size


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    cpu_group = dist.new_group(backend="gloo")

    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    from freetoken.distributed.impl import CustomAllReduceImpl

    class _Inner:
        def all_reduce(self, x):
            return x

        def all_gather(self, x):
            return x

    ca = CustomAllreduce(group=cpu_group, device=rank, max_size=MAX_BYTES)
    assert not ca.disabled, "donor P2P self-check failed on this box"
    impl = CustomAllReduceImpl(_Inner(), ca)

    s_main = torch.cuda.current_stream()
    s_alt = torch.cuda.Stream()
    torch.manual_seed(1234)  # identical sequence on both ranks
    try:
        for i in range(ITERS):
            x = torch.randn(8, HIDDEN, dtype=torch.float32, device="cuda")
            ref = x.clone()
            dist.all_reduce(ref, op=dist.ReduceOp.SUM)
            if i % 2 == 0:
                out = impl.all_reduce(x)
            else:
                with torch.cuda.stream(s_alt):
                    out = impl.all_reduce(x)
                s_main.wait_stream(s_alt)  # host-side join for the assert below
            assert torch.equal(out, ref), f"rank{rank} iter{i}: guard broke numerics"
        assert impl._ar_event is not None, "guard never saw a stream switch"
    finally:
        ca.close()
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print("TWO-STREAM-OK", flush=True)


if __name__ == "__main__":
    main()

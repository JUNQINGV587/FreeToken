from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    # DSV4/DSV4.1 derive their prefill chunk from the KV pool's window budget, which scales with
    # --num-tokens; a large pool (a 1M-token reservation) would allow a ~100k-token chunk whose
    # indexer transient is O(chunk x context). >0 caps the chunk independently of the pool
    # (rounded down to whole window pages). 0 keeps the historical pool-budget behaviour.
    prefill_chunk_tokens: int = 0
    # Adaptive variant of the cap above: ``prefill_chunk_tokens`` becomes a *ceiling* (it still
    # sizes buffers, warmup lengths and the pynccl scratch) and each prefill pass takes
    # ``adaptive_prefill_budget(longest pending context, ceiling)`` -- the whole ceiling for a
    # short prompt (one disk-tier pass instead of ceil(len/chunk)) and a context-proportional
    # chunk for a long one, so the indexer's O(chunk x context) transient stays inside the
    # envelope already measured safe on this box. See freetoken/scheduler/chunk_policy.py.
    prefill_chunk_adaptive: bool = False
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/freetoken_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/freetoken_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/freetoken_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True

"""Offline admission-policy simulators over an ordered MoE route trace.

Why this exists
---------------
``--moe-cache-size`` buys decode speed by keeping more ``(layer, expert)`` rows resident,
and the engine admits them with one unified LRU (``flashlib.kernels.slot_cache.lru_ensure``
-- device-side, fixed shapes, CUDA-graph capturable). ``route_trace`` already mirrors that
LRU exactly, so a *capacity* decision can be replayed offline (PLAN_TP_EP.md 6A). This
module extends the same replay to policies the kernel does not have yet -- a pinned hot
set, an LFU victim rule, and their hybrids -- so the prize can be measured on a captured
trace **before** anything is written into the engine or the kernel.

It also answers the question the histogram cannot. ``decode_routing_stats`` reports the
hit rate of the best *fixed* set (``static_topk_hit_*``) and deliberately warns that it is
"NOT an upper bound on a dynamic cache, because LRU exploits temporal locality and can
beat it"; a histogram has thrown the activation ORDER away, so it can neither confirm nor
refute that. Replaying an ordered trace can.

Semantics
---------
One ``ensure`` == one ``lru_ensure`` call == one step. All policies share the kernel's
front half: ids are deduped, and every slot touched by the *current* call is protected
(the kernel assigns ``usage == step``, and its victim scan maps that to ``USAGE_MAX``), so
a batch can never evict the row it just admitted. They differ only in victim selection:

* ``lru`` -- ascending ``(usage, slot)``. ``Sim(policy="lru")`` is asserted against
  ``route_trace.LRU`` in ``tests/moe/test_route_policy.py``; any divergence is a bug here.
* ``lfu`` -- ascending ``(count, usage, slot)``: activation frequency first, recency as
  the tie-break. Needs no offline calibration, so it is the policy that could ship
  without a profiling pass.
* ``pin`` -- ``pin_rows`` occupy a reserved, never-evicted set of ``len(pin_rows)`` slots;
  everything else runs LRU over the remaining ``cache_size - P``. The one-off copies that
  fill the reserved slots are charged as misses the first time each pinned row is touched
  (a real implementation pays exactly those copies, and they are bounded by P).

Pinning is the interesting one, but it is **not** implementable with a one-off host write.
Today's kernel (``flashlib/kernels/slot_cache/triton/lru_ensure.py``) already refuses to
evict a slot whose ``usage`` is the dtype maximum, so writing that sentinel into the
pinned slots looks like a zero-kernel-change pin -- except the same kernel *unconditionally*
stores ``step`` back into ``lru_usage`` on a hit (``:97``), so one hit drops the sentinel
and the pin evaporates. Two honest options, in cost order:

1. re-write the sentinel **before every** ``ensure`` call for the pinned slots (a fixed
   shape indexed fill, graph-capturable, but one extra device op per layer per step -- the
   same per-layer-per-step amplification that makes the online ``freq_pin`` learner cost
   ~1.8% decode; see ``exp/cache-freqpin`` and its t6 A/B report);
2. pass a pin bitmap into the kernel and fold it into the victim scan (one extra lane
   load; ``exp/cache-freqpin`` proves a repo-local Triton kernel can replace ``lru_ensure``).

Either way the *policy* is what this module measures; the open question is not whether a
fixed hot set can be built, but whether it beats LRU on a real workload -- and on this
model the engine's own in-run numbers say the headroom is large (see
``20261002-dsv41-moe-cache-admission-finding.md``).

Nothing here imports torch or CUDA: it is a pure-Python replay over the ids
``RouteTraceRecorder`` wrote, so it runs on a bare interpreter.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

POLICIES = ("lru", "lfu", "pin")


@dataclass
class SimResult:
    """One (policy, cache_size) replay outcome."""

    policy: str
    cache_size: int
    pin_rows: int
    calls: int
    active: int
    misses: int
    hits: int
    miss_rate: float
    hit_rate: float
    copied_gib: float | None = None
    max_call_miss: int = 0
    p95_call_miss: int = 0
    mean_call_miss: float = 0.0

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


class Sim:
    """Replay a route trace through one slot pool under one admission policy.

    ``cache_size`` is the whole pool; ``pin_rows`` (policy ``pin`` only) are global row ids
    ``layer * num_experts + expert`` that hold a reserved slot for the whole replay.
    ``row_bytes`` is the size of one expert copy, used only to turn misses into GiB.
    """

    def __init__(
        self,
        cache_size: int,
        num_experts: int,
        *,
        policy: str = "lru",
        pin_rows=(),
        row_bytes: int | None = None,
        band_slots: int = 0,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        pin = frozenset(pin_rows)
        if len(pin) > cache_size:
            raise ValueError(
                f"pin_rows ({len(pin)}) cannot exceed the pool ({cache_size})")
        self.policy = policy
        self.cache_size = cache_size
        self.num_experts = num_experts
        self.pin = pin if policy == "pin" else frozenset()
        self.row_bytes = row_bytes
        # A pinned row never needs a managed slot, so the LRU side shrinks by P.
        self.n_slots = cache_size - len(self.pin)
        # Owner-EP prefill overlap borrows the slot cache's first
        # ``prefill_depth * num_experts`` slots as the buffer ring and invalidates them
        # when the ring is reused (offload_cache._invalidate_prefill_buffer), so every
        # row that lands there is dropped at the next prefill chunk.  Modelled as a
        # positional band that a caller wipes at chunk boundaries.
        self.band = min(band_slots, self.n_slots)
        self.slot_of: dict[int, int] = {}
        self.owner: list[int | None] = [None] * self.n_slots
        self.usage: list[int] = [0] * self.n_slots
        self.count: list[int] = [0] * self.n_slots
        self.heap: list[tuple] = []
        self.free = list(range(self.n_slots))
        self.step = 0
        self.calls = 0
        self.active = 0
        self.misses = 0
        self.call_misses: list[int] = []
        self._pinned_filled: set[int] = set()

    # ------------------------------------------------------------------ internals
    def _push(self, s: int) -> None:
        if self.policy == "lfu":
            heapq.heappush(self.heap, (self.count[s], self.usage[s], s))
        else:
            heapq.heappush(self.heap, (self.usage[s], s))

    def _victim(self) -> int:
        """Coldest managed slot, honouring the kernel's in-batch protection.

        The kernel maps a slot touched by the current call (``usage == step``) to
        ``USAGE_MAX`` and claims the victim in-register, so one call can never evict a row
        it just admitted. The equivalent here is to *defer* such entries and push them
        back afterwards -- dropping them would leak the slot out of the heap, and it would
        then never be evictable again.
        """
        if self.free:
            return self.free.pop()
        lfu = self.policy == "lfu"
        deferred: list[tuple] = []
        try:
            while self.heap:
                entry = heapq.heappop(self.heap)
                s = entry[-1]
                if self.owner[s] is None:
                    continue  # stale: the slot was evicted already
                if self.usage[s] >= self.step:
                    deferred.append(entry)  # touched by this very call: not a victim
                    continue
                if lfu and (self.count[s], self.usage[s]) != entry[:2]:
                    continue  # stale frequency entry; the current one is still queued
                if not lfu and self.usage[s] != entry[0]:
                    continue  # stale recency entry
                del self.slot_of[self.owner[s]]
                self.owner[s] = None
                return s
        finally:
            for entry in deferred:
                heapq.heappush(self.heap, entry)
        raise RuntimeError(
            f"no evictable slot: pool={self.cache_size} pin={len(self.pin)} "
            f"step={self.step} -- every managed slot was touched by this call, which the "
            f"kernel cannot represent either (a call may not ask for more distinct ids "
            f"than the pool holds)")

    def wipe_band(self) -> int:
        """Invalidate the borrowed prefill-ring band (one prefill chunk boundary).

        ``_invalidate_prefill_buffer`` clears ``slot_for_id`` and zeroes ``usage`` for the
        borrowed range, which makes those slots the coldest in the pool -- so the LRU side
        refills them first and they absorb the next misses.  Returns the number of rows
        dropped.  Pinned rows never sit in the band (the engine carves the pin region out
        of the persistent region above it).
        """
        dropped = 0
        for s in range(self.band):
            fid = self.owner[s]
            if fid is None:
                continue
            del self.slot_of[fid]
            self.owner[s] = None
            self.usage[s] = 0
            self.count[s] = 0
            # The kernel sees these as empty slots (``id_of_slot == -1``), so they return
            # to the free list; any stale heap entry for them is discarded on pop.
            self.free.append(s)
            dropped += 1
        return dropped

    # ------------------------------------------------------------------ replay
    def ensure(self, layer: int, ids) -> None:
        """Replay one ``lru_ensure`` call (one layer, one step's routed ids)."""
        self.step += 1
        self.calls += 1
        base = layer * self.num_experts
        uniq = sorted(set(ids))
        self.active += len(uniq)
        missed: list[int] = []
        for e in uniq:
            fid = base + e
            if fid in self.pin:
                if fid not in self._pinned_filled:  # the one-off fill copy
                    self._pinned_filled.add(fid)
                    missed.append(fid)
                continue
            s = self.slot_of.get(fid)
            if s is None:
                missed.append(fid)
            else:
                self.usage[s] = self.step
                self.count[s] += 1
                self._push(s)
        self.misses += len(missed)
        self.call_misses.append(len(missed))
        for fid in missed:
            if fid in self.pin:
                continue  # resident: filled above, needs no managed slot
            s = self._victim()
            self.slot_of[fid] = s
            self.owner[s] = fid
            self.usage[s] = self.step
            self.count[s] = 1
            self._push(s)

    def result(self, policy: str | None = None) -> SimResult:
        misses, active = self.misses, self.active
        call_misses = sorted(self.call_misses)
        n = len(call_misses)
        p95 = call_misses[min(n - 1, int(0.95 * n))] if n else 0
        gib = None
        if self.row_bytes:
            gib = misses * self.row_bytes / (1024 ** 3)
        return SimResult(
            policy=policy or self.policy,
            cache_size=self.cache_size,
            pin_rows=len(self.pin),
            calls=self.calls,
            active=active,
            misses=misses,
            hits=active - misses,
            miss_rate=misses / active if active else 0.0,
            hit_rate=(active - misses) / active if active else 0.0,
            copied_gib=gib,
            max_call_miss=call_misses[-1] if n else 0,
            p95_call_miss=p95,
            mean_call_miss=(misses / self.calls) if self.calls else 0.0,
        )


def learn_pin_set(
    records,
    budget: int,
    num_experts: int,
    *,
    mode: str = "global",
    warmup_frac: float = 1.0,
    phase: int = 0,
) -> set[int]:
    """Pick ``budget`` global row ids to pin, from a prefix of the trace.

    ``mode="global"`` takes the most-activated rows overall (the shape of
    ``static_topk_hit_global``); ``mode="per_layer"`` takes the top
    ``budget // num_layers`` rows of every layer (the shape of
    ``static_topk_hit_by_layer_even_split``).

    ``warmup_frac < 1`` is the honest version: the pin set is learned from the first
    fraction of the trace and then frozen, so the replay never sees the future. ``1.0``
    reproduces the engine's hindsight figure and is an upper bound, not a deployable
    policy.
    """
    if mode not in ("global", "per_layer"):
        raise ValueError("mode must be 'global' or 'per_layer'")
    if not 0.0 < warmup_frac <= 1.0:
        raise ValueError("warmup_frac must be in (0, 1]")
    dec = [r for r in records if r[0] == phase]
    cut = max(1, int(len(dec) * warmup_frac))
    warm = dec[:cut]

    if mode == "global":
        counts: dict[int, int] = {}
        for _ph, layer, ids in warm:
            base = layer * num_experts
            for e in set(ids):
                fid = base + e
                counts[fid] = counts.get(fid, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return {fid for fid, _ in ranked[:budget]}

    per_layer: dict[int, dict[int, int]] = {}
    for _ph, layer, ids in warm:
        c = per_layer.setdefault(layer, {})
        for e in set(ids):
            c[e] = c.get(e, 0) + 1
    n_layers = max(1, len(per_layer))
    per = max(1, budget // n_layers)
    out: set[int] = set()
    for layer, c in per_layer.items():
        ranked = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
        out.update(layer * num_experts + e for e, _ in ranked[:per])
    return out


def simulate(
    records,
    cache_size: int,
    num_experts: int,
    *,
    policy: str = "lru",
    pin_rows=(),
    row_bytes: int | None = None,
    phase: int = 0,
) -> SimResult:
    """Replay every record of ``phase`` through one pool; return the outcome."""
    sim = Sim(cache_size, num_experts, policy=policy, pin_rows=pin_rows,
              row_bytes=row_bytes)
    for ph, layer, ids in records:
        if ph != phase:
            continue
        sim.ensure(layer, ids)  # an empty local set still advances the clock
    return sim.result()


def curve(
    records,
    cache_sizes,
    num_experts: int,
    *,
    policies=("lru",),
    pin_fracs=(0.0,),
    warmup_frac: float = 1.0,
    pin_mode: str = "global",
    row_bytes: int | None = None,
    phase: int = 0,
) -> list[SimResult]:
    """Replay the trace once per (policy, cache_size, pin_frac).

    ``pin_frac`` is the share of the pool reserved for pinned rows in policy ``pin``;
    the pin set itself is learned once per ``cache_size`` (it may not exceed the pool).
    """
    out: list[SimResult] = []
    for C in cache_sizes:
        for policy in policies:
            if policy == "pin":
                for frac in pin_fracs:
                    P = int(round(C * frac))
                    if P <= 0 or P >= C:
                        continue
                    pin = learn_pin_set(records, P, num_experts, mode=pin_mode,
                                        warmup_frac=warmup_frac, phase=phase)
                    res = simulate(records, C, num_experts, policy="pin",
                                   pin_rows=pin, row_bytes=row_bytes, phase=phase)
                    res.policy = f"pin{frac:.0%}/{pin_mode}"
                    out.append(res)
            else:
                out.append(simulate(records, C, num_experts, policy=policy,
                                    row_bytes=row_bytes, phase=phase))
    return out

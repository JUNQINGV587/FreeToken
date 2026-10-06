"""Time-based decode share during long prefills (dsv41 port).

Background (dsv41 ct_vllm.py install_decode_share, D130)
--------------------------------------------------------
On an offload box every prefill step streams the routed experts and takes seconds, so a
request already decoding advances one token per such step and sees its rate collapse
(0.1 t/s measured). The remedy: after a step that carried prefill and took T seconds of
wall clock, the scheduler may run decode-only batches for up to f*T seconds before the
next prefill chunk goes. f=0.25 costs prefill ~20% wall time while the other streams keep
decoding at ~f/(1+f) of their solo rate (3.34 t/s measured).

Two dsv41 behaviours must travel together -- porting only half re-introduces bugs they
already hit:

  1. The waiting exception: a FRESH waiting request is never held back by the allowance
     (it rides along in the next prefill pass instead of queueing behind a long prompt).
     dsv41 measured decode share without this exception pushing short TTFT to 117 s
     (D130). A chunked continuation of an already-admitted prompt is NOT "waiting" for
     this test -- delaying its next chunk is exactly what the share is for.
  2. The allowance is capped (default 10 s) so one giant prefill step cannot bank an
     unbounded decode monopoly.

The allowance is real-time: it drains while decode-only steps run. One known divergence
from dsv41, accepted and bounded: their hook runs on every engine-core iteration, so idle
wall time drains the allowance there; this scheduler blocks on message receive while
idle, so the idle span lands in the next step's dt. If that step carries prefill the idle
span over-grants, but the cap bounds the excess at cap_s and a throttle only ever fires
while a decode is actually runnable (i.e. the box was not idle), so the practical effect
is nil.

This module holds the *policy* only -- a clock, a ledger and a decision -- so it can be
unit-tested without a Scheduler, an engine, or a GPU.
"""

from __future__ import annotations

import os
import time

SHARE_ENV = "FREETOKEN_DECODE_SHARE"
CAP_ENV = "FREETOKEN_DECODE_SHARE_CAP_S"
DEFAULT_CAP_S = 10.0


class DecodeSharePolicy:
    """Ledger for the decode-only allowance a prefill step earns.

    ``share`` is f (seconds of decode-only time granted per second of prefill-carrying
    step); <= 0 disables the policy and the scheduler keeps the historical prefill-first
    order bit-for-bit. ``cap_s`` bounds the banked allowance. ``clock`` is injectable for
    tests. Counters feed /v1/stats (``snapshot``).
    """

    def __init__(
        self,
        share: float,
        cap_s: float = DEFAULT_CAP_S,
        clock=time.monotonic,
    ) -> None:
        self.share = float(share)
        self.cap_s = float(cap_s)
        self._clock = clock
        self.allow_s = 0.0
        self._last_t: float | None = None
        # Kind of the step whose span is currently open (dsv41's DS["last_prefill"]).
        self._pending_prefill = False
        # Observability (dsv41's DS dict: allow/last_t/last_prefill/throttled, plus
        # cumulative grant/consume so a poller can read the allowance economy).
        self.throttled = 0
        self.granted_s = 0.0
        self.consumed_s = 0.0

    @classmethod
    def from_env(cls, environ=None) -> "DecodeSharePolicy | None":
        """Build from FREETOKEN_DECODE_SHARE / FREETOKEN_DECODE_SHARE_CAP_S.

        None (feature off) when the share is unset, unparseable, or <= 0 -- the caller
        then simply has no policy, mirroring how a missing DecodeInterleavePolicy means
        the historical order.
        """
        env = os.environ if environ is None else environ
        try:
            share = float(env.get(SHARE_ENV, "") or 0.0)
        except ValueError:
            share = 0.0
        if share <= 0:
            return None
        try:
            cap_s = float(env.get(CAP_ENV, "") or DEFAULT_CAP_S)
        except ValueError:
            cap_s = DEFAULT_CAP_S
        if cap_s <= 0:
            cap_s = DEFAULT_CAP_S
        return cls(share, cap_s)

    @property
    def enabled(self) -> bool:
        return self.share > 0

    def close_span(self) -> None:
        """Settle the wall-clock span since the last settlement into the ledger.

        Called at the top of every scheduler slot, BEFORE the throttle decision, so the
        allowance a just-finished prefill step earned is already visible when the next
        slot is decided (dsv41's hook updates DS in the same order). A span closed on a
        prefill-carrying step grants share*dt (up to the cap); any other span consumes,
        because decode-only time is what the allowance buys.
        """
        now = self._clock()
        if self._last_t is not None:
            dt = max(0.0, now - self._last_t)
            if self._pending_prefill:
                grant = min(self.cap_s - self.allow_s, self.share * dt)
                if grant > 0:
                    self.allow_s += grant
                    self.granted_s += grant
            else:
                spend = min(self.allow_s, dt)
                self.allow_s -= spend
                self.consumed_s += spend
        self._last_t = now

    def note_step(self, *, had_prefill: bool) -> None:
        """Mark the kind of the step just launched; its span settles at the next
        ``close_span``. Not called when nothing was runnable -- the span then settles
        against the previous step's kind (the idle over-grant this module's docstring
        bounds and accepts)."""
        self._pending_prefill = had_prefill

    def wants_decode(self, *, decoding: bool, waiting_new: bool) -> bool:
        """True when a decode-only batch may claim this slot instead of prefill.

        ``waiting_new`` must count only requests that have not started prefill (fresh
        arrivals), never chunked continuations -- holding back a fresh arrival is the
        D130 regression the exception exists to prevent.
        """
        if not self.enabled:
            return False
        return self.allow_s > 0 and decoding and not waiting_new

    def note_throttle(self) -> None:
        """Record that this slot's prefill was throttled in favour of decode."""
        self.throttled += 1

    def snapshot(self) -> dict:
        return {
            "share": self.share,
            "cap_s": self.cap_s,
            "allow_s": round(self.allow_s, 3),
            "throttled": self.throttled,
            "granted_s": round(self.granted_s, 3),
            "consumed_s": round(self.consumed_s, 3),
        }

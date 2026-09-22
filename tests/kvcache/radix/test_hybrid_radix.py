"""HybridRadixCache -- the GDN-snapshot currency.

What is genuinely different from the SWA sibling, and therefore what this module pins:

  * a snapshot is ONE opaque external slot id attached at a node's END boundary (point-attached,
    not a window), so a *prefix* of a snapshot-bearing node is not reusable at all;
  * ``match_prefix`` truncates the reusable prefix to the DEEPEST live snapshot;
  * ``split_at`` does not copy the snapshot -- the root-side half comes out empty;
  * ``insert`` dedups an existing snapshot, and refills a node whose snapshot was evicted;
  * ``evict_mamba`` counts SNAPSHOTS, tombstones internal nodes in place (keeping their KV), and
    frees KV eagerly when it takes a leaf, cascading through the tombstone leaves it exposes.

Page size: apart from the constructor's ``CHUNK_SIZE % page_size == 0`` assertion, the class is
page_size-agnostic -- ``align_down`` in ``_walk`` / ``insert`` is the only page-size-sensitive
code.  The scenarios therefore run at one page size (4, big enough that page keying differs from
token keying); the truncation test sweeps the page-size matrix.

Expectations come from the page-keyed reference model in ``model.py``; ``Session.do_*`` compares
every public result against it and ``Session.check()`` runs the invariant battery.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import pytest
import torch

from .adapters import iter_nodes, node_end_path
from .driver import CacheSpec, Session

PAGE = 4
SPEC = CacheSpec("hybrid", PAGE)


def page_ids(page_size: int, *labels: int) -> Tuple[int, ...]:
    """Concatenate whole pages.  Pages deliberately share leading tokens (the lead repeats every
    three labels), which is exactly the shape a token-keyed expectation gets wrong once
    page_size > 1: the tree's reuse unit is a whole page."""
    out = []
    for lab in labels:
        if page_size == 1:
            out.append(lab)
        else:
            out.extend([(lab % 3) + 1] + [7] * (page_size - 2) + [lab])
    return tuple(out)


def ids(*labels: int) -> Tuple[int, ...]:
    return page_ids(PAGE, *labels)


def donated(s: Session) -> int:
    """The GDN slot the harness donated on the most recent insert."""
    assert s.second is not None
    return s.second.handed[-1]


def events(s: Session):
    """Reference-model branch counters (``Session.events`` is documented but not implemented)."""
    return s.model.events


def full_size(s: Session) -> int:
    return s.model.counters()["full_evictable"]


def live_snapshots(s: Session) -> int:
    c = s.model.counters()
    return c["mamba_evictable"] + c["mamba_protected"]


@pytest.fixture
def hyb() -> Session:
    """A hybrid cache at page_size=4 with its reference model and slot ledgers."""
    return Session(SPEC)


def two_node_tree(s: Session) -> Tuple[Sequence[int], int, int]:
    """X=[page1,page2] with snapshot mx, its child Y=[page3,page4] with snapshot my.

    Built through the realistic lifecycle, so Y's insert reuses X's own slots and X ends up
    strictly older than Y (the deterministic clock makes that ordering reproducible)."""
    s.do_insert(ids(1, 2))
    mx = donated(s)
    s.do_request(ids(1, 2, 3, 4), prompt_pages=2)
    my = donated(s)
    s.check()
    return ids(1, 2, 3, 4), mx, my


# --------------------------------------------------------------------------- constructor
@pytest.mark.parametrize("page_size, ok", [(1, True), (4, True), (64, True),
                                           (3, False), (48, False), (128, False)])
def test_page_size_must_divide_chunk_size(page_size, ok):
    """Snapshots land on x CHUNK_SIZE boundaries, so a page must not straddle one."""
    from freetoken.kernel.fla.chunk import CHUNK_SIZE
    from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache

    assert (CHUNK_SIZE % page_size == 0) is ok, "the parameter table assumed CHUNK_SIZE == 64"
    if not ok:
        with pytest.raises(AssertionError,
                           match=rf"CHUNK_SIZE\({CHUNK_SIZE}\) % page_size\({page_size}\)"):
            HybridRadixCache(torch.device("cpu"), page_size=page_size)
        return
    c = HybridRadixCache(torch.device("cpu"), page_size=page_size)
    assert c.page_size == page_size
    assert (c.full_evictable_size, c.mamba_evictable_size) == (0, 0)
    cold = c.match_prefix(torch.tensor([1] * page_size, dtype=torch.int64))
    assert (cold.cached_len, cold.mamba_value) == (0, None)


# --------------------------------------------------------------------------- insert / match
def test_insert_attaches_snapshot_and_match_restores_it(hyb):
    slots = hyb.kv.take(2 * PAGE)
    got, exp = hyb.do_insert(ids(1, 2), slots=slots)
    assert (got.matched_len, got.second_exists) == (0, False)
    assert exp.adopted == slots and exp.dups == []
    hyb.check()

    m, _ = hyb.do_match(ids(1, 2))
    assert m.cached_len == 2 * PAGE
    assert m.indices == slots
    assert m.second == donated(hyb)
    assert events(hyb)["insert.snapshot_attach"] == 1
    assert live_snapshots(hyb) == 1


def test_deepest_snapshot_wins_and_unmatched_suffix_is_dropped(hyb):
    chain, mx, my = two_node_tree(hyb)
    assert mx != my

    m, _ = hyb.do_match(chain)
    assert (m.cached_len, m.second) == (4 * PAGE, my)          # the deeper snapshot supersedes
    # a query running past the end of the tree still resumes from the deepest snapshot
    m, _ = hyb.do_match(chain + ids(5))
    assert (m.cached_len, m.second) == (4 * PAGE, my)
    assert events(hyb)["match.snapshot_truncation"] == 0        # nothing was truncated: 4P is the end
    hyb.check()


def test_split_leaves_the_snapshot_on_the_suffix_half(hyb):
    """``split_at`` does not copy ``mamba_value``, so the root-side half of a split node has no
    snapshot and a match that ends there falls back to the nearest ancestor that does."""
    chain, mx, my = two_node_tree(hyb)

    m, _ = hyb.do_match(ids(1, 2, 3, 5))       # diverges inside Y -> Y splits into [3] + [4]
    assert events(hyb)["node.split"] == 1
    assert events(hyb)["match.snapshot_truncation"] == 1
    assert (m.cached_len, m.second) == (2 * PAGE, mx)           # truncated back to X's snapshot
    hyb.check()

    m, _ = hyb.do_match(chain)                 # the suffix half kept my, so the deep hit survives
    assert (m.cached_len, m.second) == (4 * PAGE, my)
    assert live_snapshots(hyb) == 2
    assert full_size(hyb) == 4 * PAGE                          # a split moves no KV


def test_prefix_of_a_snapshot_node_is_not_reusable(hyb):
    """The snapshot is point-attached at the node's end boundary: matching an interior page
    boundary of that node yields nothing at all, not a shortened hit."""
    slots = hyb.kv.take(4 * PAGE)
    hyb.do_insert(ids(1, 2, 3, 4), slots=slots)
    ms = donated(hyb)
    hyb.check()

    m, _ = hyb.do_match(ids(1, 2))             # splits the node; the [1,2] half owns no snapshot
    assert (m.cached_len, m.indices, m.second) == (0, [], None)
    assert events(hyb)["node.split"] == 1
    assert events(hyb)["match.no_snapshot"] == 1

    m, _ = hyb.do_match(ids(1, 2, 3, 4))       # ... while the full prefix is still a hit
    assert (m.cached_len, m.indices, m.second) == (4 * PAGE, slots, ms)
    assert full_size(hyb) == 4 * PAGE
    hyb.check()


def test_insert_dedups_and_the_caller_frees_the_donated_slot(hyb):
    hyb.do_insert(ids(1, 2))
    mx = donated(hyb)

    slots = hyb.kv.take(2 * PAGE)
    got, exp = hyb.do_insert(ids(1, 2), slots=slots)            # same boundary, second donation
    assert (got.matched_len, got.second_exists) == (2 * PAGE, True)
    assert exp.dups == slots and exp.adopted == []               # caller keeps every duplicate page
    assert events(hyb)["insert.snapshot_dedup"] == 1
    assert live_snapshots(hyb) == 1                              # still exactly one snapshot
    hyb.check()

    m, _ = hyb.do_match(ids(1, 2))
    assert m.second == mx                                        # the original snapshot is kept
    assert donated(hyb) in hyb.second.free                       # the loser was handed back


@pytest.mark.parametrize("P", [1, 4, 64], ids=["p1", "p4", "p64"])
def test_insert_stores_whole_pages_only(P):
    """``align_down`` governs both ends: fewer tokens than a page reaches the root (which cannot
    hold a snapshot, hence ``exists=True``), and a ragged tail is left with the caller."""
    s = Session(CacheSpec("hybrid", P))
    stub = page_ids(P, 1)[: P - 1]                               # () at P == 1
    got, _ = s.do_insert(stub)
    assert (got.matched_len, got.second_exists) == (0, True)
    assert full_size(s) == 0 and live_snapshots(s) == 0
    s.check()

    slots = s.kv.take(2 * P + (P - 1))
    got, exp = s.do_insert(page_ids(P, 1, 2) + page_ids(P, 3)[: P - 1], slots=slots)
    assert (got.matched_len, got.second_exists) == (0, False)
    assert exp.adopted == slots[: 2 * P]                         # the tail was never adopted
    s.check()

    m, _ = s.do_match(page_ids(P, 1, 2))
    assert (m.cached_len, m.indices) == (2 * P, slots[: 2 * P])
    assert full_size(s) == 2 * P


# --------------------------------------------------------------------------- evict_mamba
def test_evict_mamba_tombstones_an_internal_node_and_keeps_its_kv(hyb):
    chain, mx, my = two_node_tree(hyb)

    hyb.do_evict_second(1)                     # X is the LRU snapshot and is internal
    assert events(hyb)["evict_mamba.tombstone_in_place"] == 1
    assert mx in hyb.second.free and hyb.kv.free == set()        # no KV was reclaimed
    assert full_size(hyb) == 4 * PAGE and live_snapshots(hyb) == 1
    hyb.check()

    m, _ = hyb.do_match(ids(1, 2))             # the tombstoned boundary is no longer resumable
    assert (m.cached_len, m.second) == (0, None)
    m, _ = hyb.do_match(chain)                 # the descendant snapshot is untouched
    assert (m.cached_len, m.second) == (4 * PAGE, my)


def test_insert_refills_a_tombstoned_node(hyb):
    two_node_tree(hyb)
    hyb.do_evict_second(1)                     # tombstone X (KV kept, snapshot gone)
    hyb.check()

    slots = hyb.kv.take(2 * PAGE)
    got, exp = hyb.do_insert(ids(1, 2), slots=slots)
    assert (got.matched_len, got.second_exists) == (2 * PAGE, False)   # attaches, does not dedup
    assert exp.dups == slots                                          # KV stays canonical
    assert events(hyb)["insert.snapshot_attach"] == 3
    hyb.check()

    m, _ = hyb.do_match(ids(1, 2))
    assert (m.cached_len, m.second) == (2 * PAGE, donated(hyb))
    assert live_snapshots(hyb) == 2


def test_evict_mamba_on_a_leaf_frees_kv_and_cascades_through_tombstones(hyb):
    chain, _mx, _my = two_node_tree(hyb)
    hyb.do_evict_second(1)                     # X -> KV-only tombstone
    hyb.do_evict_second(1)                     # Y is now the only snapshot node, and a leaf
    assert events(hyb)["evict_mamba.leaf_free"] == 1
    assert events(hyb)["evict_mamba.cascade"] == 1                    # X reclaimed in the same call
    hyb.check()

    assert full_size(hyb) == 0 and live_snapshots(hyb) == 0
    assert hyb.kv.in_use() == set() and hyb.second.in_use() == set()
    m, _ = hyb.do_match(chain)
    assert (m.cached_len, m.second) == (0, None)


def test_evict_mamba_counts_snapshots_not_tokens(hyb):
    """A one-page node holds PAGE tokens but one snapshot; ``evict_mamba(2)`` must take two
    snapshots, which a token-counting loop would never do."""
    hyb.do_insert(ids(1))
    hyb.do_request(ids(1, 2), prompt_pages=1)
    hyb.do_request(ids(1, 2, 3), prompt_pages=2)
    hyb.check()
    assert live_snapshots(hyb) == 3 and full_size(hyb) == 3 * PAGE

    hyb.do_evict_second(2)
    assert len(hyb.second.free) == 2                                  # exactly two snapshots
    assert events(hyb)["evict_mamba.tombstone_in_place"] == 2         # both were internal
    assert hyb.kv.free == set() and full_size(hyb) == 3 * PAGE        # no KV touched
    assert live_snapshots(hyb) == 1
    hyb.check()


def test_snapshot_slots_are_conserved_across_eviction_waves(hyb):
    """Every donated slot comes back exactly once (the ledger raises on a double free), and
    draining the snapshot currency drains the KV with it via the leaf cascade."""
    n = 8
    hyb.do_insert(ids(1))
    for k in range(2, n + 1):
        hyb.do_request(ids(*range(1, k + 1)), prompt_pages=k - 1)
    hyb.check()
    assert live_snapshots(hyb) == n and full_size(hyb) == n * PAGE
    handed = list(hyb.second.handed)

    for guard in range(10):
        if live_snapshots(hyb) == 0:
            break
        hyb.do_evict_second(3)
        hyb.check()
    else:
        pytest.fail(f"evict_mamba did not converge: {live_snapshots(hyb)} snapshot(s) left")

    assert sorted(hyb.second.free) == sorted(handed)
    assert hyb.kv.in_use() == set() and full_size(hyb) == 0
    assert hyb.do_match(ids(*range(1, n + 1)))[0].cached_len == 0


# --------------------------------------------------------------------------- evict_full / locks
def test_evict_full_takes_the_leaf_snapshot_and_leaves_the_ancestor_usable(hyb):
    chain, mx, my = two_node_tree(hyb)

    hyb.do_evict_full(2 * PAGE)                # the only unlocked leaf is Y
    assert my in hyb.second.free and mx not in hyb.second.free
    assert events(hyb)["evict_full.cascade"] == 0   # X still owns a snapshot, so it is not reclaimed
    assert full_size(hyb) == 2 * PAGE and live_snapshots(hyb) == 1
    hyb.check()

    m, _ = hyb.do_match(chain)                 # the request re-hits at the surviving boundary
    assert (m.cached_len, m.second) == (2 * PAGE, mx)


def test_evict_full_cascades_through_an_exposed_tombstone_leaf(hyb):
    """A KV-only tombstone leaf is reclaimed eagerly in the same ``evict_full`` that exposes it,
    so ``evict_full(2*PAGE)`` returns twice that many tokens."""
    two_node_tree(hyb)
    hyb.do_evict_second(1)                     # X -> tombstone
    hyb.do_evict_full(2 * PAGE)                # evicting Y exposes X and reclaims it too
    assert events(hyb)["evict_full.cascade"] == 1
    hyb.check()

    assert full_size(hyb) == 0 and live_snapshots(hyb) == 0
    assert hyb.kv.in_use() == set() and hyb.second.in_use() == set()


def test_lock_pins_the_snapshot_and_the_whole_kv_path(hyb):
    """``inc_lock`` takes a mamba ref on the matched node and a full ref up to the root, so
    neither currency can evict it -- but locking a descendant does NOT protect an ancestor's
    snapshot (``full_ref >= mamba_ref``, not the other way round)."""
    chain, mx, _my = two_node_tree(hyb)
    held = hyb.do_lock(chain)
    assert held is not None
    hyb.check()

    hyb.do_evict_full(4 * PAGE)                # Y is locked, X is internal -> nothing is evictable
    assert hyb.kv.free == set() and full_size(hyb) == 0        # all 4 pages are protected now

    hyb.do_evict_second(1)                     # X's snapshot is unprotected -> tombstoned in place
    assert hyb.second.free == {mx}
    hyb.do_evict_second(1)                     # Y's snapshot is pinned by the lock
    assert hyb.second.free == {mx}
    hyb.check()

    hyb.do_unlock(held)
    assert full_size(hyb) == 4 * PAGE
    hyb.do_evict_full(2 * PAGE)                # Y is evictable again; X cascades behind it
    assert events(hyb)["evict_full.cascade"] == 1
    assert hyb.kv.in_use() == set() and hyb.second.in_use() == set()
    hyb.check()


# --------------------------------------------------------------------------- validate-refresh
def _live_boundaries(s: Session) -> set:
    """Boundary lengths (end-path token counts) of every node holding a live snapshot."""
    return {len(node_end_path(n)) for n, _ in iter_nodes(s.ad.root)
            if n.mamba_value is not None}


def test_dedup_refresh_pins_survival_at_forced_eviction(hyb):
    """Amended Fix-3, insert dedup branch: a dedup hit re-stamps the snapshot's OWN LRU
    (snapshot_lru), which no later walk can erase, so the just-validated boundary outlives
    a peer that is newer by the walk tic but was last validated earlier. Pre-fix
    evict_mamba ranks by timestamp alone and the walked-later peer looks younger, so K is
    the stalest candidate and dies (fails-before)."""
    hyb.do_insert(ids(1, 2))                   # X=[p1,p2], mx
    hyb.do_request(ids(1, 2, 3), prompt_pages=2)   # Y=[p3], my -- K
    my = donated(hyb)
    hyb.do_evict_second(1)                     # tombstone X: its fresh walk tic must not shield K
    hyb.do_request(ids(9), prompt_pages=1)     # off-path P=[p9], mp (stale-validated decoy)
    mp = donated(hyb)
    hyb.do_insert(ids(1, 2, 3))                # dedup at K: re-validated NOW
    hyb.do_insert(ids(9, 10))                  # walks P late; Q=[p10], mq (newest of all)
    hyb.check()

    hyb.do_evict_second(1)                     # pool pressure: one snapshot must go
    hyb.check()
    assert mp in hyb.second.free and my not in hyb.second.free   # the decoy died, K survived
    m, _ = hyb.do_match(ids(1, 2, 3))
    assert m.second == my                      # K is still restorable
    hyb.check()


def test_match_use_refresh_pins_survival_at_forced_eviction(hyb):
    """Amended Fix-3, match_prefix: returning a live snapshot is a use of the reuse point;
    the returned node's snapshot_lru re-stamp out-ages a peer that is newer by the walk tic
    but stale by validation. Pre-fix the walked-later peer ranks younger by timestamp and
    K dies (fails-before)."""
    hyb.do_insert(ids(1, 2))                   # X, mx
    hyb.do_request(ids(1, 2, 3), prompt_pages=2)   # Y=[p3], my -- K
    my = donated(hyb)
    hyb.do_evict_second(1)                     # tombstone X
    hyb.do_request(ids(9), prompt_pages=1)     # off-path P=[p9], mp
    mp = donated(hyb)
    m, _ = hyb.do_match(ids(1, 2, 3))          # match-use of K: re-validated NOW
    assert m.second == my
    hyb.do_insert(ids(9, 10))                  # walks P AFTER K's refresh (stale-ts decoy)
    hyb.check()

    hyb.do_evict_second(1)
    hyb.check()
    assert mp in hyb.second.free and my not in hyb.second.free   # the decoy died, K survived
    m, _ = hyb.do_match(ids(1, 2, 3))
    assert m.second == my
    hyb.check()


def test_steady_state_churn_pins_reuse_to_live_boundaries(hyb):
    """Amended Fix-3 steady state: 6 cycles of (exact-prefix dedup request + one-unique-finish
    extend + match) with an off-path branch P born mid-run and never re-walked, plus one
    forced eviction at the end of cycles 4-6. Post-fix the victims leave FIFO by last
    validation: B0, then P (validated once, at its birth), then B1 -- deterministic by
    boundary key. Pre-fix P's stale walk-tic timestamp makes IT the first victim and B0
    dies in its place (fails-before). Reuse degrades to the deepest surviving boundary
    and never to 0."""
    chain = ids(1, 2)
    hyb.do_insert(chain)                       # B0 (2 pages)
    slot_of = {2 * PAGE: donated(hyb)}
    victims = []
    for k in range(3, 9):                      # 6 cycles, one fresh boundary each
        hyb.do_request(chain, prompt_pages=len(chain) // PAGE)   # match + boundary dedup
        chain = chain + ids(k)
        hyb.do_request(chain, prompt_pages=len(chain) // PAGE)   # match + unique finish insert
        slot_of[len(chain)] = donated(hyb)
        if k == 3:
            hyb.do_request(ids(9), prompt_pages=1)   # off-path P: validated once, never re-walked
            slot_of[1 * PAGE] = donated(hyb)
        if 4 <= k <= 6:
            before = set(hyb.second.free)
            hyb.do_evict_second(1)             # pool pressure: one snapshot must go
            slot_owner = {v: length for length, v in slot_of.items()}
            victims.append(slot_owner[(hyb.second.free - before).pop()])
    hyb.check()

    assert victims == [2 * PAGE, 1 * PAGE, 3 * PAGE]   # FIFO by last validation: B0, P, B1
    hyb.do_request(chain, prompt_pages=len(chain) // PAGE)       # the final re-validation
    m, _ = hyb.do_match(chain)
    assert m.cached_len == len(chain) and m.second is not None   # reuse never drops to 0
    assert _live_boundaries(hyb) == {4 * PAGE, 5 * PAGE, 6 * PAGE, 7 * PAGE, 8 * PAGE}
    hyb.check()


def test_exhaustion_leaves_the_deepest_validated_boundary(hyb):
    """Brief-required exhaustion pin: keep forcing evictions until exactly one live snapshot
    remains. Post-fix the victims leave FIFO by last validation (B0, B1, B2, then the
    never-re-walked decoy P) and the survivor is the DEEPEST boundary -- the just-validated
    tip is never a victim and the final match keeps length > 0. Pre-fix the decoy's stale
    walk-tic timestamp makes it the FIRST victim (fails-before)."""
    chain = ids(1, 2)
    hyb.do_insert(chain)                       # B0
    slot_of = {2 * PAGE: donated(hyb)}
    for k in range(3, 6):                      # 3 cycles -> B1, B2, B3
        hyb.do_request(chain, prompt_pages=len(chain) // PAGE)
        chain = chain + ids(k)
        hyb.do_request(chain, prompt_pages=len(chain) // PAGE)
        slot_of[len(chain)] = donated(hyb)
    hyb.do_request(ids(9), prompt_pages=1)     # off-path P, validated once, never re-walked
    slot_of[1 * PAGE] = donated(hyb)
    hyb.do_request(chain, prompt_pages=len(chain) // PAGE)       # final re-validation of B3
    hyb.check()

    victims = []
    while live_snapshots(hyb) > 1:
        before = set(hyb.second.free)
        hyb.do_evict_second(1)
        hyb.check()
        slot_owner = {v: length for length, v in slot_of.items()}
        victims.append(slot_owner[(hyb.second.free - before).pop()])
    assert victims == [2 * PAGE, 3 * PAGE, 4 * PAGE, 1 * PAGE]   # B0, B1, B2, then P
    m, _ = hyb.do_match(chain)
    assert m.cached_len == len(chain) > 0                       # reuse never 0 at the floor
    assert m.second == slot_of[5 * PAGE]                        # the deepest tip survived
    hyb.check()


# ------------------------------------------------- interleave: two disjoint conversations
def test_interleaved_conversations_wash_out_the_older_trace(hyb):
    """Two disjoint conversations under pool pressure -- the orchestrator/subagent interleave.

    Production shape (2026-09 report): conversation A finishes a turn (its tip + grid donate
    snapshots), conversation B then commits one snapshot per prefill chunk, and each commit
    needing a slot forces ensure_mamba_slots -> evict_mamba, which is FIFO by snapshot_lru
    with no notion of a recently finished conversation. A's whole trace is strictly older
    than every B boundary, so B's commits evict A's boundaries first (tip included), and A's
    leaf eviction cascades its tombstoned ancestors' KV away too. A's next match collapses
    to cached_len 0 -- a full re-prefill despite both contexts fitting the KV pool, because
    the binding currency is GDN snapshot slots, not KV tokens.

    This test PINS THE CURRENT POLICY as the measured baseline. A future protection policy
    (pin the tip of a recently finished conversation, or evict intermediate grid nodes
    first) must flip the A-resume assertion to retention; the victim order below is what it
    has to change.
    """
    # Conversation A: two boundaries, tip at 4 pages.
    hyb.do_insert(ids(1, 2))                                     # A0
    slot_names = {donated(hyb): "A0"}
    hyb.do_request(ids(1, 2, 3, 4), prompt_pages=2)              # A tip (A1)
    slot_names[donated(hyb)] = "A1"

    # Conversation B: disjoint prefix (the root is the only shared node), three boundaries,
    # one forced eviction after each commit -- what the scheduler does whenever the free-slot
    # count drops below what the next commit needs.
    victims = []
    b_chain: Tuple[int, ...] = ()
    for k, lab in enumerate((9, 10, 11)):
        b_chain = b_chain + ids(lab)
        hyb.do_insert(b_chain)
        slot_names[donated(hyb)] = f"B{k}"
        before = set(hyb.second.free)
        hyb.do_evict_second(1)                                   # pool pressure: a slot must go
        victims.append(slot_names[(hyb.second.free - before).pop()])
    hyb.check()

    # Victims leave FIFO by last validation across conversations: A's older trace dies first,
    # tip included, before B's oldest boundary.
    assert victims == ["A0", "A1", "B0"]

    # A resumes: no live snapshot anywhere on its path -> full re-prefill.
    m, _ = hyb.do_match(ids(1, 2, 3, 4))
    assert (m.cached_len, m.second) == (0, None)

    # B resumes fine, and A's KV is gone too (its leaf eviction cascaded the tombstones up):
    # the only evictable KV left is B's three 1-page nodes (B0 tombstoned, KV kept).
    assert full_size(hyb) == 3 * PAGE
    m, _ = hyb.do_match(b_chain)
    assert m.cached_len == 3 * PAGE and m.second == [s for s, n in slot_names.items() if n == "B2"][0]
    hyb.check()


def test_interleave_with_enough_snapshot_pool_keeps_both_traces(hyb):
    """Contrast pin: the same interleave with NO slot pressure (a fundable snapshot cache --
    the --linear-state-cache-ratio lever). Both traces keep their tips and both conversations
    resume from their deepest boundary; the wash-out above is a pool-size effect, not a match
    or insert defect."""
    hyb.do_insert(ids(1, 2))                                     # A0
    hyb.do_request(ids(1, 2, 3, 4), prompt_pages=2)              # A tip (A1)
    a_tip = donated(hyb)
    b_chain: Tuple[int, ...] = ()
    for lab in (9, 10, 11):
        b_chain = b_chain + ids(lab)
        hyb.do_insert(b_chain)
    b_tip = donated(hyb)
    hyb.check()

    m, _ = hyb.do_match(ids(1, 2, 3, 4))
    assert (m.cached_len, m.second) == (4 * PAGE, a_tip)
    m, _ = hyb.do_match(b_chain)
    assert (m.cached_len, m.second) == (3 * PAGE, b_tip)
    hyb.check()

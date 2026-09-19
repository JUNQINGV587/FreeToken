"""Hybrid (full-attn KV + GDN linear-state) radix cache.

A SEPARATE class from ``RadixPrefixCache`` (Option C) that REUSES the shared ``RadixTreeNode``
and walk/split logic, so the production KV radix is untouched (zero risk to non-hybrid models).
It adds a second "currency": an optional GDN state snapshot (``node.mamba_value`` = a
LinearStatePool slot id) attached at chunk/page-aligned boundary nodes, with its own LRU
eviction. Mirrors sglang ``MambaRadixCache`` (donate-not-copy, dual eviction, internal-node
tombstone, ``full_ref >= mamba_ref``) on FreeToken's tree.

Currency seam: the secondary value + its eviction is the slot a future SWA component plugs
into. This class is pool-agnostic -- it stores/returns slot ids and KV page indices; the
caller (CacheManager / scheduler) does the actual LinearStatePool / KV-pool free.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import torch

from freetoken.utils import align_down

from .base import BaseCacheHandle, HostTierKeyCollision
from .radix_cache import RadixTreeNode, _get_key_fn


@dataclass(frozen=True)
class HybridCacheHandle(BaseCacheHandle):
    """Lock handle for a matched hybrid prefix: the matched node (lock target) + the reusable
    KV page indices. ``cached_len`` is already truncated to the deepest live-snapshot boundary.
    Plugs into PrefillAdder (reads ``.cached_len`` / ``.get_matched_indices()``) like the plain
    RadixCacheHandle; the restore slot rides on ``MatchResult.mamba_value``."""

    node: RadixTreeNode
    kv_indices: torch.Tensor

    def get_matched_indices(self) -> torch.Tensor:
        return self.kv_indices


class HybridMatch(NamedTuple):
    kv_indices: torch.Tensor      # reused KV page indices for [0:cached_len)
    cached_len: int               # truncated to the deepest LIVE-snapshot boundary
    mamba_value: Optional[int]    # GDN snapshot slot to restore from (None = cold start)
    node: RadixTreeNode           # the matched node (lock target)


class EvictResult(NamedTuple):
    kv_indices: torch.Tensor      # KV page indices to free
    mamba_slots: List[int]        # GDN state slots to free


class HybridRadixCache:
    def __init__(self, device: torch.device, page_size: int, *,
                 host_spill=None, host_materialize=None) -> None:
        """``host_spill``/``host_materialize`` add a THIRD node state -- host-resident -- on top
        of GPU-resident and tombstone, so KV can leave the GPU pool without leaving the tree.

        * ``host_spill(node) -> key | None``: copy ``node.value``'s pages into the KV host tier
          and return the tier key, or None to refuse (tier full, unsupported page). The caller
          still frees the pages; the node keeps its token span and stays in the tree.
        * ``host_materialize(node, key) -> Tensor | None``: allocate fresh GPU pages and copy
          the tier's copy back into them, returning one page index per token of ``node.length``
          (same layout as ``value``), or None to give up.

        Both default to None: with no host tier the cache behaves exactly as before, because
        ``host_value`` is never set. ``host_drop`` must be called when the tier discards a
        spilled page (its LRU): the node's KV is then gone for good and it has to be unlinked.
        """
        from freetoken.kernel.fla.chunk import CHUNK_SIZE
        # Snapshots land on ×CHUNK_SIZE boundaries; require them to be page-aligned so the KV
        # node boundary and the GDN-state boundary coincide (page_size in {1,2,4,8,16,32,64}).
        assert CHUNK_SIZE % page_size == 0, (
            f"hybrid_radix needs CHUNK_SIZE({CHUNK_SIZE}) % page_size({page_size}) == 0"
        )
        self.device = device
        self.page_size = page_size
        self.key_fn = _get_key_fn(page_size)
        self.empty = torch.empty(0, dtype=torch.int32, device=device)
        self.root = RadixTreeNode(self.key_fn)
        self.root.set_key_value(self.empty, self.empty)
        self.root.ref_count = 1  # root is always protected
        self.full_evictable = 0
        self.full_protected = 0
        self.mamba_evictable = 0     # number of live, unlocked snapshots
        self.mamba_protected = 0
        self.host_spill = host_spill
        self.host_materialize = host_materialize
        # Tokens whose KV lives in the host tier. Kept OUT of full_evictable: their pages are
        # back in the pool's free list, so counting them there would double-book the page.
        self.host_resident = 0
        self._host_nodes: dict = {}  # tier key -> node, for host_drop / integrity

    # ---------------------------------------------------------------- match / insert
    def match_prefix(self, input_ids: torch.Tensor) -> HybridMatch:
        """Match the token prefix, then truncate the reusable length to the deepest node on
        the path that still owns a LIVE snapshot (a continuation can only resume the GDN
        recurrence from a checkpointed boundary)."""
        node, _ = self._walk(input_ids)
        # walk up to the deepest node whose END boundary has a live snapshot
        cur, end_len = node, self._path_len(node)
        while not cur.is_root():
            if cur.mamba_value is not None:
                return HybridMatch(self._collect_kv(cur), end_len, cur.mamba_value, cur)
            end_len -= cur.length
            cur = cur.parent
        return HybridMatch(self.empty, 0, None, self.root)

    def insert(self, input_ids: torch.Tensor, kv_indices: torch.Tensor,
               mamba_value: int) -> Tuple[int, bool]:
        """Insert the committed KV prefix and DONATE ``mamba_value`` at the (page-aligned) end
        boundary node. Returns (matched_prefix_len, mamba_exist). If the boundary node already
        owns a live snapshot, returns mamba_exist=True and does not attach (caller frees the
        donated slot -- dedup)."""
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, kv_indices = input_ids[:insert_len], kv_indices[:insert_len]
        node, prefix_len = self._walk(input_ids)
        if prefix_len != insert_len:
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], kv_indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.full_evictable += new_node.length
            node = new_node
        if node.is_root():
            return prefix_len, True   # root can't hold a snapshot; report exist so caller frees it
        if node.mamba_value is not None:
            return prefix_len, True                 # dedup: caller frees its donated slot
        node.mamba_value = mamba_value              # fills a fresh node or a tombstone
        if node.mamba_ref_count == 0:
            self.mamba_evictable += 1
        return prefix_len, False

    # ---------------------------------------------------------------- locking (dual)
    def inc_lock(self, node: RadixTreeNode) -> None:
        """Protect a matched node's snapshot (mamba ref on the node) and its KV path
        (full ref node..root). Enforces full_ref >= mamba_ref: using a snapshot at N pins the
        whole root..N KV chain."""
        if node.mamba_value is not None:
            if node.mamba_ref_count == 0:
                self.mamba_evictable -= 1
                self.mamba_protected += 1
            node.mamba_ref_count += 1
        cur = node
        while not cur.is_root():
            if cur.ref_count == 0:
                self.full_evictable -= cur.length
                self.full_protected += cur.length
            cur.ref_count += 1
            cur = cur.parent

    def dec_lock(self, node: RadixTreeNode) -> None:
        if node.mamba_value is not None and node.mamba_ref_count > 0:
            node.mamba_ref_count -= 1
            if node.mamba_ref_count == 0:
                self.mamba_evictable += 1
                self.mamba_protected -= 1
        cur = node
        while not cur.is_root():
            cur.ref_count -= 1
            assert cur.ref_count >= 0
            if cur.ref_count == 0:
                self.full_evictable += cur.length
                self.full_protected -= cur.length
            cur = cur.parent

    # ---------------------------------------------------------------- eviction (dual)
    def evict_full(self, num_tokens: int) -> EvictResult:
        """Evict KV tokens by LRU over UNLOCKED LEAF nodes (an internal node's KV is a prefix
        dependency for all descendants). Frees each evicted node's snapshot too.

        With ``host_spill`` set, a leaf that owns a LIVE snapshot is SPILLED instead: its pages
        are returned (the caller frees them) while the node stays in the tree, host-resident."""
        leaves = [n for n in self._leaves()
                  if n.ref_count == 0 and n.host_value is None]
        heapq.heapify(leaves)
        kv, mamba, freed = [], [], 0
        while freed < num_tokens and leaves:
            node = heapq.heappop(leaves)
            if node.ref_count != 0 or not node.is_leaf() or node.is_root():
                continue
            if node.host_value is not None:
                continue
            if self._spill(node, kv):
                freed += node.length
                continue
            freed += node.length
            kv.append(node.value)
            self.full_evictable -= node.length
            self._free_node_mamba(node, mamba)
            parent, casc = self._cascade_tombstone_leaves(self._unlink(node), kv)
            freed += casc
            if parent.is_leaf() and parent.ref_count == 0 and not parent.is_root():
                heapq.heappush(leaves, parent)
        return EvictResult(torch.cat(kv) if kv else self.empty, mamba)

    def evict_mamba(self, num: int) -> EvictResult:
        """Evict GDN snapshots by LRU over UNLOCKED snapshot-bearing nodes -- internal nodes
        too. Internal node -> TOMBSTONE (free the slot, keep KV + children). Leaf node -> free
        both KV and slot and unlink, then cascade-delete any KV-only tombstone leaves it exposes
        upward (so a leaf always carries a live snapshot -- mirrors sglang)."""
        cands = [n for n in self._snapshot_nodes()
                 if n.mamba_ref_count == 0 and n.host_value is None]
        heapq.heapify(cands)
        kv, mamba, freed = [], [], 0
        while freed < num and cands:
            node = heapq.heappop(cands)
            if node.mamba_value is None or node.mamba_ref_count != 0 or node.is_root():
                continue
            if node.host_value is not None:
                continue  # its KV is in host; freeing the snapshot would strand it
            if node.is_leaf() and node.ref_count == 0:
                kv.append(node.value)
                self.full_evictable -= node.length
                self._free_node_mamba(node, mamba)
                freed += 1
                self._cascade_tombstone_leaves(self._unlink(node), kv)
            else:
                self._free_node_mamba(node, mamba)  # tombstone internal (or locked-KV) node
                freed += 1
        return EvictResult(torch.cat(kv) if kv else self.empty, mamba)

    @property
    def full_evictable_size(self) -> int:
        return self.full_evictable

    @property
    def mamba_evictable_size(self) -> int:
        return self.mamba_evictable

    @property
    def host_resident_size(self) -> int:
        """Tokens whose KV lives in the host tier (their GPU pages are back in the free list)."""
        return self.host_resident

    @property
    def size_info(self):
        """KV-page currency, for code that reads a BasePrefixCache size_info (metrics/usage).
        The GDN-snapshot currency is reported via mamba_evictable_size."""
        from .base import SizeInfo
        return SizeInfo(evictable_size=self.full_evictable, protected_size=self.full_protected)

    def check_integrity(self) -> None:
        # Structural: every snapshot-bearing node holds a real slot id; ref counts non-negative.
        # (KV/page conservation is checked by CacheManager.check_integrity.)
        for n in self._snapshot_nodes():
            assert n.mamba_value is not None and n.mamba_ref_count >= 0 and n.ref_count >= 0
        resident = 0
        for key, n in self._host_nodes.items():
            assert n.host_value == key and len(n.value) == 0, "host-resident node still has pages"
            # A host-resident node keeps its KV usable only if it can still resume the GDN
            # recurrence (live snapshot) and its whole prefix is resident (leaf).
            assert n.mamba_value is not None, "host-resident node lost its snapshot"
            assert n.is_leaf(), "host-resident node has children that need its KV"
            resident += n.length
        assert resident == self.host_resident, (
            f"host_resident({self.host_resident}) != sum of host-resident nodes({resident})"
        )

    def host_drop(self, host_key) -> bool:
        """The host tier discarded a spilled page (its own LRU). That node's KV is gone for
        good, so unlink it -- which is also what lets its prefix become evictable again.
        Returns False for an unknown key."""
        node = self._host_nodes.pop(host_key, None)
        if node is None:
            return False
        node.host_value = None
        self.host_resident -= node.length
        self._cascade_tombstone_leaves(self._unlink(node), [])
        return True

    # ---------------------------------------------------------------- helpers
    def _spill(self, node: RadixTreeNode, kv_out: List[torch.Tensor]) -> bool:
        """Move one leaf's KV into the host tier and hand its pages back. Only a node that owns
        a LIVE snapshot is spittable: a hybrid prefix without a snapshot can never be resumed,
        so spilling it would write KV that no match can ever read.

        The ancestors need no pin: eviction only ever takes LEAVES, and a spilled node keeps
        its child slot in the tree, so its whole prefix stays non-evictable by construction.
        """
        if self.host_spill is None or node.mamba_value is None:
            return False
        key = self.host_spill(node)
        if key is None:
            return False
        if key in self._host_nodes:
            # Checked before the pages leave: two nodes sharing a tier key would make whichever
            # one materializes second read the other prefix's bytes.
            raise HostTierKeyCollision(
                f"host tier key {key!r} already names another node; a tier key must be unique "
                "per resident copy"
            )
        kv_out.append(node.value)          # the caller returns these pages to the pool
        self.full_evictable -= node.length
        self.host_resident += node.length
        node.set_host_resident(key)
        self._host_nodes[key] = node
        return True

    def _materialize(self, node: RadixTreeNode) -> bool:
        """Bring a host-resident node back into the GPU pool before it is read (a walk reaches
        it) or split. False = the tier could not produce the pages."""
        key = node.host_value
        assert key is not None
        if self.host_materialize is None:
            return False
        new_indices = self.host_materialize(node, key)
        if new_indices is None:
            return False
        assert len(new_indices) == node.length, (
            f"materialize returned {len(new_indices)} indices for {node.length} tokens"
        )
        node.set_key_value(node._key, new_indices)
        node.host_value = None
        del self._host_nodes[key]
        self.host_resident -= node.length
        self.full_evictable += node.length
        return True

    def _free_node_mamba(self, node: RadixTreeNode, out: List[int]) -> None:
        if node.mamba_value is not None:
            out.append(node.mamba_value)
            node.mamba_value = None
            if node.mamba_ref_count == 0:
                self.mamba_evictable -= 1

    def _unlink(self, node: RadixTreeNode) -> RadixTreeNode:
        parent = node.parent
        del parent.children[self.key_fn(node._key)]
        return parent

    def _cascade_tombstone_leaves(self, parent: RadixTreeNode, kv_out: List[torch.Tensor]):
        """After a leaf is unlinked, eagerly reclaim the KV-only tombstone leaves it exposes
        upward (mamba_value None, no children, unlocked): free their KV and unlink, walking up.
        Keeps the 'a leaf always carries a live snapshot' invariant (sglang
        _iteratively_delete_tombstone_leaf). Returns (highest surviving ancestor, freed_tokens)."""
        freed = 0
        while (parent.mamba_value is None and parent.is_leaf() and parent.host_value is None
               and parent.ref_count == 0 and not parent.is_root()):
            kv_out.append(parent.value)
            self.full_evictable -= parent.length
            freed += parent.length
            parent = self._unlink(parent)
        return parent, freed

    def _path_len(self, node: RadixTreeNode) -> int:
        n, total = node, 0
        while not n.is_root():
            total += n.length
            n = n.parent
        return total

    def _collect_kv(self, node: RadixTreeNode) -> torch.Tensor:
        vals: List[torch.Tensor] = []
        n = node
        while not n.is_root():
            vals.append(n.value)
            n = n.parent
        vals.reverse()
        return torch.cat(vals) if vals else self.empty

    def _leaves(self) -> List[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.is_leaf():
                if not n.is_root():
                    out.append(n)
            else:
                stack.extend(n.children.values())
        return out

    def _snapshot_nodes(self) -> List[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.mamba_value is not None and not n.is_root():
                out.append(n)
            stack.extend(n.children.values())
        return out

    def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len, total = 0, len(input_ids)
        node = self.root
        tic = time.monotonic_ns()
        while prefix_len < total:
            parent = node
            child = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child is None:
                return node, prefix_len
            node = child
            if node.host_value is not None and not self._materialize(node):
                # the tier lost this page (dropped without host_drop) or refused: stop here
                # rather than hand back indices that are not in the pool
                return parent, prefix_len
            match_len = align_down(node.get_match_len(input_ids[prefix_len:]), self.page_size)
            prefix_len += match_len
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len
            node.timestamp = tic
        return node, prefix_len


__all__ = ["HybridRadixCache", "HybridMatch", "EvictResult", "HybridCacheHandle"]

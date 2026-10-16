"""DSV4 compress-state ring: the carry-parity contract, and which parts of a block are read.

A radix re-prefill must be bit-identical to a from-scratch prefill, and it is the ring that
makes it so: the compressor's rolling carry is persisted per 128-token window page and read
back BY VALUE when a request resumes on a page boundary. These tests pin that end-to-end on
the real pool and the real ``Compressor`` (both compress ratios), and pin how much of a
persisted block is load-bearing -- that answer decides whether the ~8.97 GiB of per-page
carry (ratio-128 pages: 512 KiB x 20 layers) can be elided.

The resume runs scrub the rolling register after the first page, which is what a real radix
hit looks like: the ``Compressor`` module is shared across requests, so nothing but the ring
may carry state into the extend. Without that scrub the first page's own prefill would
already hold the carry and the seed read would be dead weight -- which is exactly what the
negative-control tests below check, in the opposite direction.

GPU-only: the pool packers are CUDA/Triton. Geometry is small, the addressing is production's.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from freetoken.attention import dsv4_compress as dsv4_compress_mod
from freetoken.attention.dsv4_compress import CompressorBackendMixin
from freetoken.core import Context, get_global_ctx, set_global_ctx
from freetoken.kvcache.dsv4.v4_cost_model import dsv4_pool_sizes
from freetoken.kvcache.dsv4.v4_pool import DSV4PagedKVCache
from freetoken.models.deepseek_v4.args import DeepseekV4Args
from freetoken.models.deepseek_v4.compress import Compressor
from freetoken.utils.torch_utils import torch_dtype

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the pool packers are CUDA/Triton"
)

P = 128
DIM, HEAD, ROPE, INDEX_HEAD = 64, 128, 64, 64
PREFILL, RESUME_AT, TOTAL = 384, 128, 512  # 3 pages prefilled, resume on page 1, 1 page decode
RATIOS = (4, 128)
DEVICE = torch.device("cuda")


class _RingBackend(CompressorBackendMixin):
    """Just the views the compressor mixin reaches for; the sparse backend's capture buffers
    (window_ar, cmp staging) belong to the batch path and are not exercised here."""

    def __init__(self, pool, row: int = 0):
        self.pool = pool
        self.device = DEVICE
        self.window_size = P
        self.row = row  # the request's page-table row == the key of a request-keyed ring block

    def snapshot(self):
        return self.pool.full_loc_map

    def request_rows(self):
        # The rig drives ONE request, which the pool's full_loc_map addresses as ``row`` (the
        # engine's page-table row for it). Batch position == table row here, so the request-keyed
        # (ratio-128) rings are exercised with the key they use in production.
        return torch.full((1,), self.row, dtype=torch.int64, device=self.device)


def _args() -> DeepseekV4Args:
    return DeepseekV4Args(
        max_batch_size=2, max_seq_len=1024, dim=DIM, head_dim=HEAD, rope_head_dim=ROPE,
        index_head_dim=INDEX_HEAD, index_n_heads=2, index_topk=8, n_layers=len(RATIOS),
        n_heads=4, q_lora_rank=64, o_lora_rank=64, o_groups=2, moe_inter_dim=64,
        n_routed_experts=4, n_activated_experts=2, vocab_size=64,
        compress_ratios=RATIOS, window_size=P, norm_eps=1e-20,
    )


def _ctx(pool) -> None:
    try:
        ctx = get_global_ctx()
    except AssertionError:
        ctx = Context(page_size=P)
        set_global_ctx(ctx)
    ctx.kv_cache = pool
    ctx.attn_backend = _RingBackend(pool)


class _Rig:
    """One request's worth of pool + compressor, seeded identically for every run."""

    def __init__(self, ratio: int, seed: int = 0):
        self.args = _args()
        self.ratio = ratio
        self.layer_id = RATIOS.index(ratio)
        pages = TOTAL // P + 2
        sizes = dsv4_pool_sizes(num_pages=pages, args=self.args, swa_ratio=1.0, P=P)
        self.pool = DSV4PagedKVCache(
            sizes=sizes, args=self.args, device=DEVICE, P=P, n_scratch=1
        )
        for page in range(pages):
            self.pool.bind_window_pages(page * P, page * P)
        full_loc_map = torch.full(
            (2, self.args.max_seq_len), -1, dtype=torch.int32, device=DEVICE
        )
        full_loc_map[0, :TOTAL] = torch.arange(TOTAL, dtype=torch.int32, device=DEVICE)
        self.pool.full_loc_map = full_loc_map
        _ctx(self.pool)
        self.backend = get_global_ctx().attn_backend  # so a test can move the request row

        self.ring = self.pool.state_ring[self.layer_id]
        self.item = (1 + (ratio == 4)) * HEAD
        gen = torch.Generator(device="cpu").manual_seed(seed)

        def rand(*shape, scale=0.05):
            return (torch.randn(shape, generator=gen) * scale).to(DEVICE)

        self.x = rand(1, TOTAL, DIM).to(torch.bfloat16)
        rd = ROPE // 2
        phase = torch.rand(TOTAL, rd, generator=gen) * 6.283185307179586
        self.freqs_cis = torch.polar(torch.ones(TOTAL, rd), phase).to(DEVICE)

        with torch_dtype(torch.bfloat16):
            self.comp = Compressor(self.args, ratio, HEAD)
        self.comp.load_state_dict(
            {k: v.to(DEVICE) for k, v in self.comp.state_dict().items()}
        )
        with torch.no_grad():
            self.comp.wkv.weight.copy_(rand(*self.comp.wkv.weight.shape))
            self.comp.wgate.weight.copy_(rand(*self.comp.wgate.weight.shape))
            self.comp.norm.weight.fill_(1.0)
            self.comp.ape.copy_(rand(*self.comp.ape.shape, scale=0.1))
        self.comp.bind_paged(self.pool, self.layer_id, self.freqs_cis, DEVICE, tier="attn")
        self.ws = self.pool.translate_full_to_window(
            torch.arange(TOTAL, dtype=torch.int32)
        ).to(DEVICE)

    # ----- the three entry points a serving request goes through -----------------------
    def prefill(self, upto: int = PREFILL) -> None:
        self.comp.forward(self.x[:, :upto], 0, self.ws[:upto])

    def scrub_register(self) -> None:
        """A fresh request must not inherit the previous request's rolling carry."""
        self.comp._kv_state.fill_(0.0)
        self.comp._score_state.fill_(float("-inf"))

    def resume(self, *, seed: bool = True, mutate=None, scrub: bool = True) -> None:
        self.prefill(RESUME_AT)
        if not seed:
            self.comp._seed_carry_from_ring = lambda window_slot: None
        if mutate is not None:
            self.put_seed_block(mutate(self.seed_block()))
        if scrub:
            self.scrub_register()
        self.comp.extend(
            self.x[:, RESUME_AT:PREFILL], RESUME_AT, self.ws[RESUME_AT:PREFILL],
            tail_window_slot=RESUME_AT - 1,
        )

    def decode(self, lo: int = PREFILL, hi: int = TOTAL) -> None:
        for pos in range(lo, hi):
            at = lambda v: torch.tensor([v], device=DEVICE)  # noqa: E731
            self.comp.decode_step(
                self.x[:, pos : pos + 1], at(pos), at(pos - 1), at(pos), at(0)
            )

    # ----- observations ----------------------------------------------------------------
    def rows(self, n_blocks: int | None = None) -> torch.Tensor:
        n = n_blocks if n_blocks is not None else TOTAL // self.ratio
        return self.pool.cmp_pool[self.layer_id][:n].clone()

    def seed_block(self) -> torch.Tensor:
        return self.comp.attn.read_carry(
            self.layer_id, "attn", RESUME_AT - 1, self.comp.ring_size
        ).clone()

    def put_seed_block(self, block: torch.Tensor) -> None:
        self.comp.attn.write_carry(
            self.layer_id, "attn", RESUME_AT - 1, self.comp.ring_size, block
        )

    def empty_block(self) -> torch.Tensor:
        item = self.comp.item_size
        block = torch.zeros(self.comp.ring_size, 2 * item, device=DEVICE)
        block[:, item:] = float("-inf")
        return block


@pytest.mark.parametrize("ratio", RATIOS)
def test_resume_matches_from_scratch_bit_for_bit(ratio):
    ref = _Rig(ratio)
    ref.prefill()
    ref_prefill = ref.rows()
    ref.decode()
    ref_full = ref.rows()

    got = _Rig(ratio)
    got.resume()
    assert torch.equal(got.rows(), ref_prefill), "extend drifted from a full prefill"
    got.decode()
    assert torch.equal(got.rows(), ref_full), "decode drifted after the resume"


@pytest.mark.parametrize("ratio", RATIOS)
def test_resume_state_comes_through_the_ring(ratio):
    """Negative control: the extend starts from a scrubbed register, so a ratio-4 resume must
    move without the ring seed and a ratio-128 one must not (compress.py:258 discards it)."""
    ref = _Rig(ratio)
    ref.prefill()
    ref.decode()
    ref_rows = ref.rows()

    got = _Rig(ratio)
    got.resume(seed=False)
    got.decode()
    equal = torch.equal(got.rows(), ref_rows)
    assert equal is (ratio == 128), (
        "the seeded carry is read where it must not be, or dead where it must be read"
    )


def test_ratio4_only_the_seed_quadrant_is_live():
    """The extend copies ks[:ratio, :d] and ss[:ratio, :d] out of the block (compress.py:229-236):
    one ratio-row x head-dim quadrant per half. Everything else must be unread."""
    ref = _Rig(4)
    ref.resume()
    ref_rows = ref.rows()

    def poison_rest(block):
        item = block.shape[1] // 2
        keep = torch.zeros_like(block, dtype=torch.bool)
        keep[:4, :HEAD] = True
        keep[:4, item : item + HEAD] = True
        block[~keep] = 123.0
        return block

    got = _Rig(4)
    got.resume(mutate=poison_rest)
    assert torch.equal(got.rows(), ref_rows), "ratio-4 reads more of the block than the quadrant"


def test_ratio4_seed_quadrant_is_load_bearing():
    ref = _Rig(4)
    ref.resume()
    ref_rows = ref.rows()

    def poison_quadrant(block):
        item = block.shape[1] // 2
        block[:4, :HEAD] = 123.0
        block[:4, item : item + HEAD] = 0.0
        return block

    got = _Rig(4)
    got.resume(mutate=poison_quadrant)
    assert not torch.equal(got.rows(), ref_rows), "the live quadrant must change the result"


def test_ratio128_ring_is_request_keyed_with_no_page_block():
    """A ratio-128 ring holds ONE carry block per resident request row, and there is no page block
    to persist: the per-page carry is never read back (compress.py zeroes the register before
    reducing, and a radix match re-derives the block from its own tokens). Those per-page rows are
    the 8 GiB the elided layout drops."""
    rig = _Rig(128)
    assert rig.ring.request_keyed
    assert rig.ring.n_slots == _args().max_batch_size * rig.comp.ring_size
    rig.prefill()
    assert rig.comp.attn.read_carry(rig.layer_id, "attn", PREFILL - 1, rig.comp.ring_size) is None


def test_ratio128_own_block_rolls_and_every_other_row_is_dead():
    """Mid-page decode: the request's own block carries the rolling reduction (clobbering it
    diverges), while every row outside it -- all the per-page arena the elided layout would have
    written -- is dead (clobbering them changes nothing)."""

    def run(clobber: str | None) -> torch.Tensor:
        rig = _Rig(128)
        rig.prefill()
        rig.decode(PREFILL, PREFILL + 40)  # stop mid-page, so the block is half accumulated
        item, n = rig.comp.item_size, rig.comp.ring_size
        if clobber is not None:
            where = slice(n, None) if clobber == "other" else slice(0, n)
            rig.ring.buffer[where, :item] = 7.0
            rig.ring.buffer[where, item:] = -3.0
        rig.decode(PREFILL + 40, TOTAL)
        return rig.rows()

    ref = run(None)
    assert torch.equal(run("other"), ref), "a row outside the request's own block was read"
    assert not torch.equal(run("own"), ref), "the request's own rolling block must be live"


def test_ratio128_the_request_row_selects_the_block():
    """Moving the request row mid-page moves the roll to that row's own block: the carry that
    row 0 had half accumulated is not what row 1's continued decode sees, so the compressed
    rows must move. This is the two-requests-do-not-share-a-block direction of the same key."""
    def run(switch: bool) -> torch.Tensor:
        rig = _Rig(128)
        rig.prefill()
        rig.decode(PREFILL, PREFILL + 40)
        if switch:
            rig.backend.row = 1
        rig.decode(PREFILL + 40, TOTAL)
        return rig.rows()

    assert not torch.equal(run(True), run(False)), "the row key did not select the block"


def test_request_rows_reads_the_batch_page_table_rows():
    """The compressor's per-row key comes off the decode metadata: the eager path carries
    ``batch.active_table_idx``, the replay path the captured buffer (both set ``table_rows``).
    A missing key must fail loudly -- silently keying every request at row 0 would let concurrent
    requests share a carry block."""
    rows = torch.tensor([3, 7], dtype=torch.int32)
    batch = SimpleNamespace(attn_metadata=SimpleNamespace(table_rows=rows))
    with mock.patch.object(
        dsv4_compress_mod, "get_global_ctx", return_value=SimpleNamespace(batch=batch)
    ):
        got = dsv4_compress_mod.CompressorBackendMixin.request_rows(None)
    assert got.dtype == torch.int64
    assert torch.equal(got, rows.to(torch.int64))

    missing = SimpleNamespace(batch=SimpleNamespace(attn_metadata=SimpleNamespace()))
    with mock.patch.object(dsv4_compress_mod, "get_global_ctx", return_value=missing):
        with pytest.raises(AssertionError, match="page-table rows"):
            dsv4_compress_mod.CompressorBackendMixin.request_rows(None)

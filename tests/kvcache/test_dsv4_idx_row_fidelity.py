"""Fidelity packing for the V4-Flash indexer pool: re-encoding stored rows is bit-exact.

The V4-Flash pools hold **bf16** values whose fp4/fp8 quantization is already baked in --
``kvcache/dsv4/v4_pool.py:11-12``: "the fp8/fp4 quant is an in-place round-trip already baked
into the bf16 value, so ``index_select`` staging is byte-exact".  A stored value is therefore
``bf16(code * 2**e)`` with ``code`` in the e2m1 grid and ``2**e`` a power of two; bf16 represents
that product exactly, so the compact ``fp4_e8m0_b32`` row the V4.1 indexer already uses can be
unpacked back to the *identical* bf16 bits.  Nothing about what the model computes changes.

The one way that could have failed is the scale re-derivation: ``pack_rows`` sizes the scale from
the row it is given, so a row whose largest |code| is 1, 1.5, 2 or 3 is re-encoded at **half** the
scale its writer used (codes doubled, still within the grid).  This test covers exactly that case
as well as the ordinary one.

What is pinned down here:
* ``unpack(pack(v)) == v`` bitwise for rows already in the quantized domain;
* the byte accounting that makes it worth doing: 68 B/row at ``index_head_dim=128`` against 256 B
  as bf16 (3.76x) -- the report's 0.66 GiB -> 0.17 GiB for the indexer tier;
* the pool-level scatter/gather API the pool plumbing will use (write by ``row_ids``, read back by
  ``row_ids``);
* the one difference the compact row cannot express: a stored ``-0.0`` reads back as ``+0.0``
  (codes and scales carry no zero sign), which the logits kernels cannot tell apart.

This is a *verification* test: it uses the V4.1 packing kernels unchanged.  What it does not do is
route the V4-Flash pool through them -- that is the follow-up (packed pool buffer, pack on write,
``load_rows`` in the two indexer gather kernels).
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton.dsv41.pack import pack_rows, quant_roundtrip, unpack_rows
from freetoken.kvcache.dsv4.v41_row_format import FP4_E8M0_B32

IDX_DIM = 128  # V4-Flash index_head_dim
ROW_BYTES = 68  # 128 * 4 bit values = 64 B, plus 128 / 32 = 4 scale bytes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="packing kernels need a GPU")


def _bits(t: torch.Tensor) -> torch.Tensor:
    """bf16 as int16, so the comparison is on bits and not on float equality (-0.0, NaN)."""
    return t.view(torch.int16)


def _assert_bit_exact(got: torch.Tensor, want: torch.Tensor) -> None:
    assert got.dtype == want.dtype == torch.bfloat16, (got.dtype, want.dtype)
    assert torch.equal(_bits(got), _bits(want)), "bf16 bits drifted through the packed row"


@pytest.mark.parametrize("scale", [1.0, 0.25, 4.0, 1e-3, 64.0, 1e-30])
def test_stored_rows_survive_a_pack_unpack_round_trip(scale: float) -> None:
    """Rows that went through the writer's quantizer come back bit-identical."""
    torch.manual_seed(0)
    x = torch.randn(24, IDX_DIM, device="cuda", dtype=torch.float32) * scale
    stored = quant_roundtrip(x, FP4_E8M0_B32)  # what the pool holds (line 178 of compress.py)
    assert stored.dtype == torch.bfloat16

    packed = pack_rows(stored, FP4_E8M0_B32)
    assert packed.dtype == torch.uint8 and packed.shape == (24, ROW_BYTES)

    back = unpack_rows(packed, FP4_E8M0_B32, IDX_DIM)
    _assert_bit_exact(back, stored)


def test_repacking_a_stored_row_is_idempotent() -> None:
    """Packing a row that came out of the writer's quantizer cannot shift its scale.

    ``pack_rows`` sizes the scale from the row it is handed (``s = 2**ceil(log2(amax/6))``), and a
    stored row's own amax *is* the writer's amax rounded onto code 6 -- so the re-derived scale is
    the writer's scale, and packing an already-packed row is byte-stable.  That stability is what
    makes an existing pool safe to re-encode.
    """
    torch.manual_seed(1)
    x = torch.randn(12, IDX_DIM, device="cuda", dtype=torch.float32) * 3.0
    stored = quant_roundtrip(x, FP4_E8M0_B32)

    once = pack_rows(stored, FP4_E8M0_B32)
    twice = pack_rows(unpack_rows(once, FP4_E8M0_B32, IDX_DIM), FP4_E8M0_B32)
    assert torch.equal(once, twice), "a second pack changed the row bytes"


def test_a_hand_built_half_scale_row_still_round_trips() -> None:
    """The one row shape whose scale *does* shift, and why the pool cannot hold it.

    Every value is ``code * 2**e`` with ``code`` in the e2m1 grid and the largest code 3, so
    ``pack_rows`` re-derives ``2**(e-1)`` and doubles the codes -- still inside the grid, so every
    value survives.  A real pool row never looks like this: the writer sizes its scale from the
    block's own amax, which lands on code 6.
    """
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0], device="cuda")
    torch.manual_seed(2)
    picks = grid[torch.randint(0, grid.numel(), (4, IDX_DIM), device="cuda")]
    picks[0, 0] = 3.0  # force the largest code in the row to be 3
    stored = (picks * 2.0**-3).to(torch.bfloat16)

    back = unpack_rows(pack_rows(stored, FP4_E8M0_B32), FP4_E8M0_B32, IDX_DIM)
    _assert_bit_exact(back, stored)


def test_zero_rows_and_subnormals_pick_the_smallest_exponent() -> None:
    """All-zero blocks pick the smallest exponent; subnormals flush exactly as the writer flushes.

    The stored row is produced by ``quant_roundtrip`` -- hand-setting a bf16 subnormal would test a
    value the pool can never hold, since the writer's own smallest scale (``6 * 2**-126``) flushes
    it before it reaches the pool.
    """
    x = torch.zeros(3, IDX_DIM, device="cuda", dtype=torch.float32)
    x[1, :] = 2.0**-130  # smallest bf16 subnormal
    x[2, 1::2] = 1.0
    stored = quant_roundtrip(x, FP4_E8M0_B32)
    assert stored[1].abs().sum().item() == 0.0  # flushed by the writer, not by the repacking

    back = unpack_rows(pack_rows(stored, FP4_E8M0_B32), FP4_E8M0_B32, IDX_DIM)
    _assert_bit_exact(back, stored)


def test_signed_zero_normalises_to_positive() -> None:
    """The one difference the compact row cannot express: a negative zero reads back positive.

    Codes and scales carry no sign for zero, so ``-0.0`` unpacks as ``+0.0``.  Every consumer of
    this pool sums products of these values (``tl.dot`` and ``+`` in the logits kernels), where the
    two zeros are indistinguishable -- this test exists so the difference is stated, not found later.
    """
    stored = torch.full((1, IDX_DIM), -0.0, device="cuda", dtype=torch.bfloat16)
    stored[0, :8] = 1.0

    back = unpack_rows(pack_rows(stored, FP4_E8M0_B32), FP4_E8M0_B32, IDX_DIM)
    assert torch.equal(back, stored)  # -0.0 == 0.0 under value comparison
    assert (stored.view(torch.int16) == -32768).sum().item() == IDX_DIM - 8  # stored has -0.0
    assert (back.view(torch.int16) == -32768).sum().item() == 0  # packed row has only +0.0


def test_pool_scatter_and_gather_by_row_ids() -> None:
    """The write path (scatter into a slot) and the read path (gather by slot) agree."""
    torch.manual_seed(2)
    x = torch.randn(16, IDX_DIM, device="cuda", dtype=torch.float32)
    stored = quant_roundtrip(x, FP4_E8M0_B32)

    pool = torch.zeros(64, ROW_BYTES, dtype=torch.uint8, device="cuda")
    slots = torch.tensor([37, 3, 12, 60, 0, 41, 22, 58, 7, 19, 44, 31, 5, 63, 26, 14], device="cuda")
    pack_rows(stored, FP4_E8M0_B32, pool=pool, row_ids=slots)

    # Untouched rows stay zero, and the written rows read back bit-exact through the same slots.
    assert pool[1].abs().sum().item() == 0
    _assert_bit_exact(unpack_rows(pool, FP4_E8M0_B32, IDX_DIM, row_ids=slots), stored)


def test_indexer_row_shrinks_376x_against_bf16() -> None:
    """The tier arithmetic behind the packing report: 68 B vs 256 B per indexer row."""
    assert FP4_E8M0_B32.row_bytes(IDX_DIM) == ROW_BYTES
    assert IDX_DIM * 2 == 256
    assert 256 / ROW_BYTES == pytest.approx(3.7647, abs=1e-4)

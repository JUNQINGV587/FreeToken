"""V4.1 model-side paged CSA2 (M5 slice C): the paged path against the eager oracle path.

The eager path (``Attention.forward`` over the concatenated ``[window | compressed]`` KV) is the
reference: it is the one the layer tests compare against the checkpoint's own dumps. The paged path
drives the SAME layers through the pool -- global window slots, the band's compressed pool, the
compressor's state ring -- so the two must land on the same logits for the same tokens with the
same weights. That equivalence is the point of the file; everything else here guards a piece of the
threading it depends on (the carried group, the band's published picks, lazy binding).

The routed MoE is stubbed exactly as in ``test_deepseek_v41_model``: this is attention plumbing.
The paged path ends in ``sparse_attn_paged``, a Triton kernel, so the whole file is CUDA-gated.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from freetoken.attention.dsv41_sparse import DSV41AttnMetadata, DSV41SparseAttnBackend
from freetoken.core import Batch, Context, Req, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.kvcache.dsv41_cost_model import dsv41_pool_sizes
from freetoken.kvcache.dsv41_paged_pool import DSV41PagedKVCache
from freetoken.models.deepseek_v41 import model as model_mod
from freetoken.utils.torch_utils import torch_dtype

from .test_deepseek_v41_model import VOCAB, _StubFFN, _args, _band_args, _model

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# One page per window: at P == window_size the window tier is a plain one-slot-per-token map, so a
# wrong global-slot translation shows up as a wrong attention instead of as a plausible ring.
PAGE = 8
NUM_PAGES = 16
MAX_RUNNING = 2
TOL = 2e-2  # bf16: the two paths reduce in different orders (torch vs Triton)


def _ctx() -> Context:
    try:
        ctx = get_global_ctx()
    except AssertionError:
        ctx = Context(page_size=PAGE)
        set_global_ctx(ctx)
    return ctx


def _pool_and_backend(args, device):
    """A small hand-built pool: identity full->window pages, one page-table row per request."""
    sizes = dsv41_pool_sizes(num_pages=NUM_PAGES, args=args, swa_ratio=1.0, P=PAGE)
    pool = DSV41PagedKVCache(
        sizes=sizes, args=args, device=device, P=PAGE, n_scratch=MAX_RUNNING + 1
    )
    pool._init_paged_state(MAX_RUNNING, True)
    table = torch.zeros(MAX_RUNNING + 1, args.max_seq_len, dtype=torch.int32, device=device)
    # the last full page is the pool's reserved dummy row (page_table's padding convention)
    table[MAX_RUNNING].fill_(sizes.full_token - PAGE)
    for row in range(MAX_RUNNING):
        table[row, : args.max_seq_len] = torch.arange(args.max_seq_len, dtype=torch.int32)
    for page_id in range(sizes.full_token // PAGE - 1):
        pool.bind_window_pages(page_id * PAGE, page_id * PAGE)
    pool.attach_page_table(table)

    ctx = _ctx()
    ctx.kv_cache = pool
    # the backend's __init__ reads ctx.kv_cache.device, so it comes after the pool is attached
    backend = DSV41SparseAttnBackend(SimpleNamespace(dsv41_args=args))
    ctx.attn_backend = backend
    return pool, backend


def _sink(model) -> None:
    """``attn_sink`` ships uninitialised (it is a loaded scalar, not an empty buffer): give both
    paths the same deterministic vector so the comparison is about the plumbing."""
    torch.manual_seed(7)
    for layer in model.layers.op_list:
        layer.attn.attn_sink.normal_()


def _bind_paged(model, device, pool) -> None:
    """Switch a model that ``_model`` already bound eagerly onto the pool."""
    model.bind(device, pool)


def _decode_batch(backend, ctx, row: int, pos: int, device):
    # Req.__post_init__ asserts the ids live on the host: only the batch-level tensors are CUDA
    req = Req(
        input_ids=torch.zeros(1, dtype=torch.int32),
        table_idx=row,
        cached_len=0,
        output_len=1,
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]
    batch.active_table_idx = torch.tensor([row], dtype=torch.int64, device=device)
    batch.positions = torch.tensor([pos], dtype=torch.int64, device=device)
    backend.prepare_metadata(batch)
    return batch


@requires_cuda
def test_paged_ragged_prefill_matches_the_eager_path(monkeypatch):
    """The primary contract: same weights, same tokens, paged prefill == eager prefill."""
    device = torch.device("cuda")
    args = _args()
    model = _model(monkeypatch, device, args)
    _sink(model)
    torch.manual_seed(11)
    ids = torch.randint(0, VOCAB, (1, 12), device=device)

    eager = model.forward(ids, full_logits=True).float()

    pool, _ = _pool_and_backend(args, device)
    _bind_paged(model, device, pool)
    paged = model.forward_paged(ids, segments=[(0, 12, 0, 0)], full_logits=True).float()

    diff = (paged - eager).abs().max().item()
    assert torch.allclose(paged, eager, rtol=TOL, atol=TOL), f"max abs logit diff {diff}"
    assert pool.cmp_pool_of(2) is not None and pool.cmp_pool_of(3) is not None


@requires_cuda
def test_paged_prefill_matches_the_eager_path_at_a_short_prompt(monkeypatch):
    """A prompt shorter than one window and shorter than one compressed group: the degenerate
    widths (no padding, no compressed column) still have to address the same slots."""
    device = torch.device("cuda")
    args = _args()
    model = _model(monkeypatch, device, args)
    _sink(model)
    torch.manual_seed(13)
    ids = torch.randint(0, VOCAB, (1, 3), device=device)

    eager = model.forward(ids, full_logits=True).float()

    pool, _ = _pool_and_backend(args, device)
    _bind_paged(model, device, pool)
    paged = model.forward_paged(ids, segments=[(0, 3, 0, 0)], full_logits=True).float()

    diff = (paged - eager).abs().max().item()
    assert torch.allclose(paged, eager, rtol=TOL, atol=TOL), f"max abs logit diff {diff}"


@requires_cuda
def test_paged_decode_walks_the_eager_argmax_trajectory(monkeypatch):
    """Prefill N paged, then one token at a time through the paged decode path: the argmax
    trajectory must be the eager one's, and every step finite."""
    device = torch.device("cuda")
    args = _args()
    model = _model(monkeypatch, device, args)
    _sink(model)
    torch.manual_seed(17)
    n, m = 8, 6
    ids = torch.randint(0, VOCAB, (1, n + m), device=device)

    eager = [model.forward(ids[:, :n], full_logits=True)[:, -1].float()]
    for step in range(n, n + m):
        eager.append(model.forward(ids[:, step : step + 1], start_pos=step).float())

    pool, backend = _pool_and_backend(args, device)
    _bind_paged(model, device, pool)
    paged = [model.forward_paged(ids[:, :n], segments=[(0, n, 0, 0)]).float()]
    for step in range(n, n + m):
        ctx = _ctx()
        batch = _decode_batch(backend, ctx, 0, step, device)
        pos = batch.positions
        rows = batch.active_table_idx
        with ctx.forward_batch(batch):
            paged.append(
                model.forward_paged(
                    ids[:, step : step + 1], pos=pos, rows=rows, cmp_stage_cap=args.max_seq_len
                ).float()
            )

    for step, (want, got) in enumerate(zip(eager, paged)):
        assert torch.isfinite(got).all(), f"non-finite logits at step {step}"
        assert torch.equal(got.argmax(-1), want.argmax(-1)), (
            f"argmax diverged at step {step}: {(got - want).abs().max().item()}"
        )
    diff = max((g - e).abs().max().item() for e, g in zip(eager, paged))
    assert torch.allclose(torch.stack(paged), torch.stack(eager), rtol=TOL, atol=TOL), diff


@requires_cuda
def test_the_compressor_carries_a_group_that_straddles_two_segments(monkeypatch):
    """A segment that ends mid-group persists the partial group in the band's state ring; the
    next segment reads it back and pools the completed group into the same pool row the eager
    compressor would have used."""
    device = torch.device("cuda")
    args = _args()
    model = _model(monkeypatch, device, args)
    _sink(model)
    torch.manual_seed(19)
    first, second = 5, 6  # 5 is odd: the group [4, 6) straddles the boundary between them
    ids = torch.randint(0, VOCAB, (1, first + second), device=device)

    model.forward(ids[:, :first], start_pos=0)
    for step in range(first, first + second):
        model.forward(ids[:, step : step + 1], start_pos=step)
    # layer 2 is a ratio-2 kv source: 11 tokens -> 5 whole groups, rows 0..4
    eager_rows = model.layers.op_list[2].attn._compress_cache[0, :5].clone()

    pool, _ = _pool_and_backend(args, device)
    _bind_paged(model, device, pool)
    model.forward_paged(ids[:, :first], segments=[(0, first, 0, 0)])
    model.forward_paged(ids[:, first:], segments=[(0, second, 0, first)])

    paged_rows = pool.cmp_pool_of(2)[:5]
    assert pool.cmp_ratio_of(2) == 2
    assert torch.equal(paged_rows.to(torch.float32), eager_rows.to(torch.float32)), (
        (paged_rows.to(torch.float32) - eager_rows.to(torch.float32)).abs().max().item()
    )


@requires_cuda
def test_a_consumer_reads_its_bands_pool_and_published_picks(monkeypatch):
    """One source (layer 1) owns the band; layers 2 and 3 are consumers with no compressor and no
    indexer, so they read the source's pool objects and its published picks verbatim."""
    device = torch.device("cuda")
    args = _band_args()
    model = _model(monkeypatch, device, args)
    _sink(model)

    source, consumer2, consumer3 = (model.layers.op_list[i] for i in (1, 2, 3))
    assert source.attn.indexer is not None and source.attn.compressor is not None
    # a compressor of its own is exactly what a consumer must NOT have: it would pool the same
    # tokens a second time into the same band
    for layer in (consumer2, consumer3):
        assert layer.attn.compressor is None
        assert layer.attn.indexer is None

    torch.manual_seed(23)
    ids = torch.randint(0, VOCAB, (1, 10), device=device)
    pool, _ = _pool_and_backend(args, device)
    _bind_paged(model, device, pool)

    # band identity: the same tensor objects, resolved through the pool's routing
    assert pool.cmp_pool_of(2) is pool.cmp_pool_of(1)
    assert pool.idx_pool_of(2) is pool.idx_pool_of(1)
    assert pool.cmp_pool_of(3) is pool.cmp_pool_of(1)
    # layer 3's own ratio is 1, but it shares the source's ratio-2 band
    assert args.layer_ratio(3) == 1 and consumer3.attn.band_ratio == 2
    assert consumer2.attn.band_ratio == 2

    model.forward_paged(ids, segments=[(0, 10, 0, 0)], full_logits=True)

    runtime = source.attn._runtime
    assert runtime.topk_idxs is not None
    n, topk = 10, min(args.index_topk, 10 // 2)
    # a consumer's compressed half IS the source's published slice, same buffer
    want = runtime.topk_idxs[:1, :n, :topk]
    for layer in (consumer2, consumer3):
        got = layer.attn._segment_blocks(
            torch.zeros(1, n, args.dim, device=device),
            torch.zeros(1, n, args.q_lora_rank, device=device),
            0,
            0,
            0,
        )
        assert torch.equal(got, want)


@requires_cuda
def test_lazy_binding_falls_back_to_eager_with_no_context_pool(monkeypatch):
    """``_ensure_bound`` reads the pool off the context IF there is one: with no pool the model
    must stay runnable on the eager path, and ``mark_for_rebind`` must re-run the bind."""
    from freetoken.distributed.info import get_tp_info, set_tp_info

    device = torch.device("cuda")
    try:
        get_tp_info()
    except RuntimeError:
        set_tp_info(rank=0, size=1)
    monkeypatch.setattr(model_mod, "MoE", lambda *a, **k: _StubFFN(64))
    args = dataclasses.replace(_args(), engram_layer_ids=())
    config = SimpleNamespace(
        dsv41_args=args,
        quant=None,
        moe_strategy="offload",
        decode_target="gpu",
        checkpoint_path=None,
        engram_tokenizer_path=None,
        engram_table=None,
        device=device,
    )
    with torch_dtype(torch.bfloat16), torch.device(device):
        result = model_mod.DeepseekV41ForCausalLM(config)
    torch.manual_seed(1234)
    for tensor in result.state_dict().values():
        if tensor.dtype.is_floating_point:
            tensor.copy_(torch.randn_like(tensor.float()).to(tensor.dtype) * 0.1)

    # Build a pool first so the context CAN be restored, then take it away for this test.
    pool, _ = _pool_and_backend(args, device)
    ctx = _ctx()
    for name in ("kv_cache", "attn_backend"):
        if name in vars(ctx):
            delattr(ctx, name)
    try:
        assert result._bound is False
        result._ensure_bound()
        assert result._bound is True
        assert all(not layer.attn._paged for layer in result.model.layers.op_list)

        ids = torch.randint(0, VOCAB, (1, 4), device=device)
        logits = result.model.forward(ids, full_logits=True)
        assert logits.shape == (1, 4, VOCAB) and torch.isfinite(logits).all()

        result.mark_for_rebind()
        assert result._bound is False
        with pytest.raises(NotImplementedError):
            result.forward()
    finally:
        ctx.kv_cache = pool
        ctx.attn_backend = DSV41SparseAttnBackend(SimpleNamespace(dsv41_args=args))


@requires_cuda
def test_forward_paged_refuses_an_unbound_model(monkeypatch):
    """The paged driver must not silently run the eager path (a caller that forgot to bind would
    otherwise compare two eager passes and see 'agreement')."""
    device = torch.device("cuda")
    model = _model(monkeypatch, device)
    ids = torch.randint(0, VOCAB, (1, 4), device=device)
    with pytest.raises(AssertionError, match="bound to a pool"):
        model.forward_paged(ids, segments=[(0, 4, 0, 0)])
    with pytest.raises(AssertionError, match="segments"):
        model.forward_paged(ids, segments=[(0, 4, 0, 0)], pos=torch.zeros(1, dtype=torch.int64))
    assert DSV41AttnMetadata is not None  # imported for the decode helper above

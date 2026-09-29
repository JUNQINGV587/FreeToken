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
from freetoken.models.deepseek_v41.engram import EngramLayout, NgramHashState
from freetoken.utils.torch_utils import torch_dtype

from .test_deepseek_v41_model import (
    VOCAB,
    _adapter_config,
    _args,
    _band_args,
    _FakeTokenizer,
    _model,
    _set_tp_info,
    _StubFFN,
    _write_engram_checkpoint,
)

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


def _disjoint_rows(pool, args, device) -> None:
    """Give every page-table row its own full-location block, the way the scheduler does.

    ``_pool_and_backend``'s identity table is right for a single request, but pointing all rows at
    ``arange(max_seq_len)`` would let two rows address the SAME pool slots -- something the engine
    never does: ``scheduler/cache.py:841`` hands each request a disjoint full-location range, which
    is exactly what keeps concurrent requests from reading each other's window and compressed KV.
    """
    sizes = dsv41_pool_sizes(num_pages=NUM_PAGES, args=args, swa_ratio=1.0, P=PAGE)
    table = torch.zeros(MAX_RUNNING + 1, args.max_seq_len, dtype=torch.int32, device=device)
    table[MAX_RUNNING].fill_(sizes.full_token - PAGE)
    for row in range(MAX_RUNNING):
        table[row] = row * args.max_seq_len + torch.arange(args.max_seq_len, dtype=torch.int32)
    pool.attach_page_table(table)


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
        # the engine path needs a scheduler batch: with the context stripped of its pool it
        # re-binds eagerly (that is the point of the fallback) and then has nothing to step
        with pytest.raises(AssertionError, match="batch"):
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


def _req(ids_cpu, table_idx: int, cached_len: int = 0) -> Req:
    """A host-side Req exactly as the scheduler builds it (``Req`` asserts the ids are on CPU)."""
    return Req(
        input_ids=ids_cpu.cpu().to(torch.int32),
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=1,
        uid=table_idx,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )


def _adapter(monkeypatch, device, args) -> "model_mod.DeepseekV41ForCausalLM":
    """The registered adapter around the stubbed stack, with no engram layers so the engine path
    needs no checkpoint directory. Random weights: this compares plumbing, not values."""
    from freetoken.distributed.info import get_tp_info, set_tp_info

    try:
        get_tp_info()
    except RuntimeError:
        set_tp_info(rank=0, size=1)
    monkeypatch.setattr(model_mod, "MoE", lambda *a, **k: _StubFFN(64))
    config = SimpleNamespace(
        dsv41_args=dataclasses.replace(args, engram_layer_ids=()),
        quant=None,
        moe_strategy="offload",
        decode_target="gpu",
        checkpoint_path=None,
        engram_tokenizer_path=None,
        engram_table=None,
        device=device,
    )
    with torch_dtype(torch.bfloat16), torch.device(device):
        adapter = model_mod.DeepseekV41ForCausalLM(config)
    torch.manual_seed(1234)
    for tensor in adapter.state_dict().values():
        if tensor.dtype.is_floating_point:
            tensor.copy_(torch.randn_like(tensor.float()).to(tensor.dtype) * 0.1)
    return adapter


@requires_cuda
def test_the_engine_forward_answers_per_request_in_a_ragged_prefill(monkeypatch):
    """``DeepseekV41ForCausalLM.forward`` on a prefill batch: the addressing comes off the
    attention metadata (segments), and the head has to answer per REQUEST -- two requests of
    different lengths share the flat token axis, each with its own table row."""
    device = torch.device("cuda")
    args = _args()
    adapter = _adapter(monkeypatch, device, args)
    _sink(adapter.model)
    torch.manual_seed(29)
    a = torch.randint(0, VOCAB, (1, 5), device=device)
    b = torch.randint(0, VOCAB, (1, 3), device=device)
    # independent requests: the eager answer for each is a pass of its own
    want = torch.cat(
        [
            adapter.model.forward(a, full_logits=True)[:, -1].float(),
            adapter.model.forward(b, full_logits=True)[:, -1].float(),
        ]
    )

    pool, backend = _pool_and_backend(args, device)
    _disjoint_rows(pool, args, device)
    adapter.model.bind(device, pool)
    ctx = _ctx()
    ids = torch.cat([a[0], b[0]])
    reqs = [_req(ids[:5], table_idx=0), _req(ids[5:], table_idx=1)]
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    batch.input_ids = ids.to(device)
    batch.positions = torch.cat(
        [torch.arange(5, device=device), torch.arange(3, device=device)]
    )
    backend.prepare_metadata(batch)
    assert batch.attn_metadata.segments == [(0, 5, 0, 0), (5, 3, 1, 0)]
    with ctx.forward_batch(batch):
        got = adapter.forward().float()

    assert got.shape == (2, VOCAB)
    diff = (got - want).abs().max().item()
    assert torch.allclose(got, want, rtol=TOL, atol=TOL), f"max abs logit diff {diff}"


@requires_cuda
def test_the_engine_forward_decodes_a_row_onto_the_eager_trajectory(monkeypatch):
    """The decode half of the adapter: a paged prefill followed by single-token engine steps has
    to walk the eager argmax trajectory, with the compressed staging cap taken from the batch's
    position (the eager branch of the graph-vs-eager split)."""
    device = torch.device("cuda")
    args = _args()
    adapter = _adapter(monkeypatch, device, args)
    _sink(adapter.model)
    torch.manual_seed(31)
    n, m = 6, 3
    ids = torch.randint(0, VOCAB, (1, n + m), device=device)
    want = [adapter.model.forward(ids[:, :n], full_logits=True)[:, -1].float()]
    for step in range(n, n + m):
        want.append(adapter.model.forward(ids[:, step : step + 1], start_pos=step).float())

    pool, backend = _pool_and_backend(args, device)
    adapter.model.bind(device, pool)
    ctx = _ctx()
    req = _req(ids[0, :n], table_idx=0)
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = [req]
    batch.input_ids = ids[0, :n].to(device)
    batch.positions = torch.arange(n, device=device)
    backend.prepare_metadata(batch)
    with ctx.forward_batch(batch):
        got = [adapter.forward().float()]

    for step in range(n, n + m):
        decode = _decode_batch(backend, ctx, 0, step, device)
        decode.input_ids = ids[0, step].to(device).view(1)
        with ctx.forward_batch(decode):
            got.append(adapter.forward().float())

    for step, (e, g) in enumerate(zip(want, got)):
        assert g.shape == (1, VOCAB), g.shape
        assert torch.isfinite(g).all(), f"non-finite logits at step {step}"
        assert torch.equal(g.argmax(-1), e.argmax(-1)), (
            f"argmax diverged at step {step}: {(g - e).abs().max().item()}"
        )
    diff = max((e - g).abs().max().item() for e, g in zip(want, got))
    assert diff < TOL, diff


@requires_cuda
def test_a_two_row_decode_step_matches_the_same_rows_run_alone(monkeypatch):
    """A decode batch advances its rows independently -- one token each, at each row's own
    position -- so the snapshot rows, the staging cap and the page-table rows all have to stay
    per-row. Run together, the two requests must land where they land separately."""
    device = torch.device("cuda")
    args = _args()
    adapter = _adapter(monkeypatch, device, args)
    _sink(adapter.model)
    pool, backend = _pool_and_backend(args, device)
    _disjoint_rows(pool, args, device)
    adapter.model.bind(device, pool)
    ctx = _ctx()

    torch.manual_seed(37)
    lens = (5, 3)
    ids = torch.randint(0, VOCAB, (2, max(lens) + 1), device=device)
    # each request fills ITS OWN table row, one request at a time: row 1 has seen three tokens
    # and row 0 five, so a batch-wide position would put them in each other's history
    for row, n in enumerate(lens):
        req = _req(ids[row, :n], table_idx=row)
        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = [req]
        batch.input_ids = ids[row, :n].to(device)
        batch.positions = torch.arange(n, device=device)
        backend.prepare_metadata(batch)
        with ctx.forward_batch(batch):
            adapter.forward()

    steps = {row: lens[row] for row in (0, 1)}
    alone = []
    for row in (0, 1):
        decode = _decode_batch(backend, ctx, row, steps[row], device)
        decode.input_ids = ids[row, steps[row]].to(device).view(1)
        with ctx.forward_batch(decode):
            alone.append(adapter.forward().float())

    reqs = [_req(ids[row, steps[row]].unsqueeze(0), table_idx=row) for row in (0, 1)]
    joint = Batch(reqs=reqs, phase="decode")
    joint.padded_reqs = reqs
    joint.active_table_idx = torch.tensor([0, 1], dtype=torch.int64, device=device)
    joint.positions = torch.tensor([steps[0], steps[1]], dtype=torch.int64, device=device)
    joint.input_ids = torch.stack([ids[0, steps[0]], ids[1, steps[1]]]).to(device)
    backend.prepare_metadata(joint)
    with ctx.forward_batch(joint):
        got = adapter.forward().float()

    want = torch.cat(alone, dim=0)
    assert got.shape == (2, VOCAB), got.shape
    diff = (got - want).abs().max().item()
    assert torch.allclose(got, want, rtol=TOL, atol=TOL), f"max abs logit diff {diff}"


def _adapter_with_engram(monkeypatch, device, tmp_path):
    """The adapter with its engram layers LIVE: the token map and the row views come off a
    checkpoint directory laid out like the real one (``_write_engram_checkpoint``), so a packed
    pass exercises disk-backed tables, not a resident stub."""
    args = _args()
    folder = _write_engram_checkpoint(str(tmp_path), args)
    _set_tp_info()
    monkeypatch.setattr(model_mod, "MoE", lambda *a, **k: _StubFFN(64))
    config = SimpleNamespace(
        **{
            **vars(_adapter_config(folder, args)),
            "engram_table": None,
            "device": device,
        }
    )
    with torch_dtype(torch.bfloat16), torch.device(device):
        adapter = model_mod.DeepseekV41ForCausalLM(config)
    torch.manual_seed(1234)
    for tensor in adapter.state_dict().values():
        if tensor.dtype.is_floating_point:
            tensor.copy_(torch.randn_like(tensor.float()).to(tensor.dtype) * 0.1)
    return args, adapter


@requires_cuda
def test_a_packed_prefill_keeps_the_two_requests_n_gram_histories_apart(
    monkeypatch, tmp_path
):
    """The n-gram hash is keyed on the request's TABLE row and each segment restarts at its own
    position, so two requests packed into one flat token axis have to hash to what they hash to
    alone.

    Hashing the packed axis in one call -- one cache row, positions 0..T -- instead places the
    second request's tokens at the first request's positions, so it reads the first request's
    tokens as n-gram context and the row ids land on the WRONG 94 GiB entries. A wrong table row
    is a different vector, not a rounding difference, which is why the negative control below has
    to fail this tolerance."""
    device = torch.device("cuda")
    args, adapter = _adapter_with_engram(monkeypatch, device, tmp_path)
    model = adapter.model
    _sink(model)
    assert model._engram_hash is not None and [b.engram.layer_id for b in adapter.engram_layers()] == [1]
    model.bind(device)
    assert adapter.bind_engram_tier(device, use_io_uring=False) is not None

    torch.manual_seed(29)
    a = torch.randint(0, VOCAB, (1, 5), device=device)
    b = torch.randint(0, VOCAB, (1, 3), device=device)
    # each request's answer ALONE: the two eager passes are the reference the packed pass must meet
    want = torch.cat(
        [
            model.forward(a, full_logits=True)[:, -1].float(),
            model.forward(b, full_logits=True)[:, -1].float(),
        ]
    )

    pool, backend = _pool_and_backend(args, device)
    _disjoint_rows(pool, args, device)
    model.bind(device, pool)
    ctx = _ctx()
    ids = torch.cat([a[0], b[0]])
    reqs = [_req(ids[:5], table_idx=0), _req(ids[5:], table_idx=1)]
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    batch.input_ids = ids.to(device)
    batch.positions = torch.cat([torch.arange(5, device=device), torch.arange(3, device=device)])
    backend.prepare_metadata(batch)
    with ctx.forward_batch(batch):
        got = adapter.forward().float()
    diff = (got - want).abs().max().item()
    assert torch.allclose(got, want, rtol=TOL, atol=TOL), f"max abs logit diff {diff}"

    # The guard the LOGITS cannot express: the hash ROW IDS. Every id is a row of the 94 GiB
    # table, so an id that differs at all is a different vector -- but with the shrunken fake
    # table the resulting logit shift happens to stay inside bf16 tolerance, which is why this
    # compares ids instead of relying on the negative control above.
    hash_state = model._engram_hash
    segments = [(0, 5, 0, 0), (5, 3, 1, 0)]
    rows = torch.tensor([0], dtype=torch.int64, device=device)
    others = torch.tensor([1], dtype=torch.int64, device=device)

    hash_state.reset()
    packed = hash_state.forward_segments(ids.view(1, -1), segments)
    hash_state.reset()
    alone = torch.cat(
        [
            hash_state.forward(ids.view(1, -1)[:, :5], 0, rows=rows),
            hash_state.forward(ids.view(1, -1)[:, 5:], 0, rows=others),
        ],
        dim=1,
    )
    assert torch.equal(packed, alone), "the packed pass hashed to something the requests do not"

    # the call shape this replaced -- one row, positions 0..T -- must NOT agree: the second
    # request's tokens would read the first request's as n-gram context
    hash_state.reset()
    as_one_sequence = hash_state.forward(ids.view(1, -1), 0)
    assert not torch.equal(as_one_sequence, alone), (
        "hashing the packed axis in one call agreed with the per-request ids: this test no longer "
        "guards the bug it was written for"
    )


@requires_cuda
def test_decode_rows_at_different_positions_keep_their_own_n_gram_context():
    """Decode advances every row by one token, so the hash needs a position PER ROW.

    The scalar ``start_pos`` this replaced handed row 1 row 0's position, so row 1 wrote its token
    into the wrong cache cell AND read row 0's history as its own n-gram context -- both land on
    the wrong 94 GiB rows, and only the row ids show it (see the note in the packed-prefill test
    about why the logits cannot)."""
    device = torch.device("cuda")
    args = _args()
    layout = EngramLayout.from_args(args)
    assert layout is not None
    state = NgramHashState(args, layout, _FakeTokenizer(VOCAB))
    state.bind(device)

    torch.manual_seed(11)
    history = torch.randint(0, VOCAB, (2, 4), device=device)
    step = torch.randint(0, VOCAB, (2, 1), device=device)
    rows = torch.tensor([0, 1], dtype=torch.int64, device=device)
    # row 0's prefix sits at positions 0..3, row 1's at 6..9: two requests at different offsets
    starts = (0, 6)
    positions = torch.tensor([[4], [10]], dtype=torch.int64, device=device)

    def prime():
        state.reset()
        for row in (0, 1):
            state.forward(history[row : row + 1], starts[row], rows=rows[row : row + 1])

    prime()
    batched = state.forward(step, None, rows=rows, positions=positions)
    prime()
    alone = torch.cat(
        [state.forward(step[row : row + 1], starts[row] + 4, rows=rows[row : row + 1]) for row in (0, 1)],
        dim=0,
    )
    assert torch.equal(batched, alone), "the batched step hashed row 1 at row 0's position"

    # and the scalar call -- the shape this replaced -- must not agree, or the test guards nothing
    prime()
    scalar = state.forward(step, starts[0] + 4, rows=rows)
    assert not torch.equal(scalar, alone)

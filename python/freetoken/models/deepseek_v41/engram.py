"""DeepSeek-V4.1 engram layers 1/14 (port of ``inference/engram.py`` + ``model.py``'s ``Engram``).

An engram layer writes an n-gram lookup into the hyper-connection residual stream, gated by how
well the lookup matches that stream. Three pieces, in the order the reference runs them:

  * ``EngramLayout`` -- the prime-sized bucket ranges each (layer, n-gram size, head) owns. The
    primes start at ``engram_vocab_size`` and are handed out in order, never reused, which is what
    keeps the ranges disjoint; every hash multiplier derives from their count, so getting this
    layout wrong silently rehashes the whole table.
  * ``NgramHashState`` -- token ids -> compressed ids (tokens that normalize alike collapse) ->
    ``(max_ngram - 1) * n_heads`` hash columns, rolling-XOR over the lookbacks, each landing in its
    own prime range. The compressed vocab size is asserted against the config: it must equal the
    checkpoint's, or every multiplier moves.
  * ``Engram`` -- the row gather, ``wkv``, and the gate. The table itself (94 GiB per layer) is
    NOT a resident parameter; it is served by a table object, which for tests is a small resident
    pair of tensors and for the real checkpoint is read from NVMe row-by-row.

The token map depends on the *tokenizer*, so it is built from the checkpoint's ``tokenizer.json``
and asserted against ``args.engram_compressed_vocab_size`` (99092 for this checkpoint). Training
and inference must agree bit for bit here; a different normalizer rehashes the table.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, LinearReplicated, RMSNorm

from .args import DeepseekV41Args

__all__ = [
    "EngramLayout",
    "NgramHashState",
    "Engram",
    "ResidentEngramTable",
    "build_compressed_token_map",
    "compute_hash_multipliers",
    "find_next_prime",
]


def _is_prime(n: int) -> bool:
    """Trial division; the bucket primes live just above ``engram_vocab_size`` (~1.6e7)."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def find_next_prime(start: int, seen: set[int]) -> int:
    """The smallest prime above ``start`` that has not been handed out yet."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, tokenizer_vocab_size: int
) -> torch.Tensor:
    """One multiplier per (layer, lookback), from a per-layer RNG so layers hash differently.

    Kept odd, and bounded so that ``token_id * multiplier`` cannot overflow int64. The per-layer
    RNG is what ties the engine's hashes to the trained table, so the generator and its seed
    (``10007 * layer_id``) are load-bearing.
    """
    import numpy as np

    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize alike collapse.

    The normalizer chain is the training one and its ORDER is the semantics: NFKC, NFD, strip
    accents, lowercase, whitespace collapse, a private-use sentinel so a token that is exactly one
    space survives ``Strip()``, strip, sentinel back to a space. Tokens containing U+FFFD are
    partial UTF-8 bytes -- nothing to normalize -- and are keyed by their raw form.
    """
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # a transformers tokenizer reaches its Rust backend through `.backend_tokenizer`; a bare
    # `tokenizers.Tokenizer` IS that backend
    backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
    try:  # a transformers tokenizer knows its own length; the Rust one does not
        n_tokens = len(tokenizer)
    except TypeError:
        n_tokens = backend.get_vocab_size()
    key_to_new: dict[str, int] = {}
    lookup = [0] * n_tokens
    for token_id in range(n_tokens):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables (see the module docstring)."""

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]  # table rows, per engram layer
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [layer][n-gram size][head] bucket modulus
    n_heads: int
    head_dim: int

    @property
    def n_hash_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    @classmethod
    def from_args(cls, args: DeepseekV41Args) -> "EngramLayout | None":
        layer_ids = tuple(args.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size, n_heads = args.engram_max_ngram_size, args.engram_n_heads
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], args.engram_vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(args.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=args.engram_head_dim,
        )

    def bucket_offsets(self) -> torch.Tensor:
        """``[n_layers, n_hash_cols]`` start row of every (n-gram size, head) bucket range."""
        import numpy as np

        flat = [[p for per_ngram in layer for p in per_ngram] for layer in self.primes]
        return torch.tensor(np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat]))


class NgramHashState(BaseOP):
    """Maps each position to the hash ids of the n-grams ending there.

    Look-back stops at the start of the sequence and at any dead token (an image span, stored as
    ``DEAD``), so an n-gram never spans one; the cache carries that history across the
    prefill/decode split. Returns ``[B, L, n_engram_layers, n_hash_cols]`` row ids.
    """

    DEAD = -1

    def __init__(self, args: DeepseekV41Args, layout: EngramLayout, tokenizer):
        self.layout = layout
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        # Every multiplier derives from this count: a mismatch would rehash the entire table
        # silently, which is why the reference asserts it too.
        assert vocab_size == args.engram_compressed_vocab_size, (
            vocab_size,
            args.engram_compressed_vocab_size,
        )
        self.vocab_size = vocab_size
        self.pad_id = token_map[args.engram_pad_id]
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        # buffers: matched by the loader by name, not by dtype/shape brittleness
        self.primes = torch.tensor(layout.primes)
        self.offsets = layout.bucket_offsets()
        self.multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, vocab_size
        )
        self.token_map = torch.tensor(token_map)

    def bind(self, device: torch.device) -> None:
        for name in ("primes", "offsets", "multipliers", "token_map"):
            setattr(self, name, getattr(self, name).to(device))
        self._cache = torch.zeros(
            self.max_batch_size, self.max_seq_len, dtype=torch.int64, device=device
        )

    @property
    def cache(self) -> torch.Tensor:
        return self._cache

    def reset(self) -> None:
        """Blank the history: the next pass starts a new sequence at position 0.

        The buffer is CLEARED, not dropped -- dropping it would make the next forward depend on
        someone remembering to bind again, and the DEAD/pad substitutes already keep a blank slot
        from being read as a real token (every lookback past the start is blocked).
        """
        if self._cache is not None:
            self._cache.zero_()

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int | None,
        token_mask: torch.Tensor | None = None,
        *,
        rows: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``token_mask``: [B, L], False for tokens that take no part in an n-gram (image spans).

        ``rows`` -- [B] cache rows, defaulting to ``arange(B)``. A paged call passes each
        request's TABLE row instead: the history must follow the request, and a decode batch
        rebuilds its row order every step, so a batch position is not an identity. ``max_batch_size``
        is exactly the number of live table rows, which is why the cache is that wide.

        ``positions`` -- [B, L] absolute positions, defaulting to ``start_pos + arange(L)``. Pass
        it when the rows are at different offsets (a decode step whose requests are not in lock
        step); a ragged prefill instead calls this once per segment, so ``start_pos`` stays an int
        there.
        """
        batch, seqlen = input_ids.shape
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, self.DEAD)
        if rows is None:
            rows = torch.arange(batch, device=input_ids.device)
        if positions is None:
            positions = torch.arange(
                start_pos, start_pos + seqlen, device=input_ids.device
            ).expand(batch, seqlen)

        # Flat index_put_ rather than ``cache[rows, lo:hi] = ...``: advanced indexing on a tensor
        # copies, so writing back through it would move the whole [B, max_seq_len] buffer per call.
        flat = rows.to(torch.int64).unsqueeze(1) * self.max_seq_len + positions
        self.cache.view(-1).index_put_((flat.reshape(-1),), compressed.reshape(-1))

        history = self.cache.index_select(0, rows)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.layout.max_ngram_size):
            source = history.gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        tokens = torch.stack(tokens, dim=-1)  # [B, L, max_ngram_size]

        # XOR the multiplied ids together one lookback at a time: the running value after step i is
        # the hash of the (i+1)-gram, and each lands in its own prime-sized bucket range.
        products = tokens.unsqueeze(2) * self.multipliers  # [B, L, n_layers, max_ngram_size]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets

    @torch.inference_mode()
    def forward_segments(
        self, input_ids: torch.Tensor, segments: list, token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Hash a ragged prefill: one call per ``(offset, n, table_idx, start_pos)`` segment.

        Each segment is hashed on ITS OWN cache row with positions restarting at its own
        ``start_pos``, so requests packed into one flat token axis keep separate n-gram histories.
        Hashing the packed axis in one call would let the second request read the first one's
        tokens as context AND place its own tokens at the first request's positions, so the row
        ids it produced would address the wrong table entries -- a wrong lookup, not a rounding
        difference.
        """
        bsz, total = input_ids.shape
        hashes = None
        for off, n, table_idx, start_pos in segments:
            row = torch.tensor([table_idx], dtype=torch.int64, device=input_ids.device)
            part = self.forward(
                input_ids[:, off : off + n],
                start_pos,
                None if token_mask is None else token_mask[:, off : off + n],
                rows=row,
            )
            if hashes is None:
                hashes = part.new_empty((bsz, total, *part.shape[2:]))
            hashes[:, off : off + n] = part
        assert hashes is not None, "a ragged prefill carries at least one segment"
        return hashes


class ResidentEngramTable:
    """A whole engram table in memory: ``[rows, head_dim]`` fp8-e4m3 + ``[rows, head_dim/32]`` e8m0.

    Only usable for tests and for a shrunken table -- the real one is 94 GiB per layer. The gather
    dequantizes with the block scale, exactly as the reference's ``ParallelEngramEmbedding`` does
    (it never materializes the table in bf16 either).
    """

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor, *, block: int = 32):
        assert weight.dim() == 2, weight.shape
        self.weight = weight
        self.scale = scale
        self.block = block
        self.num_embeddings = weight.size(0)
        self.head_dim = weight.size(1)
        assert scale.shape == (self.num_embeddings, self.head_dim // block), scale.shape

    def to(self, device) -> "ResidentEngramTable":
        self.weight = self.weight.to(device)
        self.scale = self.scale.to(device)
        return self

    @torch.inference_mode()
    def gather(self, rows: torch.Tensor) -> torch.Tensor:
        """``rows``: int of any shape -> bf16 ``[*rows.shape, head_dim]``; out-of-range rows are 0."""
        mask = (rows < 0) | (rows >= self.num_embeddings)
        local = rows.masked_fill(mask, 0)
        values = F.embedding(local, self.weight)
        scales = F.embedding(local, self.scale)
        values = values.float().unflatten(-1, (-1, self.block)) * scales.float().unsqueeze(-1)
        values = values.flatten(-2).to(torch.bfloat16)
        return values.masked_fill(mask.unsqueeze(-1), 0)


class Engram(BaseOP):
    """Writes an n-gram lookup into the residual stream, gated by how well it matches.

    ``x`` is the 4-copy hyper-connection residual ``[B, L, hc_mult, dim]``; ``hash_ids`` are this
    layer's columns of ``NgramHashState`` output. The gate is a normalized dot product of the
    stream against the lookup key, squashed through a signed square root -- matching the training
    kernel, which is why the clamp and ``copysign`` are not cosmetic.
    """

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV41Args,
        layout: EngramLayout,
        *,
        quant_config=None,
        prefix: str = "",
        table=None,
    ):
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.clamp_value = 1e-6
        self.eps = args.norm_eps
        self.n_hash_cols = layout.n_hash_cols
        self.head_dim = layout.head_dim
        self.num_embeddings = layout.num_embeddings[self.layer_hash_index]
        # wkv is a 32-block fp8 linear in the checkpoint (6144 -> dim * (hc_mult + 1) = 25600).
        self.wkv = LinearReplicated(
            self.n_hash_cols * self.head_dim,
            args.dim * (args.hc_mult + 1),
            has_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wkv",
        )
        self.q_weight = torch.empty(args.hc_mult, args.dim, dtype=torch.bfloat16)
        self.k_weight = torch.empty(args.hc_mult, args.dim, dtype=torch.bfloat16)
        self.table = table

    def bind(self, *, table=None, device=None) -> None:
        if table is not None:
            self.table = table
        if device is not None and self.table is not None:
            self.table = self.table.to(device)

    @torch.inference_mode()
    def forward(
        self, x: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.table is None:
            raise RuntimeError(
                "deepseek_v41: engram layer "
                f"{self.layer_id} has no table bound; the 94 GiB tables are served by the "
                "EngramTier (pass table= at construction or bind(table=...))."
            )
        kv = self.wkv.forward(self.table.gather(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()
        h = x.float()
        # normalized per (token, hc copy) over `dim`, NOT jointly over the copies
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(
            key.square().mean(-1) + self.eps
        )
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


def tokenizer_of(model_dir: str):
    """The reference builds its token map from the checkpoint's own tokenizer.

    Returns a bare ``tokenizers.Tokenizer``: the map only ever calls ``decode``/``id_to_token`` on
    the Rust backend, so the transformers wrapper is not needed.
    """
    from tokenizers import Tokenizer

    return Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))

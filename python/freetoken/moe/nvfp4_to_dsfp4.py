"""Lossless NVFP4 -> DS-FP4 (OCP MXFP4 per-32 UE8M0) repack for DeepSeek-V4.1.

The V4.1 checkpoint stores routed experts as ModelOpt NVFP4: packed e2m1 codes,
fp8-e4m3 scales per 16 values and one fp16/fp32 global scale per projection. A
full-model audit (tools/trace/audit_nvfp4_pow2.py) shows every e4m3 scale is an
exact power of two AND every adjacent pair of 16-value groups shares one scale,
so the effective on-disk format is already MXFP4: one power-of-two scale per 32
values. Folding the (also power-of-two) global into the scale exponent and
dropping every other scale byte is therefore BIT-EXACT -- this module is a pure
re-pack, no re-quantization, no checkpoint rewrite.

Two consumers:
* load time -- ``convert_pieces`` turns the NVFP4 reader's piece dict into the
  ``gate_up`` / ``gate_up_scale`` / ``down`` / ``down_scale`` pieces the DS-FP4
  (TritonMxfp4MoEKernel) pack expects;
* fetch time -- the disk tier reads the NVFP4 scale extent from the checkpoint
  and calls ``convert_scale_rows`` before the H2D copy into the DS-FP4 cache row.

Invalid inputs (a non-power-of-two scale byte, a zero scale, a scale pair that
disagrees within a 32-group) never silently corrupt: ``verify=True`` raises, and
the always-on LUT path maps them to the e8m0 NaN code 0xFF so a bad row fails
loudly downstream instead of serving quietly wrong weights.
"""

from __future__ import annotations

import math

import numpy as np
import torch

_INVALID = -32768  # exponent LUT sentinel: byte is not a positive power-of-two e4m3
_E8M0_NAN = 0xFF   # e8m0 has no inf; 0xFF is NaN -- the loud-failure marker


def _build_e4m3_exp_lut() -> np.ndarray:
    """e4m3 byte -> base-2 exponent, valid only for positive exact powers of two."""
    lut = np.full(256, _INVALID, dtype=np.int16)
    for b in range(256):
        if b & 0x80:
            continue  # negative scale: never valid for weights
        exp, man = (b >> 3) & 0xF, b & 0x7
        if man == 0 and exp >= 1:
            lut[b] = exp - 7
        elif exp == 0 and man in (1, 2, 4):
            lut[b] = -9 + (man.bit_length() - 1)  # subnormal 1/2/4 x 2^-6 = 2^-9/-8/-7
    return lut


E4M3_EXP_LUT = _build_e4m3_exp_lut()


def global_exponent(value) -> int:
    """Base-2 exponent of a power-of-two global scale (fp16/fp32 scalar or tensor)."""
    v = float(np.asarray(value).reshape(-1)[0])
    if not (v > 0.0) or not math.isfinite(v):
        raise ValueError(f"NVFP4 global scale is not a positive finite value: {v!r}")
    e = math.log2(v)
    r = round(e)
    if abs(e - r) > 1e-6:
        raise ValueError(f"NVFP4 global scale is not a power of two: {v!r} (log2={e})")
    return int(r)


def convert_scale_rows(scale_u8: np.ndarray, gexp: int | np.ndarray, *,
                       verify: bool = False) -> np.ndarray:
    """[..., K//16] e4m3 bytes -> [..., K//32] e8m0 codes with the global folded in.

    Every output code = e4m3_exponent(even column) + 127 + gexp. ``gexp`` is one
    int or a vector with one exponent per leading (batch) element. With
    ``verify=True`` the odd columns must equal the even ones and every byte must
    be a valid power of two (raises otherwise); without it invalid bytes still
    map to the e8m0 NaN code so corruption is loud, never silent.
    """
    if scale_u8.shape[-1] % 2:
        raise ValueError(f"scale row width {scale_u8.shape[-1]} is not even")
    even = scale_u8[..., 0::2]
    if verify:
        if not np.array_equal(even, scale_u8[..., 1::2]):
            raise ValueError("NVFP4 scales: adjacent 16-value groups disagree "
                             "(checkpoint is not per-32 uniform; lossless conversion impossible)")
    e = E4M3_EXP_LUT[even].astype(np.int32)
    if verify and (e == _INVALID).any():
        bad = int((e == _INVALID).sum())
        raise ValueError(f"NVFP4 scales: {bad} byte(s) are not positive powers of two; "
                         "lossless conversion impossible")
    g = np.asarray(gexp, dtype=np.int32)
    # Broadcast: scalar, or one exponent per leading batch element.
    out = e + (127 + g.reshape((-1,) + (1,) * (e.ndim - 1)) if g.ndim else 127 + int(g))
    out = np.where(e == _INVALID, _E8M0_NAN, out)
    if ((out < 0) | (out > _E8M0_NAN)).any():
        raise ValueError("folded scale exponent out of e8m0 range")
    return out.astype(np.uint8)


def _scale_exponents(scale_u8: np.ndarray) -> np.ndarray:
    """Per-row leading-dim exponents for validation/reporting (even columns only)."""
    return E4M3_EXP_LUT[scale_u8[..., 0::2]].astype(np.int32)


def convert_pieces(pieces: dict[str, torch.Tensor], *, verify: bool = True) -> dict[str, torch.Tensor]:
    """NVFP4 reader piece dict -> DS-FP4 pack pieces.

    In:  gate/up [B, I, H//2] u8, down [B, H, I//2] u8,
         gate_scale/up_scale [B, I, H//16] e4m3-as-u8, down_scale [B, H, I//16],
         gate_global/up_global/down_global [B] (or [B, 1]) fp16 pow2 scalars.
    Out: gate_up [B, 2I, H//2] u8 (gate rows first), gate_up_scale [B, 2I, H//32]
         e8m0-as-u8, down [B, H, I//2], down_scale [B, H, I//32] -- exactly what
         TritonMxfp4MoEKernel.pack consumes. Empty (disk-skipped) pieces pass
         through unchanged so build_expert_banks keeps its completion contract.
    """
    if not pieces:
        return pieces
    out: dict[str, torch.Tensor] = {
        "gate_up": torch.cat([pieces["gate"], pieces["up"]], dim=1),
        "down": pieces["down"],
    }
    scales: dict[str, torch.Tensor] = {}
    for proj, role in (("gate", "gate_up_scale"), ("up", "gate_up_scale"),
                       ("down", "down_scale")):
        su8 = pieces[f"{proj}_scale"]
        if su8.dtype != torch.uint8:
            su8 = su8.view(torch.uint8)
        gexp = np.array([global_exponent(v) for v in pieces[f"{proj}_global"].reshape(-1)],
                        dtype=np.int32)
        conv = convert_scale_rows(su8.numpy(), gexp, verify=verify)
        t = torch.from_numpy(conv)
        scales[role] = t if role not in scales else torch.cat([scales[role], t], dim=1)
    out["gate_up_scale"] = scales["gate_up_scale"]
    out["down_scale"] = scales["down_scale"]
    return out

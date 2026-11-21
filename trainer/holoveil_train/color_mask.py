"""Color space, exact resampling, perceptual masks, Blue-Noise dithering, and quality metrics.

Implements Section 3 of docs/DESIGN.md (Version 0.5.2-design) with exact mathematical
parity targets for the future Pure Rust (holoveil-core) implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Section 3.1 / 3.3 / 3.4 Locked Constants
# ---------------------------------------------------------------------------

FIXED_CHROMA_RATIO: float = 0.40
BASE_ETA_Y_255: float = 12.0
ETA_Y: float = BASE_ETA_Y_255 / 255.0
ETA_CB: float = FIXED_CHROMA_RATIO * ETA_Y
ETA_CR: float = FIXED_CHROMA_RATIO * ETA_Y

TEX_REF_LONG_SIDE: int = 512
TEX_BOX_RADIUS: int = 4
TEX_SIGMA_LO: float = 0.012
TEX_SIGMA_HI: float = 0.080
TEX_M_MIN: float = 0.30
GF_RADIUS: int = 8
GF_EPS: float = 1e-3

ANTI_CLAMP_MARGIN: float = 24.0 / 255.0
NO_OP_GUARD_EPS_255: float = 1e-3
CHROMA_LP_SIGMA: float = 3.0
MAX_DIRECT_PIXELS: int = 1_150_000

ScaleTargetType = Literal["Standard512", "Bucket1024"]


@dataclass(frozen=True)
class WorkMaskBundle:
    """Perceptual masks aligned to reference (512), working grid (H_w, W_w), and native (H, W)."""

    work_h: int
    work_w: int
    m_tex_ref: torch.Tensor       # [B, 1, ref_h, ref_w]
    w_flat_ref: torch.Tensor      # [B, 1, ref_h, ref_w]
    m_tex_work: torch.Tensor      # [B, 1, H_w, W_w]
    w_flat_work: torch.Tensor     # [B, 1, H_w, W_w]
    m_tex_native: torch.Tensor    # [B, 1, H, W]
    w_flat_native: torch.Tensor   # [B, 1, H, W]
    m_clip_native: torch.Tensor   # [B, 1, H, W]
    m_clip_work: torch.Tensor     # [B, 1, H_w, W_w]
    m_roi_work: torch.Tensor      # [B, 1, H_w, W_w]
    m_roi_native: torch.Tensor    # [B, 1, H, W]
    m_in: torch.Tensor            # [B, 1, H_w, W_w] = m_tex_work * m_roi_work * m_clip_work
    m_final: torch.Tensor         # [B, 1, H, W]     = m_tex_native * m_roi_native * m_clip_native


# ---------------------------------------------------------------------------
# Shape Helpers
# ---------------------------------------------------------------------------


def _ensure_4d(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.ndim == 3:
        return x.unsqueeze(0), True
    if x.ndim == 4:
        return x, False
    raise ValueError(f"Expected 3D [C,H,W] or 4D [B,C,H,W] tensor, got shape {tuple(x.shape)}")


def _restore_shape(x: torch.Tensor, was_3d: bool) -> torch.Tensor:
    return x.squeeze(0) if was_3d else x


# ---------------------------------------------------------------------------
# Section 3.1: ITU-R BT.601 Full-Range Color Conversions
# ---------------------------------------------------------------------------


def rgb_to_luma(rgb: torch.Tensor) -> torch.Tensor:
    """Compute BT.601 full-range luma Y in [0, 1] from RGB in [0, 1]."""
    x, was_3d = _ensure_4d(rgb)
    if x.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, got {x.shape[1]}")
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299000 * r + 0.587000 * g + 0.114000 * b
    return _restore_shape(y, was_3d)


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB in [0, 1] to unnormalized BT.601 full-range (Y in [0,1], Cb,Cr in [-0.5,0.5])."""
    x, was_3d = _ensure_4d(rgb)
    if x.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, got {x.shape[1]}")
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299000 * r + 0.587000 * g + 0.114000 * b
    cb = -0.168736 * r - 0.331264 * g + 0.500000 * b
    cr = 0.500000 * r - 0.418688 * g - 0.081312 * b
    return _restore_shape(torch.cat([y, cb, cr], dim=1), was_3d)


def rgb_to_ycbcr_net(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB in [0, 1] to network-normalized YCbCr X_net = (2Y-1, 2Cb, 2Cr) in [-1, 1]^3."""
    ycbcr = rgb_to_ycbcr(rgb)
    x, was_3d = _ensure_4d(ycbcr)
    y_net = 2.0 * x[:, 0:1] - 1.0
    cb_net = 2.0 * x[:, 1:2]
    cr_net = 2.0 * x[:, 2:3]
    return _restore_shape(torch.cat([y_net, cb_net, cr_net], dim=1), was_3d)


def ycbcr_net_to_rgb(ycbcr_net: torch.Tensor) -> torch.Tensor:
    """Convert network-normalized YCbCr in [-1, 1]^3 back to RGB (analysis/evaluation only)."""
    x, was_3d = _ensure_4d(ycbcr_net)
    if x.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, got {x.shape[1]}")
    y = 0.5 * (x[:, 0:1] + 1.0)
    cb = 0.5 * x[:, 1:2]
    cr = 0.5 * x[:, 2:3]
    r = y + 1.402000 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772000 * cb
    return _restore_shape(torch.cat([r, g, b], dim=1), was_3d)


def delta_ycbcr_to_rgb(
    hat_delta: torch.Tensor,
    mask: torch.Tensor | float = 1.0,
    lam: torch.Tensor | float = 1.0,
    chroma_ratio: float = FIXED_CHROMA_RATIO,
    eta_y_255: float = BASE_ETA_Y_255,
) -> torch.Tensor:
    """Map normalized YCbCr perturbation hat_delta in [-1, 1]^3 directly to Delta_RGB.

    Applies Delta_RGB = M_inv * (lam * mask * [eta_Y, chroma_ratio*eta_Y, chroma_ratio*eta_Y] * hat_delta).
    Clean pixels are never round-tripped through RGB->YCbCr->RGB; Delta_RGB is added directly.
    """
    if chroma_ratio != FIXED_CHROMA_RATIO:
        raise ValueError(f"chroma_ratio is locked to {FIXED_CHROMA_RATIO} per DESIGN.md §3.1, got {chroma_ratio}")
    x, was_3d = _ensure_4d(hat_delta)
    if x.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, got {x.shape[1]}")

    eta_y = eta_y_255 / 255.0
    eta_c = chroma_ratio * eta_y

    if isinstance(lam, torch.Tensor):
        if lam.ndim == 1:
            lam_t = lam.view(-1, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        else:
            lam_t = lam.to(device=x.device, dtype=x.dtype)
    else:
        lam_t = float(lam)

    if isinstance(mask, torch.Tensor):
        mask_t, _ = _ensure_4d(mask.to(device=x.device, dtype=x.dtype))
    else:
        mask_t = float(mask)

    scale_base = lam_t * mask_t
    d_y = scale_base * eta_y * x[:, 0:1]
    d_cb = scale_base * eta_c * x[:, 1:2]
    d_cr = scale_base * eta_c * x[:, 2:3]

    d_r = d_y + 1.402000 * d_cr
    d_g = d_y - 0.344136 * d_cb - 0.714136 * d_cr
    d_b = d_y + 1.772000 * d_cb
    return _restore_shape(torch.cat([d_r, d_g, d_b], dim=1), was_3d)


# ---------------------------------------------------------------------------
# Section 3.2: Exact Separable Resampling Kernels (Python <-> Rust Parity)
# ---------------------------------------------------------------------------


def _gather_1d(x: torch.Tensor, dim: int, idx: torch.Tensor) -> torch.Tensor:
    """Index into x along `dim` using 2D index tensor `idx` of shape [dst_len, K]."""
    dst_len, k_len = idx.shape
    flat_idx = idx.reshape(-1)
    gathered = torch.index_select(x, dim=dim, index=flat_idx)
    shape = list(x.shape)
    new_shape = shape[:dim] + [dst_len, k_len] + shape[dim + 1 :]
    return gathered.reshape(new_shape)


def _area_box_1d(x: torch.Tensor, dim: int, dst_len: int) -> torch.Tensor:
    n_in = x.shape[dim]
    if n_in == dst_len:
        return x
    if dst_len <= 0:
        raise ValueError(f"dst_len must be positive, got {dst_len}")

    ratio = n_in / float(dst_len)
    o = torch.arange(dst_len, dtype=torch.float64, device=x.device)
    s0 = o * ratio
    s1 = torch.clamp((o + 1.0) * ratio, max=float(n_in))
    k0 = torch.floor(s0).to(torch.long)
    max_k = int(math.ceil(ratio)) + 2
    offsets = torch.arange(max_k, dtype=torch.long, device=x.device)
    k_raw = k0[:, None] + offsets[None, :]  # [dst_len, max_k]
    valid = k_raw < n_in
    k_safe = torch.clamp(k_raw, 0, n_in - 1)

    k_f = k_raw.to(torch.float64)
    overlap = torch.clamp(torch.minimum(s1[:, None], k_f + 1.0) - torch.maximum(s0[:, None], k_f), min=0.0)
    overlap = overlap * valid.to(torch.float64)
    w = (overlap / overlap.sum(dim=-1, keepdim=True).clamp_min(1e-12)).to(dtype=x.dtype)

    gathered = _gather_1d(x, dim=dim, idx=k_safe)
    # Broadcast w to gathered's dimensions: [..., dst_len, max_k, ...]
    view_shape = [1] * gathered.ndim
    view_shape[dim] = dst_len
    view_shape[dim + 1] = max_k
    return (gathered * w.reshape(view_shape)).sum(dim=dim + 1)


def exact_area_box_resize(x: torch.Tensor, dst_h: int, dst_w: int) -> torch.Tensor:
    """Exact Weighted Area Box Filter (separable, differentiable).

    Used for downsampling (original -> 512 ref, working grid, or s=0.50 target).
    """
    t, was_3d = _ensure_4d(x)
    if t.shape[-2] == dst_h and t.shape[-1] == dst_w:
        return _restore_shape(t, was_3d)
    t = _area_box_1d(t, dim=2, dst_len=dst_h)
    t = _area_box_1d(t, dim=3, dst_len=dst_w)
    return _restore_shape(t, was_3d)


def _bilinear_1d(x: torch.Tensor, dim: int, dst_len: int) -> torch.Tensor:
    n_in = x.shape[dim]
    if n_in == dst_len:
        return x
    if dst_len <= 0:
        raise ValueError(f"dst_len must be positive, got {dst_len}")

    o = torch.arange(dst_len, dtype=torch.float64, device=x.device)
    u = torch.clamp((o + 0.5) * (n_in / float(dst_len)) - 0.5, 0.0, float(n_in - 1))
    k0 = torch.floor(u).to(torch.long)
    k1 = torch.clamp(k0 + 1, max=n_in - 1)
    t = (u - k0.to(torch.float64)).to(dtype=x.dtype)

    idx = torch.stack([k0, k1], dim=-1)  # [dst_len, 2]
    w = torch.stack([1.0 - t, t], dim=-1)  # [dst_len, 2]

    gathered = _gather_1d(x, dim=dim, idx=idx)
    view_shape = [1] * gathered.ndim
    view_shape[dim] = dst_len
    view_shape[dim + 1] = 2
    return (gathered * w.reshape(view_shape)).sum(dim=dim + 1)


def bilinear_resize(x: torch.Tensor, dst_h: int, dst_w: int) -> torch.Tensor:
    """Exact Bilinear resampling with align_corners=False (separable, differentiable)."""
    t, was_3d = _ensure_4d(x)
    if t.shape[-2] == dst_h and t.shape[-1] == dst_w:
        return _restore_shape(t, was_3d)
    t = _bilinear_1d(t, dim=2, dst_len=dst_h)
    t = _bilinear_1d(t, dim=3, dst_len=dst_w)
    return _restore_shape(t, was_3d)


def _catmull_rom_1d(x: torch.Tensor, dim: int, dst_len: int, a: float = -0.5) -> torch.Tensor:
    n_in = x.shape[dim]
    if n_in == dst_len:
        return x
    if dst_len <= 0:
        raise ValueError(f"dst_len must be positive, got {dst_len}")

    o = torch.arange(dst_len, dtype=torch.float64, device=x.device)
    u = (o + 0.5) * (n_in / float(dst_len)) - 0.5
    base = torch.floor(u).to(torch.long)
    offsets = torch.tensor([-1, 0, 1, 2], dtype=torch.long, device=x.device)
    k_raw = base[:, None] + offsets[None, :]  # [dst_len, 4]
    dist = torch.abs(u[:, None] - k_raw.to(torch.float64))

    w = torch.zeros_like(dist)
    m1 = dist <= 1.0
    m2 = (dist > 1.0) & (dist < 2.0)
    d1 = dist[m1]
    w[m1] = (a + 2.0) * (d1**3) - (a + 3.0) * (d1**2) + 1.0
    d2 = dist[m2]
    w[m2] = a * (d2**3) - 5.0 * a * (d2**2) + 8.0 * a * d2 - 4.0 * a

    w = (w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)).to(dtype=x.dtype)
    k_idx = torch.clamp(k_raw, 0, n_in - 1)

    gathered = _gather_1d(x, dim=dim, idx=k_idx)
    view_shape = [1] * gathered.ndim
    view_shape[dim] = dst_len
    view_shape[dim + 1] = 4
    return (gathered * w.reshape(view_shape)).sum(dim=dim + 1)


def catmull_rom_resize(
    x: torch.Tensor,
    dst_h: int,
    dst_w: int,
    clamp_unit: bool = False,
) -> torch.Tensor:
    """Separable Catmull-Rom Bicubic (a = -0.5, align_corners=False) per DESIGN.md §3.2.

    Do NOT substitute PyTorch F.interpolate(mode='bicubic') which uses a = -0.75.
    """
    t, was_3d = _ensure_4d(x)
    if t.shape[-2] == dst_h and t.shape[-1] == dst_w:
        out = t.clamp(-1.0, 1.0) if clamp_unit else t
        return _restore_shape(out, was_3d)
    t = _catmull_rom_1d(t, dim=2, dst_len=dst_h, a=-0.5)
    t = _catmull_rom_1d(t, dim=3, dst_len=dst_w, a=-0.5)
    if clamp_unit:
        t = t.clamp(-1.0, 1.0)
    return _restore_shape(t, was_3d)


# ---------------------------------------------------------------------------
# Section 3.5: Working Grid Planner
# ---------------------------------------------------------------------------


def round_mult16(val: float) -> int:
    return max(16, int(round(val / 16.0)) * 16)


def round_mult8(val: float) -> int:
    return max(8, int(round(val / 8.0)) * 8)


def compute_work_grid(orig_h: int, orig_w: int, scale_target: ScaleTargetType = "Standard512") -> tuple[int, int]:
    """Compute padding-free multiple-of-16 working grid (H_w, W_w) per DESIGN.md §3.5."""
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"Image dimensions must be positive, got ({orig_h}, {orig_w})")

    if scale_target == "Standard512":
        long_orig = max(orig_h, orig_w)
        short_orig = min(orig_h, orig_w)
        long_w = 512
        short_w = round_mult16(512.0 * short_orig / long_orig)
        if short_w < 256:
            # Extreme aspect ratio handling (e.g. 600x3000 -> 256x1280)
            short_w = 256
            long_w = round_mult16(short_w * long_orig / short_orig)
        return (long_w, short_w) if orig_h >= orig_w else (short_w, long_w)

    if scale_target == "Bucket1024":
        scale = 1024.0 / math.sqrt(float(orig_h) * float(orig_w))
        h_w = round_mult16(orig_h * scale)
        w_w = round_mult16(orig_w * scale)
        if min(h_w, w_w) < 256:
            long_orig = max(orig_h, orig_w)
            short_orig = min(orig_h, orig_w)
            short_w = 256
            long_w = round_mult16(short_w * long_orig / short_orig)
            return (long_w, short_w) if orig_h >= orig_w else (short_w, long_w)
        return (h_w, w_w)

    raise ValueError(f"Unknown scale_target: {scale_target!r}")


# ---------------------------------------------------------------------------
# Section 3.3: Scale-Invariant Texture Mask (m_tex), Anti-Clamp (m_clip), MinPool
# ---------------------------------------------------------------------------


def box_filter_2d(x: torch.Tensor, radius: int) -> torch.Tensor:
    """2D mean box filter with edge (replicate) padding."""
    if radius <= 0:
        return x
    t, was_3d = _ensure_4d(x)
    padded = F.pad(t, (radius, radius, radius, radius), mode="replicate")
    k = 2 * radius + 1
    out = F.avg_pool2d(padded, kernel_size=k, stride=1)
    return _restore_shape(out, was_3d)


def guided_filter_gray(
    guide: torch.Tensor,
    src: torch.Tensor,
    radius: int = GF_RADIUS,
    eps: float = GF_EPS,
) -> torch.Tensor:
    """Edge-preserving Guided Filter with grayscale guide I and edge-padded box filter."""
    i_t, was_3d = _ensure_4d(guide)
    p_t, _ = _ensure_4d(src)

    mean_i = box_filter_2d(i_t, radius)
    mean_p = box_filter_2d(p_t, radius)
    corr_i = box_filter_2d(i_t * i_t, radius)
    corr_ip = box_filter_2d(i_t * p_t, radius)

    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = box_filter_2d(a, radius)
    mean_b = box_filter_2d(b, radius)
    q = mean_a * i_t + mean_b
    return _restore_shape(q, was_3d)


def compute_texture_and_flat_mask(
    rgb: torch.Tensor,
    ref_long_side: int = TEX_REF_LONG_SIDE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute fixed-512-reference texture mask m_tex_ref and flatness W_flat_ref (§3.3 A)."""
    x, was_3d = _ensure_4d(rgb)
    _, _, h, w = x.shape
    long_side = max(h, w)
    if long_side == ref_long_side:
        rgb_ref = x
    else:
        if h >= w:
            ref_h = ref_long_side
            ref_w = max(1, int(round(ref_long_side * w / float(h))))
        else:
            ref_w = ref_long_side
            ref_h = max(1, int(round(ref_long_side * h / float(w))))
        if long_side > ref_long_side:
            rgb_ref = exact_area_box_resize(x, ref_h, ref_w)
        else:
            rgb_ref = bilinear_resize(x, ref_h, ref_w)

    y_ref = rgb_to_luma(rgb_ref)
    mu_y = box_filter_2d(y_ref, TEX_BOX_RADIUS)
    mu_y2 = box_filter_2d(y_ref * y_ref, TEX_BOX_RADIUS)
    sigma_y = torch.sqrt(torch.clamp(mu_y2 - mu_y * mu_y, min=0.0))

    u_tex = torch.clamp((sigma_y - TEX_SIGMA_LO) / (TEX_SIGMA_HI - TEX_SIGMA_LO), 0.0, 1.0)
    m_raw = TEX_M_MIN + (1.0 - TEX_M_MIN) * u_tex
    w_flat_raw = 1.0 - u_tex

    m_tex_ref = guided_filter_gray(y_ref, m_raw, radius=GF_RADIUS, eps=GF_EPS).clamp(TEX_M_MIN, 1.0)
    w_flat_ref = guided_filter_gray(y_ref, w_flat_raw, radius=GF_RADIUS, eps=GF_EPS).clamp(0.0, 1.0)
    return _restore_shape(m_tex_ref, was_3d), _restore_shape(w_flat_ref, was_3d)


def _smoothstep(u: torch.Tensor) -> torch.Tensor:
    return u * u * (3.0 - 2.0 * u)


def compute_anti_clamp_mask(
    rgb: torch.Tensor,
    margin: float = ANTI_CLAMP_MARGIN,
    min_filter_size: int = 3,
) -> torch.Tensor:
    """Compute native-resolution anti-clamp mask m_clip_native with 3x3 min filter (§3.3 B)."""
    x, was_3d = _ensure_4d(rgb)
    c_max = x.max(dim=1, keepdim=True).values
    c_min = x.min(dim=1, keepdim=True).values

    u_hi = torch.clamp((1.0 - c_max) / margin, 0.0, 1.0)
    u_lo = torch.clamp(c_min / margin, 0.0, 1.0)
    m_clip_raw = _smoothstep(u_hi) * _smoothstep(u_lo)

    if min_filter_size > 1:
        pad = min_filter_size // 2
        padded = F.pad(m_clip_raw, (pad, pad, pad, pad), mode="replicate")
        m_clip_native = -F.max_pool2d(-padded, kernel_size=min_filter_size, stride=1)
    else:
        m_clip_native = m_clip_raw
    return _restore_shape(m_clip_native, was_3d)


def _min_pool_exact_1d(x: torch.Tensor, dim: int, dst_len: int) -> torch.Tensor:
    n_in = x.shape[dim]
    if n_in == dst_len:
        return x
    if dst_len <= 0:
        raise ValueError(f"dst_len must be positive, got {dst_len}")

    idx_out = torch.arange(dst_len, dtype=torch.long, device=x.device)
    starts = torch.clamp((idx_out * n_in) // dst_len, max=n_in - 1)
    ceil_ends = ((idx_out + 1) * n_in + dst_len - 1) // dst_len
    ends = torch.clamp(torch.maximum(starts + 1, ceil_ends), max=n_in)

    max_k = int((ends - starts).max().item())
    offsets = torch.arange(max_k, dtype=torch.long, device=x.device)
    raw_idx = starts[:, None] + offsets[None, :]
    valid = raw_idx < ends[:, None]
    clamped_idx = torch.clamp(raw_idx, 0, n_in - 1)

    gathered = _gather_1d(x, dim=dim, idx=clamped_idx)
    view_shape = [1] * gathered.ndim
    view_shape[dim] = dst_len
    view_shape[dim + 1] = max_k
    valid_mask = valid.reshape(view_shape)
    gathered = gathered.masked_fill(~valid_mask, float("inf"))
    return gathered.amin(dim=dim + 1)


def min_pool_exact(src: torch.Tensor, dst_h: int, dst_w: int) -> torch.Tensor:
    """Exact non-integer ratio min pooling per DESIGN.md §3.3 C."""
    t, was_3d = _ensure_4d(src)
    t = _min_pool_exact_1d(t, dim=2, dst_len=dst_h)
    t = _min_pool_exact_1d(t, dim=3, dst_len=dst_w)
    return _restore_shape(t, was_3d)


def build_masks(
    rgb: torch.Tensor,
    scale_target: ScaleTargetType = "Standard512",
    work_hw: tuple[int, int] | None = None,
    m_roi_native: torch.Tensor | None = None,
    texture_adaptive: bool = True,
    anti_clamp_native: bool = True,
) -> WorkMaskBundle:
    """Construct complete perceptual mask bundle for native (H, W) and working grid (H_w, W_w)."""
    x, _ = _ensure_4d(rgb)
    b, _, h, w = x.shape
    if work_hw is None:
        h_w, w_w = compute_work_grid(h, w, scale_target)
    else:
        h_w, w_w = work_hw

    if texture_adaptive:
        m_tex_ref, w_flat_ref = compute_texture_and_flat_mask(x)
    else:
        ref_h, ref_w = (512, max(1, round(512 * w / h))) if h >= w else (max(1, round(512 * h / w)), 512)
        m_tex_ref = torch.ones((b, 1, ref_h, ref_w), device=x.device, dtype=x.dtype)
        w_flat_ref = torch.ones((b, 1, ref_h, ref_w), device=x.device, dtype=x.dtype)

    m_tex_work = bilinear_resize(m_tex_ref, h_w, w_w)
    w_flat_work = bilinear_resize(w_flat_ref, h_w, w_w)
    m_tex_native = bilinear_resize(m_tex_ref, h, w)
    w_flat_native = bilinear_resize(w_flat_ref, h, w)

    if anti_clamp_native:
        m_clip_native = compute_anti_clamp_mask(x)
    else:
        m_clip_native = torch.ones((b, 1, h, w), device=x.device, dtype=x.dtype)
    m_clip_work = min_pool_exact(m_clip_native, h_w, w_w)

    if m_roi_native is None:
        m_roi_nat_4d = torch.ones((b, 1, h, w), device=x.device, dtype=x.dtype)
        m_roi_work = torch.ones((b, 1, h_w, w_w), device=x.device, dtype=x.dtype)
    else:
        m_roi_nat_4d, _ = _ensure_4d(m_roi_native.to(device=x.device, dtype=x.dtype))
        m_roi_work = bilinear_resize(m_roi_nat_4d, h_w, w_w)

    m_in = m_tex_work * m_roi_work * m_clip_work
    m_final = m_tex_native * m_roi_nat_4d * m_clip_native

    return WorkMaskBundle(
        work_h=h_w,
        work_w=w_w,
        m_tex_ref=m_tex_ref,
        w_flat_ref=w_flat_ref,
        m_tex_work=m_tex_work,
        w_flat_work=w_flat_work,
        m_tex_native=m_tex_native,
        w_flat_native=w_flat_native,
        m_clip_native=m_clip_native,
        m_clip_work=m_clip_work,
        m_roi_work=m_roi_work,
        m_roi_native=m_roi_nat_4d,
        m_in=m_in,
        m_final=m_final,
    )


# ---------------------------------------------------------------------------
# Section 3.4: 64x64 Void-and-Cluster Blue-Noise Dithering & Residual Injection
# ---------------------------------------------------------------------------


def _torus_gaussian_kernel_64(sigma: float = 1.5) -> np.ndarray:
    n = 64
    coords = np.arange(n, dtype=np.float64)
    d = np.minimum(coords, n - coords)
    dx2 = d[:, None] ** 2 + d[None, :] ** 2
    kernel = np.exp(-dx2 / (2.0 * sigma * sigma))
    kernel[0, 0] = 0.0  # Exclude self-contribution when finding tightest cluster / largest void
    return kernel


@lru_cache(maxsize=1)
def get_blue_noise_64x64() -> np.ndarray:
    """Deterministic Ulichney (1993) Void-and-Cluster 64x64 Blue-Noise threshold matrix in [0, 1).

    Returns float32 array of shape (64, 64) containing a permutation of {0..4095} / 4096.0.
    """
    n = 64
    total = n * n
    n_init = int(round(total * 0.10))  # ~10% initial seed points
    rng = np.random.Generator(np.random.PCG64(42))

    kernel = _torus_gaussian_kernel_64(sigma=1.5)
    # Include self weight in LUT or handle with additive energy updates
    binary = np.zeros((n, n), dtype=np.uint8)
    init_indices = rng.choice(total, size=n_init, replace=False)
    binary.reshape(-1)[init_indices] = 1

    # Compute initial energy field by summing shifted kernels
    energy = np.zeros((n, n), dtype=np.float64)
    ys, xs = np.where(binary == 1)
    for y, x in zip(ys, xs):
        energy += np.roll(np.roll(kernel, int(y), axis=0), int(x), axis=1)

    def _add_point(e: np.ndarray, b: np.ndarray, y: int, x: int) -> None:
        b[y, x] = 1
        e += np.roll(np.roll(kernel, y, axis=0), x, axis=1)

    def _remove_point(e: np.ndarray, b: np.ndarray, y: int, x: int) -> None:
        b[y, x] = 0
        e -= np.roll(np.roll(kernel, y, axis=0), x, axis=1)

    # Phase 0: Relax initial binary pattern
    for _ in range(total):
        e_ones = np.where(binary == 1, energy, -np.inf)
        cy, cx = np.unravel_index(int(np.argmax(e_ones)), (n, n))
        _remove_point(energy, binary, int(cy), int(cx))

        e_zeros = np.where(binary == 0, energy, np.inf)
        vy, vx = np.unravel_index(int(np.argmin(e_zeros)), (n, n))
        if int(vy) == int(cy) and int(vx) == int(cx):
            _add_point(energy, binary, int(cy), int(cx))
            break
        _add_point(energy, binary, int(vy), int(vx))

    rank = np.empty((n, n), dtype=np.int32)
    rank_work_bin = binary.copy()
    rank_work_e = energy.copy()

    # Phase 1: Remove tightest clusters from initial minority pattern (ranks n_init-1 down to 0)
    for r in range(n_init - 1, -1, -1):
        e_ones = np.where(rank_work_bin == 1, rank_work_e, -np.inf)
        cy, cx = np.unravel_index(int(np.argmax(e_ones)), (n, n))
        rank[cy, cx] = r
        _remove_point(rank_work_e, rank_work_bin, int(cy), int(cx))

    # Phase 2: Insert into largest voids up to half total pixels (ranks n_init .. total//2 - 1)
    phase2_bin = binary.copy()
    phase2_e = energy.copy()
    half = total // 2
    for r in range(n_init, half):
        e_zeros = np.where(phase2_bin == 0, phase2_e, np.inf)
        vy, vx = np.unravel_index(int(np.argmin(e_zeros)), (n, n))
        rank[vy, vx] = r
        _add_point(phase2_e, phase2_bin, int(vy), int(vx))

    # Phase 3: Invert pattern and remove tightest clusters of zeros (ranks half .. total - 1)
    inv_bin = 1 - phase2_bin
    inv_e = np.zeros((n, n), dtype=np.float64)
    ys, xs = np.where(inv_bin == 1)
    for y, x in zip(ys, xs):
        inv_e += np.roll(np.roll(kernel, int(y), axis=0), int(x), axis=1)

    for r in range(half, total):
        e_ones = np.where(inv_bin == 1, inv_e, -np.inf)
        cy, cx = np.unravel_index(int(np.argmax(e_ones)), (n, n))
        rank[cy, cx] = r
        _remove_point(inv_e, inv_bin, int(cy), int(cx))

    return (rank.astype(np.float32)) / float(total)


def get_blue_noise_tile(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Tile the 64x64 Blue-Noise threshold matrix to [1, 1, height, width] in [0, 1)."""
    bn = torch.from_numpy(get_blue_noise_64x64()).to(device=device, dtype=dtype)
    rep_h = (height + 63) // 64
    rep_w = (width + 63) // 64
    tiled = bn.repeat(rep_h, rep_w)[:height, :width]
    return tiled.unsqueeze(0).unsqueeze(0)


def inject_and_quantize(
    x_orig_01: torch.Tensor,
    hat_delta_work: torch.Tensor,
    m_tex_native: torch.Tensor,
    m_clip_native: torch.Tensor,
    lam: torch.Tensor | float = 1.0,
    m_roi_native: torch.Tensor | float = 1.0,
    alpha_mask: torch.Tensor | None = None,
    blue_noise_dither: bool = True,
    ste_quantize: bool = False,
    x_orig_u8: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Apply Catmull-Rom upsampling, direct residual addition, strict OR guard, and 8-bit dithered quantization.

    Implements DESIGN.md §3.1, §3.2, §3.3, and §3.4:
      - hat_delta_work in [-1, 1]^3 at (H_w, W_w) -> Catmull-Rom (a=-0.5) -> [H, W] clamped to [-1, 1].
      - m_final = m_tex_native * m_roi_native * m_clip_native
      - Delta_RGB = M_inv * (lam * m_final * eta * hat_delta_up)
      - Strict OR no-op guard: alpha==0 OR m_clip_native==0.0 OR |255 * Delta_RGB| < 1e-3 -> bit-exact original.
    """
    x_orig, was_3d = _ensure_4d(x_orig_01)
    d_work, _ = _ensure_4d(hat_delta_work)
    m_tex, _ = _ensure_4d(m_tex_native)
    m_clip, _ = _ensure_4d(m_clip_native)

    _, _, h, w = x_orig.shape
    hat_delta_up = catmull_rom_resize(d_work, h, w, clamp_unit=True)

    if isinstance(m_roi_native, torch.Tensor):
        m_roi_t, _ = _ensure_4d(m_roi_native.to(device=x_orig.device, dtype=x_orig.dtype))
    else:
        m_roi_t = float(m_roi_native)

    m_final = m_tex * m_roi_t * m_clip
    delta_rgb = delta_ycbcr_to_rgb(hat_delta_up, mask=m_final, lam=lam)

    x_cont = torch.clamp(x_orig + delta_rgb, 0.0, 1.0)

    # Strict OR guard: exact bit preservation for transparent, clamped edge, or near-zero delta pixels
    guard = (m_clip == 0.0) | (delta_rgb.abs() * 255.0 < NO_OP_GUARD_EPS_255)
    if alpha_mask is not None:
        a_t, _ = _ensure_4d(alpha_mask.to(device=x_orig.device))
        guard = guard | (a_t == 0)

    if blue_noise_dither:
        bn = get_blue_noise_tile(h, w, device=x_orig.device, dtype=x_orig.dtype)
        dither = bn - 0.5  # [-0.5, 0.5)
    else:
        dither = 0.0

    with torch.no_grad():
        q_float = torch.clamp(torch.floor(x_orig * 255.0 + 255.0 * delta_rgb + dither + 0.5), 0.0, 255.0)
        if x_orig_u8 is not None:
            orig_u8_4d, _ = _ensure_4d(x_orig_u8.to(device=x_orig.device))
            orig_q = orig_u8_4d.to(dtype=q_float.dtype)
        else:
            orig_q = torch.clamp(torch.round(x_orig * 255.0), 0.0, 255.0)
        x_q_255 = torch.where(guard, orig_q, q_float)
        x_q_u8 = x_q_255.to(torch.uint8)
        x_q_01 = x_q_255 / 255.0

    if ste_quantize:
        x_out = x_cont + (x_q_01 - x_cont).detach()
    else:
        x_out = x_q_01

    delta_eff_norm = hat_delta_up * m_final * (
        lam.view(-1, 1, 1, 1) if isinstance(lam, torch.Tensor) and lam.ndim == 1 else lam
    )

    return {
        "x_out": _restore_shape(x_out, was_3d),
        "x_cont": _restore_shape(x_cont, was_3d),
        "x_u8": _restore_shape(x_q_u8, was_3d),
        "delta_rgb": _restore_shape(delta_rgb, was_3d),
        "delta_eff_norm": _restore_shape(delta_eff_norm, was_3d),
        "hat_delta_up": _restore_shape(hat_delta_up, was_3d),
        "m_final": _restore_shape(m_final, was_3d),
        "guard": _restore_shape(guard, was_3d),
    }


# ---------------------------------------------------------------------------
# Section 3.4.1: Flat-Area Low-Frequency Chroma Penalty
# ---------------------------------------------------------------------------


def gaussian_lowpass_2d(x: torch.Tensor, sigma: float = CHROMA_LP_SIGMA) -> torch.Tensor:
    """Separable 2D Gaussian low-pass filter with reflect padding."""
    t, was_3d = _ensure_4d(x)
    if sigma <= 0.0:
        return _restore_shape(t, was_3d)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=t.dtype, device=t.device)
    k1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    k1d = k1d / k1d.sum()

    c = t.shape[1]
    kh = k1d.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    kv = k1d.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    padded = F.pad(t, (radius, radius, radius, radius), mode="reflect")
    blurred = F.conv2d(padded, kh, groups=c)
    blurred = F.conv2d(blurred, kv, groups=c)
    return _restore_shape(blurred, was_3d)


def flat_low_freq_chroma_loss(
    delta_eff: torch.Tensor,
    w_flat: torch.Tensor,
    beta_chroma: float,
    sigma: float = CHROMA_LP_SIGMA,
) -> torch.Tensor:
    """Compute L_chroma per DESIGN.md §3.4.1:

    L_chroma(delta_eff) = beta_chroma / (2 H W) * sum_{c in {Cb, Cr}} || W_flat * LP_{sigma=3.0}(delta_eff, c) ||_2^2.
    """
    d_4d, _ = _ensure_4d(delta_eff)
    wf_4d, _ = _ensure_4d(w_flat.to(device=d_4d.device, dtype=d_4d.dtype))
    if wf_4d.shape[-2:] != d_4d.shape[-2:]:
        wf_4d = bilinear_resize(wf_4d, d_4d.shape[-2], d_4d.shape[-1])
    chroma = d_4d[:, 1:3]
    lp_chroma = gaussian_lowpass_2d(chroma, sigma=sigma)
    weighted = wf_4d * lp_chroma
    return float(beta_chroma) * (weighted * weighted).mean()


# ---------------------------------------------------------------------------
# Quality, Invisibility, and Spectral Metrics (§0.2, §2.2, §3.1, §7)
# ---------------------------------------------------------------------------


def compute_psnr(clean_01: torch.Tensor, prot_01: torch.Tensor) -> float:
    """Compute PSNR (dB) on [0, 1] RGB tensors."""
    c, _ = _ensure_4d(clean_01.detach().to(torch.float64))
    p, _ = _ensure_4d(prot_01.detach().to(torch.float64))
    mse = torch.mean((c - p) ** 2).item()
    if mse <= 1e-16:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def compute_ssim(clean_01: torch.Tensor, prot_01: torch.Tensor) -> float:
    """Compute mean separable Gaussian SSIM (11x11, sigma=1.5) on [0, 1] RGB tensors."""
    c, _ = _ensure_4d(clean_01.detach().to(torch.float64))
    p, _ = _ensure_4d(prot_01.detach().to(torch.float64))

    radius = 5
    sigma = 1.5
    coords = torch.arange(-radius, radius + 1, dtype=torch.float64, device=c.device)
    k1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    k1d = k1d / k1d.sum()
    ch = c.shape[1]
    kh = k1d.view(1, 1, 1, -1).expand(ch, 1, 1, -1)
    kv = k1d.view(1, 1, -1, 1).expand(ch, 1, -1, 1)

    def _blur(t: torch.Tensor) -> torch.Tensor:
        pad = F.pad(t, (radius, radius, radius, radius), mode="reflect")
        return F.conv2d(F.conv2d(pad, kh, groups=ch), kv, groups=ch)

    mu1 = _blur(c)
    mu2 = _blur(p)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _blur(c * c) - mu1_sq
    sigma2_sq = _blur(p * p) - mu2_sq
    sigma12 = _blur(c * p) - mu1_mu2

    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / den
    return float(ssim_map.mean().item())


def _srgb_to_lab(rgb_01: np.ndarray) -> np.ndarray:
    """Convert [..., 3] sRGB in [0, 1] to CIE L*a*b* (D65)."""
    rgb = np.clip(rgb_01, 0.0, 1.0).astype(np.float64)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = lin @ m.T
    ref_white = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
    xyz_n = xyz / ref_white

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz_n > eps, np.cbrt(xyz_n), (kappa * xyz_n + 16.0) / 116.0)
    l_star = 116.0 * f[..., 1] - 16.0
    a_star = 500.0 * (f[..., 0] - f[..., 1])
    b_star = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([l_star, a_star, b_star], axis=-1)


def compute_ciede2000_map(clean_01: torch.Tensor, prot_01: torch.Tensor) -> np.ndarray:
    """Exact CIEDE2000 color difference map (Sharma et al. 2005, kL=kC=kH=1) over [B, H, W]."""
    c_4d, _ = _ensure_4d(clean_01.detach().cpu())
    p_4d, _ = _ensure_4d(prot_01.detach().cpu())
    c_hwc = c_4d.permute(0, 2, 3, 1).numpy()
    p_hwc = p_4d.permute(0, 2, 3, 1).numpy()

    lab1 = _srgb_to_lab(c_hwc)
    lab2 = _srgb_to_lab(p_hwc)

    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)
    c_bar = 0.5 * (c1 + c2)
    c_bar7 = c_bar**7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + 25.0**7)))

    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = np.sqrt(a1p * a1p + b1 * b1)
    c2p = np.sqrt(a2p * a2p + b2 * b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    h1p = np.where((c1p == 0.0), 0.0, h1p)
    h2p = np.where((c2p == 0.0), 0.0, h2p)

    dlp = l2 - l1
    dcp = c2p - c1p

    c_prod = c1p * c2p
    dh_raw = h2p - h1p
    dhp = np.where(
        c_prod == 0.0,
        0.0,
        np.where(
            np.abs(dh_raw) <= 180.0,
            dh_raw,
            np.where(dh_raw > 180.0, dh_raw - 360.0, dh_raw + 360.0),
        ),
    )
    d_hp = 2.0 * np.sqrt(c_prod) * np.sin(np.radians(0.5 * dhp))

    lp_bar = 0.5 * (l1 + l2)
    cp_bar = 0.5 * (c1p + c2p)

    h_sum = h1p + h2p
    hp_bar = np.where(
        c_prod == 0.0,
        h_sum,
        np.where(
            np.abs(dh_raw) <= 180.0,
            0.5 * h_sum,
            np.where(h_sum < 360.0, 0.5 * (h_sum + 360.0), 0.5 * (h_sum - 360.0)),
        ),
    )

    t = (
        1.0
        - 0.17 * np.cos(np.radians(hp_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hp_bar))
        + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0))
    )
    d_theta = 30.0 * np.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    cp_bar7 = cp_bar**7
    r_c = 2.0 * np.sqrt(cp_bar7 / (cp_bar7 + 25.0**7))

    l_m50_sq = (lp_bar - 50.0) ** 2
    s_l = 1.0 + (0.015 * l_m50_sq) / np.sqrt(20.0 + l_m50_sq)
    s_c = 1.0 + 0.045 * cp_bar
    s_h = 1.0 + 0.015 * cp_bar * t
    r_t = -np.sin(np.radians(2.0 * d_theta)) * r_c

    de00 = np.sqrt(
        (dlp / s_l) ** 2
        + (dcp / s_c) ** 2
        + (d_hp / s_h) ** 2
        + r_t * (dcp / s_c) * (d_hp / s_h)
    )
    return de00


def compute_flat_ciede2000_p95(
    clean_01: torch.Tensor,
    prot_01: torch.Tensor,
    w_flat: torch.Tensor | None = None,
    flat_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute CIEDE2000 p95 over flat regions (W_flat >= flat_threshold) and all pixels."""
    de_map = compute_ciede2000_map(clean_01, prot_01)
    all_p95 = float(np.percentile(de_map, 95.0))
    all_mean = float(np.mean(de_map))

    if w_flat is not None:
        wf, _ = _ensure_4d(w_flat.detach().cpu())
        if wf.shape[-2:] != de_map.shape[-2:]:
            wf = bilinear_resize(wf, de_map.shape[-2], de_map.shape[-1])
        wf_np = wf[:, 0].numpy()
        flat_mask = wf_np >= flat_threshold
        if np.any(flat_mask):
            flat_vals = de_map[flat_mask]
            flat_p95 = float(np.percentile(flat_vals, 95.0))
            flat_mean = float(np.mean(flat_vals))
            flat_frac = float(np.mean(flat_mask))
        else:
            flat_p95 = all_p95
            flat_mean = all_mean
            flat_frac = 0.0
    else:
        flat_p95 = all_p95
        flat_mean = all_mean
        flat_frac = 1.0

    return {
        "flat_delta_e2000_p95": flat_p95,
        "flat_delta_e2000_mean": flat_mean,
        "all_delta_e2000_p95": all_p95,
        "flat_pixel_fraction": flat_frac,
    }


def compute_delta_stats(clean_01: torch.Tensor, prot_01: torch.Tensor) -> dict[str, float]:
    """Compute RGB and YCbCr perturbation magnitudes in 255-scaled units (§3.1)."""
    c, _ = _ensure_4d(clean_01.detach().to(torch.float64))
    p, _ = _ensure_4d(prot_01.detach().to(torch.float64))
    diff_rgb = p - c
    max_abs_rgb_255 = float(diff_rgb.abs().max().item() * 255.0)
    rgb_rms_255 = float(torch.sqrt(torch.mean(diff_rgb * diff_rgb)).item() * 255.0)

    diff_ycbcr = rgb_to_ycbcr(p) - rgb_to_ycbcr(c)
    max_y_255 = float(diff_ycbcr[:, 0].abs().max().item() * 255.0)
    max_cb_255 = float(diff_ycbcr[:, 1].abs().max().item() * 255.0)
    max_cr_255 = float(diff_ycbcr[:, 2].abs().max().item() * 255.0)

    return {
        "max_abs_delta_rgb_255": max_abs_rgb_255,
        "rgb_rms_255": rgb_rms_255,
        "max_abs_delta_y_255": max_y_255,
        "max_abs_delta_cb_255": max_cb_255,
        "max_abs_delta_cr_255": max_cr_255,
    }


def compute_spectral_metrics(
    delta_rgb_or_y: torch.Tensor,
    lf_cutoff_cycles_per_px: float = 0.11,
) -> dict[str, float]:
    """Compute 512-normalized low-frequency energy ratio rho_LF and 2x2 grid f=0.5 peak checks (§0.2.2, §2.2).

    Inputs are resampled to a 512x512 reference grid via Exact Area Box (down) / Bilinear (up)
    when dimensions differ from 512x512, matching the 512-grid definition in DESIGN.md §0.2.2.
    """
    d, _ = _ensure_4d(delta_rgb_or_y.detach().to(torch.float64))
    _, _, h, w = d.shape
    if (h, w) != (512, 512):
        if h >= 512 and w >= 512:
            d_ref = exact_area_box_resize(d, 512, 512)
        else:
            d_ref = bilinear_resize(d, 512, 512)
    else:
        d_ref = d

    fft = torch.fft.fft2(d_ref, dim=(-2, -1))
    power = (fft.real**2 + fft.imag**2).sum(dim=1)  # [B, 512, 512]

    fu = torch.fft.fftfreq(512, d=1.0, device=d.device, dtype=torch.float64)
    fv = torch.fft.fftfreq(512, d=1.0, device=d.device, dtype=torch.float64)
    fr = torch.sqrt(fu[:, None] ** 2 + fv[None, :] ** 2)

    lf_mask = fr <= lf_cutoff_cycles_per_px
    total_power = power.sum(dim=(-2, -1)).clamp_min(1e-12)
    lf_power = power[:, lf_mask].sum(dim=-1)
    rho_lf = float((lf_power / total_power).mean().item())

    # Monitor 2x2 checkerboard / Nyquist grid artifact near f = 0.5 cycles/px
    nyq_mask = (torch.abs(fu[:, None]) >= 0.45) | (torch.abs(fv[None, :]) >= 0.45)
    nyq_band_ratio = float((power[:, nyq_mask].sum(dim=-1) / total_power).mean().item())

    # Exact f=0.5 indices in 512x512 FFT: 256 corresponds to -0.5 cycles/px
    p_05_05 = power[:, 256, 256]
    p_05_00 = power[:, 256, 0]
    p_00_05 = power[:, 0, 256]
    nyq_exact_max = torch.maximum(p_05_05, torch.maximum(p_05_00, p_00_05))
    median_power = power.flatten(1).median(dim=-1).values.clamp_min(1e-12)
    checker_peak_to_median = float((nyq_exact_max / median_power).mean().item())
    checker_exact_ratio = float(((p_05_05 + p_05_00 + p_00_05) / total_power).mean().item())

    return {
        "rho_lf": rho_lf,
        "nyquist_band_ratio": nyq_band_ratio,
        "checkerboard_f05_exact_ratio": checker_exact_ratio,
        "checkerboard_f05_peak_to_median": checker_peak_to_median,
    }


def evaluate_perceptual_metrics(clean_01: torch.Tensor, prot_01: torch.Tensor) -> dict[str, Any]:
    """Evaluate optional LPIPS and SSIMULACRA2 metrics if installed and weights are available."""
    results: dict[str, Any] = {
        "lpips": None,
        "ssimulacra2": None,
        "lpips_status": "not_installed",
        "ssimulacra2_status": "not_installed",
    }
    c, _ = _ensure_4d(clean_01.detach().to(torch.float32))
    p, _ = _ensure_4d(prot_01.detach().to(torch.float32))

    try:
        import lpips  # type: ignore

        model = lpips.LPIPS(net="alex", verbose=False).to(c.device)
        model.eval()
        with torch.no_grad():
            val = model(c * 2.0 - 1.0, p * 2.0 - 1.0).mean().item()
        results["lpips"] = float(val)
        results["lpips_status"] = "ok"
    except Exception as exc:
        results["lpips_status"] = f"unavailable:{type(exc).__name__}"

    try:
        import pyiqa  # type: ignore

        metric = pyiqa.create_metric("ssimulacra2", device=c.device)
        with torch.no_grad():
            val = metric(p, c).mean().item()
        results["ssimulacra2"] = float(val)
        results["ssimulacra2_status"] = "ok"
    except Exception as exc:
        results["ssimulacra2_status"] = f"unavailable:{type(exc).__name__}"

    return results


def evaluate_image_quality(
    clean_01: torch.Tensor,
    prot_01: torch.Tensor,
    w_flat: torch.Tensor | None = None,
    include_optional_perceptual: bool = False,
) -> dict[str, Any]:
    """Compute complete DESIGN.md quality and spectral metrics."""
    metrics: dict[str, Any] = {
        "psnr_db": compute_psnr(clean_01, prot_01),
        "ssim": compute_ssim(clean_01, prot_01),
    }
    metrics.update(compute_flat_ciede2000_p95(clean_01, prot_01, w_flat=w_flat))
    metrics.update(compute_delta_stats(clean_01, prot_01))
    metrics.update(compute_spectral_metrics(prot_01 - clean_01))
    if include_optional_perceptual:
        metrics.update(evaluate_perceptual_metrics(clean_01, prot_01))
    return metrics

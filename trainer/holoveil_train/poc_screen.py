"""Phase 0a CLI & Screening Pipeline: Step 1 Target Gates + Step 2 Direct PGD Pareto Sweep.

Implements DESIGN.md §5.2 and §7 Phase 0a:
  - --step1-only: Run only Step 1 (stitch fidelity gate >= 0.90 & r_t spatial autocorrelation
    across ell_c in {48, 96, 160}px, selecting shortest ell_c* with rho(32px) >= 0.50).
  - --step2: Run Direct PGD (N=100, alpha_vae=0.35, lambda in {0.65, 0.72, 0.85, 1.00},
    beta_chroma in {1.0, 2.0, 4.0}, Standard512 vs Bucket1024) and output top-3 Pareto configs.
  - --dry-run-synthetic: Run full Step 1 + Step 2 pipeline on CPU using deterministic synthetic
    surrogate and synthetic multi-domain test images (clearly marked synthetic; never real SD 1.5 data).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image

from holoveil_train import DESIGN_VERSION, seed_everything
from holoveil_train.color_mask import (
    ScaleTargetType,
    bilinear_resize,
    build_masks,
    compute_work_grid,
    evaluate_image_quality,
    exact_area_box_resize,
    flat_low_freq_chroma_loss,
    inject_and_quantize,
    round_mult8,
    round_mult16,
)
from holoveil_train.rewards import (
    AUTOCORR_DELTAS_PX,
    AUTOCORR_THRESHOLD_32PX,
    DISCRETE_SCALES,
    ELL_C_CANDIDATES_PX,
    LATENT_DOWNSAMPLE,
    MASTER_SIZE,
    STITCH_COSINE_THRESHOLD,
    VAE_LOSS_WEIGHTS,
    EmaNormalizedTrpLoss,
    ScaleSpec,
    SD15MultiVaeSurrogate,
    SurrogateBackend,
    SyntheticDryRunSurrogate,
    build_rt_master_for_ell_c,
    build_target_cache,
    get_bucket1024_valid_scales,
    get_lpid_valid_scales,
    get_scale_spec,
    measure_rt_autocorrelation,
    multi_vae_relative_l2_loss,
    select_shortest_valid_ell_c,
    slice_target_latent,
    verify_rt_stitch_fidelity,
)

# ---------------------------------------------------------------------------
# Differentiable JPEG STE (q=75) & Real PIL JPEG q=75 Evaluation Helper
# ---------------------------------------------------------------------------

_JPEG_LUMA_Q50 = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)

_JPEG_CHROMA_Q50 = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.float32,
)


def _scale_jpeg_qtable(base_q50: np.ndarray, quality: int) -> np.ndarray:
    q = max(1, min(100, int(quality)))
    factor = 5000.0 / q if q < 50 else 200.0 - 2.0 * q
    table = np.floor((base_q50 * factor + 50.0) / 100.0)
    return np.clip(table, 1.0, 255.0).astype(np.float32)


def _build_dct_8x8_matrix() -> np.ndarray:
    n = 8
    k = np.arange(n, dtype=np.float32)[:, None]
    i = np.arange(n, dtype=np.float32)[None, :]
    alpha = np.where(k == 0, np.sqrt(1.0 / n), np.sqrt(2.0 / n))
    return alpha * np.cos(np.pi * (2.0 * i + 1.0) * k / (2.0 * n))


class DiffJpegSte(nn.Module):
    """Differentiable 4:2:0 JPEG compression with Straight-Through Estimator rounding."""

    def __init__(self, quality: int = 75) -> None:
        super().__init__()
        self.quality = int(quality)
        dct = torch.from_numpy(_build_dct_8x8_matrix())
        q_y = torch.from_numpy(_scale_jpeg_qtable(_JPEG_LUMA_Q50, self.quality))
        q_c = torch.from_numpy(_scale_jpeg_qtable(_JPEG_CHROMA_Q50, self.quality))
        self.register_buffer("dct", dct)
        self.register_buffer("q_y", q_y)
        self.register_buffer("q_c", q_c)

    @staticmethod
    def _ste_round(x: torch.Tensor) -> torch.Tensor:
        return x + (torch.round(x) - x).detach()

    def _block_dct_quant_idct(self, plane: torch.Tensor, q_table: torch.Tensor) -> torch.Tensor:
        # plane: [B, H, W] in [0, 255] centered by -128
        b, h, w = plane.shape
        x = plane - 128.0
        blocks = x.view(b, h // 8, 8, w // 8, 8).permute(0, 1, 3, 2, 4).reshape(-1, 8, 8)
        dct_m = self.dct.to(dtype=plane.dtype, device=plane.device)
        q_t = q_table.to(dtype=plane.dtype, device=plane.device)
        coeff = dct_m @ blocks @ dct_m.T
        q_coeff = self._ste_round(coeff / q_t) * q_t
        rec = dct_m.T @ q_coeff @ dct_m
        rec_plane = rec.reshape(b, h // 8, w // 8, 8, 8).permute(0, 1, 3, 2, 4).reshape(b, h, w)
        return rec_plane + 128.0

    def forward(self, rgb_01: torch.Tensor) -> torch.Tensor:
        b, _, h, w = rgb_01.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x_in = F.pad(rgb_01, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            x_in = rgb_01

        x_255 = x_in * 255.0
        r, g, b_ch = x_255[:, 0], x_255[:, 1], x_255[:, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b_ch
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b_ch + 128.0
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b_ch + 128.0

        y_q = self._block_dct_quant_idct(y, self.q_y)
        cb_sub = F.avg_pool2d(cb.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        cr_sub = F.avg_pool2d(cr.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        cb_q_sub = self._block_dct_quant_idct(cb_sub, self.q_c)
        cr_q_sub = self._block_dct_quant_idct(cr_sub, self.q_c)

        cb_up = F.interpolate(cb_q_sub.unsqueeze(1), size=y_q.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        cr_up = F.interpolate(cr_q_sub.unsqueeze(1), size=y_q.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

        cb_c = cb_up - 128.0
        cr_c = cr_up - 128.0
        r_out = y_q + 1.402 * cr_c
        g_out = y_q - 0.344136 * cb_c - 0.714136 * cr_c
        b_out = y_q + 1.772 * cb_c
        rgb_rec = torch.clamp(torch.stack([r_out, g_out, b_out], dim=1) / 255.0, 0.0, 1.0)
        return rgb_rec[:, :, :h, :w]


def real_jpeg_recompress(rgb_01: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """Apply real PIL JPEG recompression (4:2:0) for Phase 0a Step 2 proxy evaluation."""
    outs = []
    for idx in range(rgb_01.shape[0]):
        arr = (rgb_01[idx].detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        pil_img = Image.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=int(quality), subsampling="4:2:0")
        buf.seek(0)
        dec = Image.open(buf).convert("RGB")
        dec_arr = np.asarray(dec, dtype=np.float32) / 255.0
        outs.append(torch.from_numpy(dec_arr).permute(2, 0, 1))
    return torch.stack(outs, dim=0).to(device=rgb_01.device, dtype=rgb_01.dtype)


# ---------------------------------------------------------------------------
# Purification Operators & s_eff Target Slicing (DESIGN.md §5.2 Table)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PurificationOutcome:
    """Purified images and exact aligned target slices for TRP & Multi-VAE evaluation."""

    op_name: str
    sd15_input: torch.Tensor                 # Image tensor fed to SD 1.5 TRP & SD 1.5 VAE
    extra_vae_input: torch.Tensor            # Image tensor fed to SDXL / Flux VAEs
    r_t_aligned: torch.Tensor                # [1, 4, h_trp, w_trp]
    vae_targets_aligned: dict[str, torch.Tensor]
    metadata: dict[str, Any]


def apply_purification_with_targets(
    x_work_01: torch.Tensor,
    op_name: str,
    y_base: int,
    x_base: int,
    target_tensors: dict[str, torch.Tensor],
    rng: torch.Generator,
    diff_jpeg: DiffJpegSte | None = None,
    lpid_candidate_scales: tuple[float, ...] = (1.15, 1.30, 1.50, 1.75, 2.00),
    lpid_low_sizes: tuple[int, ...] = (112, 160, 224, 256),
    bucket1024_candidate_scales: tuple[float, ...] = (1.50, 1.75, 2.00, 2.25, 2.50),
    bucket1024_max_scaled_pixels: int = 1_200_000,
    forced_scale: float | None = None,
    forced_low_size: int | None = None,
) -> PurificationOutcome:
    """Apply a purification operator from DESIGN.md §5.2 and extract exact s_eff-aligned targets."""
    _, _, h_b, w_b = x_work_01.shape
    h_lat = h_b // LATENT_DOWNSAMPLE
    w_lat = w_b // LATENT_DOWNSAMPLE
    spec_1_0 = get_scale_spec(1.00)

    def _collect_vaes(spec: ScaleSpec, y_px: int, x_px: int, patch_h_lat: int, patch_w_lat: int) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, tensor in target_tensors.items():
            if not key.startswith("zt.") or not key.endswith(f".s_{spec.tag}"):
                continue
            vae_name = key.split(".")[1]
            out[vae_name] = slice_target_latent(tensor, y_px, x_px, patch_h_lat, patch_w_lat, spec)
        return out

    if op_name == "Identity":
        sd_in = x_work_01
        extra_in = x_work_01
        if max(h_b, w_b) > 512 and (h_b * w_b > 512 * 512):
            # 768..1216 bucket SD 1.5 TRP uses a random 512x512 window within (H_b, W_b)
            max_yw = max(0, (h_b - 512) // 8)
            max_xw = max(0, (w_b - 512) // 8)
            y_win = int(torch.randint(0, max_yw + 1, (1,), generator=rng).item()) * 8 if max_yw > 0 else 0
            x_win = int(torch.randint(0, max_xw + 1, (1,), generator=rng).item()) * 8 if max_xw > 0 else 0
            sd_in = x_work_01[:, :, y_win : y_win + 512, x_win : x_win + 512]
            r_t = slice_target_latent(target_tensors["rt.s_1_00"], y_base + y_win, x_base + x_win, 64, 64, spec_1_0)
            vaes = _collect_vaes(spec_1_0, y_base, x_base, h_lat, w_lat)
            if "sd15" in vaes:
                vaes["sd15"] = slice_target_latent(
                    target_tensors["zt.sd15.s_1_00"], y_base + y_win, x_base + x_win, 64, 64, spec_1_0
                )
        else:
            r_t = slice_target_latent(target_tensors["rt.s_1_00"], y_base, x_base, h_lat, w_lat, spec_1_0)
            vaes = _collect_vaes(spec_1_0, y_base, x_base, h_lat, w_lat)
        return PurificationOutcome(op_name, sd_in, extra_in, r_t, vaes, {"s_eff": 1.0})

    if op_name == "DiffJPEG_STE":
        jpeg_mod = diff_jpeg if diff_jpeg is not None else DiffJpegSte(quality=75).to(x_work_01.device)
        purified = jpeg_mod(x_work_01)
        r_t = slice_target_latent(target_tensors["rt.s_1_00"], y_base, x_base, h_lat, w_lat, spec_1_0)
        vaes = _collect_vaes(spec_1_0, y_base, x_base, h_lat, w_lat)
        return PurificationOutcome(op_name, purified, purified, r_t, vaes, {"s_eff": 1.0})

    if op_name == "BilinearDownUp":
        alpha = 0.50 + 0.25 * float(torch.rand((1,), generator=rng).item())
        h_mid = max(16, round_mult8(h_b * alpha))
        w_mid = max(16, round_mult8(w_b * alpha))
        down = F.interpolate(x_work_01, size=(h_mid, w_mid), mode="bilinear", align_corners=False)
        purified = F.interpolate(down, size=(h_b, w_b), mode="bilinear", align_corners=False)
        r_t = slice_target_latent(target_tensors["rt.s_1_00"], y_base, x_base, h_lat, w_lat, spec_1_0)
        vaes = _collect_vaes(spec_1_0, y_base, x_base, h_lat, w_lat)
        return PurificationOutcome(op_name, purified, purified, r_t, vaes, {"alpha": alpha, "s_eff": 1.0})

    if op_name == "GaussianNoise":
        sigma = 0.01 + 0.03 * float(torch.rand((1,), generator=rng).item())
        noise = torch.randn(x_work_01.shape, generator=rng, dtype=torch.float32).to(
            device=x_work_01.device, dtype=x_work_01.dtype
        )
        purified = torch.clamp(x_work_01 + sigma * noise, 0.0, 1.0)
        r_t = slice_target_latent(target_tensors["rt.s_1_00"], y_base, x_base, h_lat, w_lat, spec_1_0)
        vaes = _collect_vaes(spec_1_0, y_base, x_base, h_lat, w_lat)
        return PurificationOutcome(op_name, purified, purified, r_t, vaes, {"sigma": sigma, "s_eff": 1.0})

    if op_name == "LPID_CropResize":
        valid = get_lpid_valid_scales(h_b, w_b, candidate_scales=lpid_candidate_scales)
        if not valid:
            # Extreme skinny fallback to Identity if min(H_b, W_b) < min s_crop
            return apply_purification_with_targets(
                x_work_01, "Identity", y_base, x_base, target_tensors, rng, diff_jpeg=diff_jpeg
            )
        if forced_scale is not None:
            matched = [pair for pair in valid if abs(pair[0].s_nominal - forced_scale) < 1e-6]
            spec_s, s_crop = matched[0] if matched else valid[0]
        else:
            pick = int(torch.randint(0, len(valid), (1,), generator=rng).item())
            spec_s, s_crop = valid[pick]

        max_y0 = max(0, (h_b - s_crop) // 8)
        max_x0 = max(0, (w_b - s_crop) // 8)
        y0 = int(torch.randint(0, max_y0 + 1, (1,), generator=rng).item()) * 8 if max_y0 > 0 else 0
        x0 = int(torch.randint(0, max_x0 + 1, (1,), generator=rng).item()) * 8 if max_x0 > 0 else 0

        if forced_low_size is not None:
            s_low = int(forced_low_size)
        else:
            low_list = list(lpid_low_sizes)
            s_low = int(low_list[int(torch.randint(0, len(low_list), (1,), generator=rng).item())])

        crop = x_work_01[:, :, y0 : y0 + s_crop, x0 : x0 + s_crop]
        low = F.interpolate(crop, size=(s_low, s_low), mode="bilinear", align_corners=False)
        purified = F.interpolate(low, size=(512, 512), mode="bilinear", align_corners=False)

        r_t = slice_target_latent(target_tensors[f"rt.s_{spec_s.tag}"], y_base + y0, x_base + x0, 64, 64, spec_s)
        vaes = _collect_vaes(spec_s, y_base + y0, x_base + x0, 64, 64)
        return PurificationOutcome(
            op_name,
            purified,
            purified,
            r_t,
            vaes,
            {
                "s_nominal": spec_s.s_nominal,
                "s_eff": spec_s.s_eff,
                "s_crop": s_crop,
                "s_low": s_low,
                "y0": y0,
                "x0": x0,
            },
        )

    if op_name == "Bucket1024Scale":
        valid = get_bucket1024_valid_scales(
            h_b,
            w_b,
            candidate_scales=bucket1024_candidate_scales,
            max_scaled_pixels=bucket1024_max_scaled_pixels,
        )
        if not valid:
            return apply_purification_with_targets(
                x_work_01, "Identity", y_base, x_base, target_tensors, rng, diff_jpeg=diff_jpeg
            )
        if forced_scale is not None:
            matched = [item for item in valid if abs(item[0].s_nominal - forced_scale) < 1e-6]
            spec_s, h_s, w_s = matched[0] if matched else valid[0]
        else:
            pick = int(torch.randint(0, len(valid), (1,), generator=rng).item())
            spec_s, h_s, w_s = valid[pick]

        upscaled = bilinear_resize(x_work_01, h_s, w_s)
        h_s_lat = h_s // LATENT_DOWNSAMPLE
        w_s_lat = w_s // LATENT_DOWNSAMPLE

        # SD 1.5 TRP stays at native working grid (H_b, W_b); SDXL/Flux VAEs evaluate at scaled (H_s, W_s)
        r_t = slice_target_latent(target_tensors["rt.s_1_00"], y_base, x_base, h_lat, w_lat, spec_1_0)
        vaes = _collect_vaes(spec_s, y_base, x_base, h_s_lat, w_s_lat)
        if f"zt.sd15.s_{spec_1_0.tag}" in target_tensors:
            vaes["sd15"] = slice_target_latent(
                target_tensors[f"zt.sd15.s_{spec_1_0.tag}"], y_base, x_base, h_lat, w_lat, spec_1_0
            )
        return PurificationOutcome(
            op_name,
            x_work_01,
            upscaled,
            r_t,
            vaes,
            {
                "s_nominal": spec_s.s_nominal,
                "s_eff": spec_s.s_eff,
                "h_scaled": h_s,
                "w_scaled": w_s,
            },
        )

    if op_name == "Downscale05x_SD15":
        spec_05 = get_scale_spec(0.50)
        h_half = max(16, round_mult8(h_b * spec_05.s_eff))
        w_half = max(16, round_mult8(w_b * spec_05.s_eff))
        down = exact_area_box_resize(x_work_01, h_half, w_half)
        h_half_lat = h_half // LATENT_DOWNSAMPLE
        w_half_lat = w_half // LATENT_DOWNSAMPLE

        r_t = slice_target_latent(
            target_tensors[f"rt.s_{spec_05.tag}"], y_base, x_base, h_half_lat, w_half_lat, spec_05
        )
        vaes = _collect_vaes(spec_1_0, y_base, x_base, h_lat, w_lat)
        if f"zt.sd15.s_{spec_05.tag}" in target_tensors:
            vaes["sd15"] = slice_target_latent(
                target_tensors[f"zt.sd15.s_{spec_05.tag}"], y_base, x_base, h_half_lat, w_half_lat, spec_05
            )
        return PurificationOutcome(
            op_name,
            down,
            x_work_01,
            r_t,
            vaes,
            {
                "s_nominal": spec_05.s_nominal,
                "s_eff": spec_05.s_eff,
                "h_half": h_half,
                "w_half": w_half,
            },
        )

    raise ValueError(f"Unknown purification operator: {op_name!r}")


# ---------------------------------------------------------------------------
# Step 1 & Step 2 Execution and Reporting
# ---------------------------------------------------------------------------


def _make_step1_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 0a Step 1: Target Verification Report",
        "",
        f"- **Design Version**: `{report['design_version']}`",
        f"- **SD 1.5 Model**: `{report['sd15_model_id']}`",
        f"- **SD 1.5 Revision SHA**: `{report['sd15_revision']}`",
        f"- **Mode**: `{'DRY-RUN SYNTHETIC' if report['dry_run_synthetic'] else 'REAL SD 1.5'}`",
        f"- **Selected Correlation Length $\\ell_c^*$**: `{report['selected_ell_c_px']}` px",
        f"- **Fallback Grid-Snap (256px) Recommended**: `{report['fallback_grid_snap_recommended']}`",
        f"- **Antithetic Variates (`+eps, -eps`)**: `{report.get('antithetic_variates', True)}` | **Global Noise Shared**: `{report.get('share_global_noise', True)}`",
        "",
        "## Gate 1: Window Stitching Fidelity (`verify_rt_stitch_fidelity`, Threshold >= 0.90)",
        "",
        "| $\\ell_c$ (px) | Mean Cosine (64x64) | Min (64x64) | Median (64x64) | Mean Cosine (Inner 48x48) | Min (48x48) | Gate Status |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for ell_c in report["ell_c_candidates_px"]:
        cand = report["candidates"][str(ell_c)]
        sf = cand["stitch_fidelity"]
        f64 = sf["full_64x64"]
        i48 = sf["inner_48x48"]
        status = "PASS" if sf["passed"] else "FAIL (Fallback)"
        lines.append(
            f"| {ell_c} | {f64['mean']:.4f} | {f64['min']:.4f} | {f64['median']:.4f} | "
            f"{i48['mean']:.4f} | {i48['min']:.4f} | {status} |"
        )

    lines.extend(
        [
            "",
            "## Gate 2: Spatial Autocorrelation $\\rho_{r_t}(\\Delta)$ (`measure_rt_autocorrelation`, Threshold $\\rho(32\\text{px}) \\ge 0.50$)",
            "",
            "| $\\ell_c$ (px) | $\\sigma_{\\text{corr}}$ (px) | $\\Delta=0$ | $\\Delta=8$ | $\\Delta=16$ | $\\Delta=32$ | $\\Delta=48$ | $\\Delta=64$ | $\\rho(32\\text{px}) \\ge 0.50$ |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for ell_c in report["ell_c_candidates_px"]:
        cand = report["candidates"][str(ell_c)]
        ac = cand["autocorrelation"]
        by_d = ac["by_delta_px"]
        status = "PASS" if ac["passes_32px_gate"] else "FAIL"
        lines.append(
            f"| {ell_c} | {ell_c / 2:.0f} | "
            f"{by_d.get('0', by_d.get(0, 0.0)):.4f} | "
            f"{by_d.get('8', by_d.get(8, 0.0)):.4f} | "
            f"{by_d.get('16', by_d.get(16, 0.0)):.4f} | "
            f"{by_d.get('32', by_d.get(32, 0.0)):.4f} | "
            f"{by_d.get('48', by_d.get(48, 0.0)):.4f} | "
            f"{by_d.get('64', by_d.get(64, 0.0)):.4f} | "
            f"{status} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_step1(
    cfg: dict[str, Any],
    output_dir: Path,
    surrogate: SurrogateBackend,
    dry_run_synthetic: bool = False,
) -> dict[str, Any]:
    """Execute Phase 0a Step 1: verify_rt_stitch_fidelity + measure_rt_autocorrelation."""
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    tf_cfg = cfg["target_field"]
    seed = int(cfg["seed"])
    ell_c_list = [int(v) for v in tf_cfg.get("ell_c_candidates_px", ELL_C_CANDIDATES_PX)]
    deltas_px = [int(v) for v in tf_cfg.get("autocorr_deltas_px", AUTOCORR_DELTAS_PX)]
    min_rho_32 = float(tf_cfg.get("autocorr_min_rho_at_32px", AUTOCORR_THRESHOLD_32PX))
    min_cos = float(tf_cfg.get("stitch_min_cosine", STITCH_COSINE_THRESHOLD))
    num_samples = int(tf_cfg.get("stitch_num_random_samples", 20))
    num_t = int(tf_cfg.get("rt_num_timesteps", 20))
    num_n = int(tf_cfg.get("rt_num_noises", 6))
    use_antithetic = bool(tf_cfg.get("use_antithetic_variates", True))
    share_global_noise = bool(tf_cfg.get("share_global_noise_step1", True))
    batch_size = int(tf_cfg.get("rt_eval_batch_size", 8))

    if dry_run_synthetic:
        num_t = min(num_t, 4)
        num_n = min(num_n, 2)

    candidates_results: dict[int, dict[str, Any]] = {}
    for ell_c in ell_c_list:
        x_raw, r_t_1_0, _ = build_rt_master_for_ell_c(
            ell_c_px=ell_c,
            surrogate=surrogate,
            seed=seed,
            num_timesteps=num_t,
            num_noises=num_n,
            batch_size=batch_size,
            antithetic=use_antithetic,
            share_global_noise=share_global_noise,
            device="cpu",
        )
        stitch_res = verify_rt_stitch_fidelity(
            x_raw=x_raw,
            r_t_master=r_t_1_0,
            surrogate=surrogate,
            num_samples=num_samples,
            seed=seed,
            min_cosine=min_cos,
            num_timesteps=num_t,
            num_noises=num_n,
            batch_size=batch_size,
            antithetic=use_antithetic,
            share_global_noise=share_global_noise,
        )
        autocorr_res = measure_rt_autocorrelation(
            r_t_master=r_t_1_0,
            deltas_px=deltas_px,
            threshold_32px=min_rho_32,
        )
        candidates_results[ell_c] = {
            "ell_c_px": ell_c,
            "sigma_corr_px": ell_c / 2.0,
            "stitch_fidelity": stitch_res,
            "autocorrelation": autocorr_res,
        }

    selected_ell_c = select_shortest_valid_ell_c(candidates_results, min_rho_32px=min_rho_32)
    fallback_grid_snap = False
    if selected_ell_c is not None:
        fallback_grid_snap = candidates_results[selected_ell_c]["stitch_fidelity"]["fallback_grid_snap_recommended"]

    report: dict[str, Any] = {
        "design_version": DESIGN_VERSION,
        "dry_run_synthetic": bool(dry_run_synthetic),
        "sd15_model_id": cfg["model"]["sd15_model_id"],
        "sd15_revision": getattr(surrogate, "sd15_revision", "unknown"),
        "seed": seed,
        "antithetic_variates": use_antithetic,
        "share_global_noise": share_global_noise,
        "elapsed_sec": round(time.time() - t0, 2),
        "ell_c_candidates_px": ell_c_list,
        "selected_ell_c_px": selected_ell_c,
        "fallback_grid_snap_recommended": bool(fallback_grid_snap),
        "candidates": {str(k): v for k, v in candidates_results.items()},
    }

    (output_dir / "step1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = _make_step1_markdown(report)
    (output_dir / "step1_report.md").write_text(md, encoding="utf-8")
    return report


def _generate_synthetic_images() -> list[dict[str, Any]]:
    """Generate deterministic multi-domain synthetic images for --dry-run-synthetic verification."""
    imgs: list[dict[str, Any]] = []

    # 1. Illustration-like: pure white background with anti-aliased black line art + flat skin tone region
    h, w = 512, 512
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    img_ill = torch.ones((1, 3, h, w), dtype=torch.float32)
    ring = torch.abs(torch.sqrt(xx * xx + yy * yy) - 0.55)
    line_alpha = torch.clamp(1.0 - ring / 0.015, 0.0, 1.0)
    img_ill = img_ill * (1.0 - line_alpha)
    skin_mask = (torch.sqrt(xx * xx + yy * yy) < 0.53).to(torch.float32)
    skin_color = torch.tensor([0.96, 0.82, 0.74]).view(1, 3, 1, 1)
    img_ill = torch.where(skin_mask > 0.5, skin_color * (1.0 - line_alpha), img_ill)
    imgs.append({"name": "synthetic_white_lineart_512x512", "domain": "illustration", "rgb": img_ill})

    # 2. Portrait-like rectangular 352x512 smooth gradient + subtle texture
    h2, w2 = 512, 352
    gy = torch.linspace(0.15, 0.85, h2).view(1, 1, h2, 1)
    gx = torch.linspace(0.20, 0.80, w2).view(1, 1, 1, w2)
    gen = torch.Generator(device="cpu").manual_seed(101)
    tex = 0.08 * torch.randn((1, 3, h2, w2), generator=gen)
    img_rect = torch.clamp(torch.cat([gy.expand(1, 1, h2, w2), gx.expand(1, 1, h2, w2), 0.5 * (gy + gx)], dim=1) + tex, 0.0, 1.0)
    imgs.append({"name": "synthetic_portrait_512x352", "domain": "photo", "rgb": img_rect})

    return imgs


def _infer_domain_from_path(p: Path) -> str:
    low = str(p).lower()
    if any(k in low for k in ("illust", "anime", "lineart", "manga")):
        return "illustration"
    if any(k in low for k in ("paint", "art")):
        return "painting"
    if any(k in low for k in ("photo", "face", "portrait")):
        return "photo"
    return "unknown"


def _load_real_images(image_dir: Path) -> list[dict[str, Any]]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in exts])
    if not files:
        raise FileNotFoundError(f"No image files found under {image_dir}")
    loaded: list[dict[str, Any]] = []
    for p in files:
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            loaded.append({"name": p.stem, "path": str(p), "domain": _infer_domain_from_path(p), "rgb": t})
    return loaded


def _sample_purification_op(probs: dict[str, float], rng: torch.Generator) -> str:
    names = list(probs.keys())
    weights = torch.tensor([float(probs[n]) for n in names], dtype=torch.float64)
    weights = weights / weights.sum()
    idx = int(torch.multinomial(weights, num_samples=1, generator=rng).item())
    return names[idx]


def _pick_master_origin(
    h_w: int,
    w_w: int,
    rng: torch.Generator,
    snap_to_256_grid: bool = False,
) -> tuple[int, int]:
    max_y = max(0, MASTER_SIZE - h_w)
    max_x = max(0, MASTER_SIZE - w_w)
    step = 256 if snap_to_256_grid else 8
    y_choices = max(1, (max_y // step) + 1)
    x_choices = max(1, (max_x // step) + 1)
    y_base = int(torch.randint(0, y_choices, (1,), generator=rng).item()) * step
    x_base = int(torch.randint(0, x_choices, (1,), generator=rng).item()) * step
    return y_base, x_base


def select_pareto_top_k(records: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
    """Filter configs passing invisibility gates and select top-k non-dominated Pareto configs."""
    eligible = [r for r in records if r["passes_invisibility_gates"]]
    if not eligible:
        return []

    # Objectives to maximize: psnr_db, -flat_delta_e2000_p95, proxy_protection_score
    def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
        va = (a["mean_psnr_db"], -a["mean_flat_delta_e2000_p95"], a["proxy_protection_score"])
        vb = (b["mean_psnr_db"], -b["mean_flat_delta_e2000_p95"], b["proxy_protection_score"])
        return all(x >= y for x, y in zip(va, vb)) and any(x > y for x, y in zip(va, vb))

    pareto_front: list[dict[str, Any]] = []
    for cand in eligible:
        if not any(_dominates(other, cand) for other in eligible if other is not cand):
            pareto_front.append(cand)

    pareto_sorted = sorted(pareto_front, key=lambda r: r["proxy_protection_score"], reverse=True)
    if len(pareto_sorted) >= top_k:
        return pareto_sorted[:top_k]

    remaining = [r for r in eligible if r not in pareto_sorted]
    remaining.sort(key=lambda r: r["proxy_protection_score"], reverse=True)
    return (pareto_sorted + remaining)[:top_k]


def _make_step2_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 0a Step 2: Direct PGD Pareto Screening Report",
        "",
        f"- **Design Version**: `{report['design_version']}`",
        f"- **Mode**: `{'DRY-RUN SYNTHETIC' if report['dry_run_synthetic'] else 'REAL SD 1.5 + MULTI-VAE'}`",
        f"- **Selected $\\ell_c^*$**: `{report['ell_c_px']}` px",
        f"- **Optimizer**: `{report.get('optimizer', 'projected_adam')}` | **EoT Identity Pairing**: `{report.get('eot_always_include_identity', True)}`",
        f"- **Fallback Grid Snap (256px)**: `{report['snap_to_256_grid']}`",
        f"- **Num Images**: `{report['num_images']}` | **PGD Steps**: `{report['pgd_steps']}` | **Fixed $\\alpha_{{\\text{{vae}}}}$**: `{report['alpha_vae']}`",
        "",
        "## Top-3 Pareto Configurations",
        "",
        "| Rank | ScaleTarget | $\\lambda$ | $\\beta_{\\text{chroma}}$ | PSNR (dB) | Active Flat $\\Delta E_{2000}$ p95 | Max $|\\delta_{\\text{RGB}}|_{255}$ | $\\rho_{\\text{LF}}$ | Checker Peak | TRP cos (JPEG75) | TRP cos (LPID) | Proxy Score |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    if not report["top_pareto"]:
        lines.append("> **No configuration satisfied all invisibility gates.** Review the full sweep table below.")
        lines.append("")
    for idx, rec in enumerate(report["top_pareto"], start=1):
        lines.append(
            f"| {idx} | {rec['scale_target']} | {rec['lambda']:.2f} | {rec['beta_chroma']:.1f} | "
            f"{rec['mean_psnr_db']:.2f} | {rec['mean_flat_delta_e2000_p95']:.3f} | "
            f"{rec['mean_max_abs_delta_rgb_255']:.2f} | {rec['mean_rho_lf']:.3f} | "
            f"{rec.get('mean_work_checkerboard_peak_to_median', 0.0):.2f} | "
            f"{rec['proxy_trp_cos_jpeg75']:.4f} | {rec['proxy_trp_cos_lpid']:.4f} | "
            f"{rec['proxy_protection_score']:.4f} |"
        )
    if report.get("top_by_scale_target"):
        lines.extend(["", "### Best Configuration per ScaleTarget", ""])
        for st_name, rec in report["top_by_scale_target"].items():
            if rec is None:
                lines.append(f"- **{st_name}**: No configuration passed gates.")
            else:
                lines.append(
                    f"- **{st_name}**: $\\lambda={rec['lambda']:.2f}$, $\\beta_{{\\text{{chroma}}}}={rec['beta_chroma']:.1f}$, "
                    f"PSNR `{rec['mean_psnr_db']:.2f}` dB, Active Flat $\\Delta E_{{2000}}$ p95 `{rec['mean_flat_delta_e2000_p95']:.3f}`, "
                    f"Proxy Score `{rec['proxy_protection_score']:.4f}`"
                )
    lines.extend(
        [
            "",
            "## All Sweep Configurations",
            "",
            "| ScaleTarget | $\\lambda$ | $\\beta_{\\text{chroma}}$ | Gates | Illus Gate | PSNR (dB) | Active Flat $\\Delta E_{2000}$ p95 | $\\rho_{\\text{LF}}$ | TRP cos (JPEG75) | TRP cos (LPID) | VAE L2 (sd15/sdxl/flux) | Proxy Score |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for rec in report["all_configs"]:
        gate_str = "PASS" if rec["passes_invisibility_gates"] else "FAIL"
        ill_str = "PASS" if rec.get("passes_illustration_gates", True) else "FAIL"
        bd = rec.get("proxy_vae_breakdown_jpeg75", {})
        vae_str = f"{bd.get('sd15', 0.0):.3f}/{bd.get('sdxl', 0.0):.3f}/{bd.get('flux', 0.0):.3f}"
        lines.append(
            f"| {rec['scale_target']} | {rec['lambda']:.2f} | {rec['beta_chroma']:.1f} | {gate_str} | {ill_str} | "
            f"{rec['mean_psnr_db']:.2f} | {rec['mean_flat_delta_e2000_p95']:.3f} | "
            f"{rec['mean_rho_lf']:.3f} | {rec['proxy_trp_cos_jpeg75']:.4f} | "
            f"{rec['proxy_trp_cos_lpid']:.4f} | {vae_str} | {rec['proxy_protection_score']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_step2(
    cfg: dict[str, Any],
    output_dir: Path,
    surrogate: SurrogateBackend,
    ell_c_px: int,
    snap_to_256_grid: bool = False,
    image_dir: Path | None = None,
    dry_run_synthetic: bool = False,
    allow_missing_perceptual_gates: bool = False,
) -> dict[str, Any]:
    """Execute Phase 0a Step 2: Projected Adam + EoT sweep across (lambda, beta_chroma, ScaleTarget)."""
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    device = getattr(surrogate, "device", torch.device("cpu"))

    s2_cfg = cfg["step2_screen"]
    tf_cfg = cfg["target_field"]
    gates = s2_cfg["invisibility_gates"]

    pgd_steps = int(s2_cfg.get("pgd_steps", 100))
    if dry_run_synthetic:
        pgd_steps = min(pgd_steps, 8)

    optimizer_name = str(s2_cfg.get("optimizer", "projected_adam"))
    adam_betas = tuple(float(b) for b in s2_cfg.get("adam_betas", [0.9, 0.999]))
    eot_always_identity = bool(s2_cfg.get("eot_always_include_identity", True))

    alpha_vae = float(s2_cfg.get("alpha_vae", 0.35))
    vae_weights = {k: float(v) for k, v in s2_cfg.get("vae_weights", VAE_LOSS_WEIGHTS).items()}
    lambdas = [float(v) for v in s2_cfg["lambda_values"]]
    betas = [float(v) for v in s2_cfg["beta_chroma_values"]]
    scale_targets: list[ScaleTargetType] = list(s2_cfg["scale_targets"])
    jpeg_quality = int(s2_cfg.get("eval_jpeg_quality", 75))
    proxy_t = 2 if dry_run_synthetic else int(s2_cfg.get("proxy_eval_timesteps", 10))
    proxy_n = 2 if dry_run_synthetic else int(s2_cfg.get("proxy_eval_noises", 2))
    active_clip_thr = float(gates.get("active_flat_clip_threshold", 0.05))

    if image_dir is not None:
        images = _load_real_images(image_dir)
    elif dry_run_synthetic:
        images = _generate_synthetic_images()
    else:
        raise ValueError("Real Step 2 requires --image-dir pointing to the 3-domain evaluation dataset.")

    # Build 9-scale target cache for ell_c*
    cache_dir = output_dir / f"target_cache_ell{ell_c_px}"
    scales = tuple(float(s) for s in tf_cfg.get("discrete_scales", DISCRETE_SCALES))
    cache = build_target_cache(
        ell_c_px=ell_c_px,
        surrogate=surrogate,
        output_dir=cache_dir,
        scales=scales,
        seed=seed,
        num_timesteps=4 if dry_run_synthetic else int(tf_cfg.get("rt_num_timesteps", 20)),
        num_noises=2 if dry_run_synthetic else int(tf_cfg.get("rt_num_noises", 6)),
        batch_size=int(tf_cfg.get("rt_eval_batch_size", 8)),
        antithetic=bool(tf_cfg.get("use_antithetic_variates", True)),
        include_extra_vaes=True,
        device="cpu",
    )
    target_tensors_cpu = cache["tensors"]
    target_tensors = {k: v.to(device) for k, v in target_tensors_cpu.items()}

    diff_jpeg = DiffJpegSte(quality=jpeg_quality).to(device)
    lr_start = float(s2_cfg.get("pgd_step_size_start", 0.08))
    lr_end = float(s2_cfg.get("pgd_step_size_end", 0.015))
    all_records: list[dict[str, Any]] = []

    for scale_target in scale_targets:
        probs = s2_cfg["purification_probs_512"] if scale_target == "Standard512" else s2_cfg["purification_probs_1024"]
        for lam in lambdas:
            for beta_chroma in betas:
                per_img_metrics: list[dict[str, Any]] = []
                for img_idx, item in enumerate(images):
                    x_orig = item["rgb"].to(device=device, dtype=torch.float32)
                    _, _, orig_h, orig_w = x_orig.shape
                    h_w, w_w = compute_work_grid(orig_h, orig_w, scale_target=scale_target)
                    masks = build_masks(x_orig, scale_target=scale_target, work_hw=(h_w, w_w))

                    rng = torch.Generator(device="cpu").manual_seed(seed + img_idx * 97 + int(lam * 100))
                    y_base, x_base = _pick_master_origin(h_w, w_w, rng, snap_to_256_grid=snap_to_256_grid)

                    hat_delta = torch.zeros((1, 3, h_w, w_w), device=device, dtype=torch.float32, requires_grad=True)
                    opt = torch.optim.Adam([hat_delta], lr=lr_start, betas=adam_betas)
                    trp_loss_fn = EmaNormalizedTrpLoss(decay=float(s2_cfg.get("ema_trp_decay", 0.99))).to(device)
                    use_ckpt = (device.type == "cuda") and ((h_w * w_w) > (512 * 512))

                    def _compute_surrogate_branch_loss(pur: PurificationOutcome) -> torch.Tensor:
                        r_p = surrogate.forward_trp_residual(pur.sd15_input, use_checkpoint=use_ckpt)
                        l_trp = trp_loss_fn(r_p, pur.r_t_aligned)
                        vae_latents: dict[str, torch.Tensor] = {
                            "sd15": surrogate.encode_sd15_vae(pur.sd15_input, input_range="01", use_checkpoint=use_ckpt)
                        }
                        vae_latents.update(surrogate.encode_extra_vaes(pur.extra_vae_input, use_checkpoint=use_ckpt))
                        l_vae, _ = multi_vae_relative_l2_loss(
                            vae_latents, pur.vae_targets_aligned, weights=vae_weights
                        )
                        return l_trp + alpha_vae * l_vae

                    for step in range(pgd_steps):
                        progress = step / max(1, pgd_steps - 1)
                        step_lr = lr_end + 0.5 * (lr_start - lr_end) * (1.0 + math.cos(math.pi * progress))
                        opt.param_groups[0]["lr"] = step_lr
                        opt.zero_grad(set_to_none=True)

                        inj = inject_and_quantize(
                            x_orig_01=x_orig,
                            hat_delta_work=hat_delta,
                            m_tex_native=masks.m_tex_native,
                            m_clip_native=masks.m_clip_native,
                            lam=lam,
                            blue_noise_dither=True,
                            ste_quantize=True,
                        )
                        x_p_nat = inj["x_out"]
                        if (orig_h, orig_w) == (h_w, w_w):
                            x_p_work = x_p_nat
                        elif orig_h >= h_w and orig_w >= w_w:
                            x_p_work = exact_area_box_resize(x_p_nat, h_w, w_w)
                        else:
                            x_p_work = bilinear_resize(x_p_nat, h_w, w_w)

                        op_name = _sample_purification_op(probs, rng)
                        pur_kw = dict(
                            y_base=y_base,
                            x_base=x_base,
                            target_tensors=target_tensors,
                            rng=rng,
                            diff_jpeg=diff_jpeg,
                            lpid_candidate_scales=tuple(s2_cfg.get("lpid_candidate_scales", [1.15, 1.30, 1.50, 1.75, 2.00])),
                            lpid_low_sizes=tuple(s2_cfg.get("lpid_low_sizes", [112, 160, 224, 256])),
                            bucket1024_candidate_scales=tuple(
                                s2_cfg.get("bucket1024_candidate_scales", [1.50, 1.75, 2.00, 2.25, 2.50])
                            ),
                            bucket1024_max_scaled_pixels=int(s2_cfg.get("bucket1024_max_scaled_pixels", 1_200_000)),
                        )
                        pur_op = apply_purification_with_targets(x_work_01=x_p_work, op_name=op_name, **pur_kw)
                        loss_op = _compute_surrogate_branch_loss(pur_op)

                        if eot_always_identity and op_name != "Identity":
                            pur_id = apply_purification_with_targets(x_work_01=x_p_work, op_name="Identity", **pur_kw)
                            loss_id = _compute_surrogate_branch_loss(pur_id)
                            surr_loss = 0.5 * (loss_id + loss_op)
                        else:
                            surr_loss = loss_op

                        # L_chroma evaluated on fixed 512-ref grid in normalized space (§3.4.1 v0.5.3)
                        loss_chroma = flat_low_freq_chroma_loss(
                            inj["delta_eff_norm"], masks.w_flat_ref, beta_chroma=beta_chroma
                        )
                        total_loss = surr_loss + loss_chroma
                        total_loss.backward()

                        if optimizer_name == "projected_adam":
                            opt.step()
                        else:
                            with torch.no_grad():
                                grad = hat_delta.grad
                                assert grad is not None
                                hat_delta.sub_(step_lr * grad.sign())
                        with torch.no_grad():
                            hat_delta.clamp_(-1.0, 1.0)

                    # Final evaluation using deterministic quantized 8-bit image
                    with torch.no_grad():
                        final_inj = inject_and_quantize(
                            x_orig_01=x_orig,
                            hat_delta_work=hat_delta,
                            m_tex_native=masks.m_tex_native,
                            m_clip_native=masks.m_clip_native,
                            lam=lam,
                            blue_noise_dither=True,
                            ste_quantize=False,
                        )
                        x_p_u8_01 = final_inj["x_u8"].to(torch.float32) / 255.0
                        qual = evaluate_image_quality(
                            x_orig,
                            x_p_u8_01,
                            w_flat=masks.w_flat_native,
                            m_clip_native=masks.m_clip_native,
                            hat_delta_work=hat_delta,
                            active_clip_threshold=active_clip_thr,
                            include_optional_perceptual=(not dry_run_synthetic),
                        )
                        if (not dry_run_synthetic) and (not allow_missing_perceptual_gates):
                            if qual.get("lpips") is None or qual.get("ssimulacra2") is None:
                                raise RuntimeError(
                                    f"Missing perceptual gate metrics in real Step 2 run "
                                    f"(lpips={qual.get('lpips_status')}, ssimulacra2={qual.get('ssimulacra2_status')}). "
                                    "Install optional `[eval]` dependencies or pass --allow-missing-perceptual-gates."
                                )

                        # Resize protected image to working grid for LPID and JPEG proxy evaluation
                        if (orig_h, orig_w) == (h_w, w_w):
                            x_eval_work = x_p_u8_01
                        elif orig_h >= h_w and orig_w >= w_w:
                            x_eval_work = exact_area_box_resize(x_p_u8_01, h_w, w_w)
                        else:
                            x_eval_work = bilinear_resize(x_p_u8_01, h_w, w_w)

                        # Proxy 1: Apply real PIL JPEG q=75 on NATIVE output first, then attacker resize
                        x_jpeg_native = real_jpeg_recompress(x_p_u8_01, quality=jpeg_quality)
                        if (orig_h, orig_w) == (h_w, w_w):
                            x_jpeg_work = x_jpeg_native
                        elif orig_h >= h_w and orig_w >= w_w:
                            x_jpeg_work = exact_area_box_resize(x_jpeg_native, h_w, w_w)
                        else:
                            x_jpeg_work = bilinear_resize(x_jpeg_native, h_w, w_w)

                        spec_1_0 = get_scale_spec(1.00)
                        if scale_target == "Bucket1024":
                            # Fair SD 1.5 0.5x downscale evaluation after native JPEG q=75
                            jpeg_pur = apply_purification_with_targets(
                                x_jpeg_work,
                                op_name="Downscale05x_SD15",
                                y_base=y_base,
                                x_base=x_base,
                                target_tensors=target_tensors,
                                rng=torch.Generator(device="cpu").manual_seed(seed + 100 + img_idx),
                            )
                            sd15_jpeg_in = jpeg_pur.sd15_input
                            extra_jpeg_in = jpeg_pur.extra_vae_input
                            r_t_jpeg = jpeg_pur.r_t_aligned
                            targets_jpeg = jpeg_pur.vae_targets_aligned
                        else:
                            sd15_jpeg_in = x_jpeg_work
                            extra_jpeg_in = x_jpeg_work
                            r_t_jpeg = slice_target_latent(
                                target_tensors["rt.s_1_00"], y_base, x_base, h_w // 8, w_w // 8, spec_1_0
                            )
                            targets_jpeg = {
                                k.split(".")[1]: slice_target_latent(v, y_base, x_base, h_w // 8, w_w // 8, spec_1_0)
                                for k, v in target_tensors.items()
                                if k.startswith("zt.") and k.endswith(".s_1_00")
                            }

                        r_p_jpeg = surrogate.compute_expected_rt(
                            sd15_jpeg_in * 2.0 - 1.0,
                            seed=seed + img_idx,
                            num_timesteps=proxy_t,
                            num_noises=proxy_n,
                            antithetic=True,
                        )
                        cos_jpeg = float(
                            F.cosine_similarity(r_p_jpeg.flatten(1), r_t_jpeg.flatten(1), dim=-1).mean().item()
                        )
                        vaes_jpeg = {"sd15": surrogate.encode_sd15_vae(sd15_jpeg_in, input_range="01")}
                        vaes_jpeg.update(surrogate.encode_extra_vaes(extra_jpeg_in))
                        l2_jpeg_t, bd_jpeg = multi_vae_relative_l2_loss(vaes_jpeg, targets_jpeg, weights=vae_weights)

                        # Optional original SDXL VAE transfer check
                        z_orig_sdxl = surrogate.encode_original_sdxl_vae(extra_jpeg_in)
                        if z_orig_sdxl is not None and "sdxl" in targets_jpeg:
                            z_t_sdxl = targets_jpeg["sdxl"].to(device=z_orig_sdxl.device, dtype=z_orig_sdxl.dtype)
                            bd_jpeg["sdxl_original"] = float(
                                ((z_orig_sdxl - z_t_sdxl).pow(2).sum() / (z_t_sdxl.pow(2).sum() + 1e-8)).item()
                            )

                        # Proxy 2: LPID_CropResize (valid s_eff scale & s_low=160) with seeded expected r_t
                        eval_rng = torch.Generator(device="cpu").manual_seed(seed + 777 + img_idx)
                        lpid_pur = apply_purification_with_targets(
                            x_eval_work,
                            op_name="LPID_CropResize",
                            y_base=y_base,
                            x_base=x_base,
                            target_tensors=target_tensors,
                            rng=eval_rng,
                            forced_low_size=160,
                        )
                        r_p_lpid = surrogate.compute_expected_rt(
                            lpid_pur.sd15_input * 2.0 - 1.0,
                            seed=seed + 500 + img_idx,
                            num_timesteps=proxy_t,
                            num_noises=proxy_n,
                            antithetic=True,
                        )
                        cos_lpid = float(
                            F.cosine_similarity(
                                r_p_lpid.flatten(1), lpid_pur.r_t_aligned.flatten(1), dim=-1
                            ).mean().item()
                        )
                        vaes_lpid = {"sd15": surrogate.encode_sd15_vae(lpid_pur.sd15_input, input_range="01")}
                        vaes_lpid.update(surrogate.encode_extra_vaes(lpid_pur.extra_vae_input))
                        l2_lpid_t, bd_lpid = multi_vae_relative_l2_loss(
                            vaes_lpid, lpid_pur.vae_targets_aligned, weights=vae_weights
                        )

                        qual["domain"] = item.get("domain", "unknown")
                        qual["proxy_trp_cos_jpeg75"] = cos_jpeg
                        qual["proxy_multi_vae_l2_jpeg75"] = float(l2_jpeg_t.item())
                        qual["proxy_vae_breakdown_jpeg75"] = bd_jpeg
                        qual["proxy_trp_cos_lpid"] = cos_lpid
                        qual["proxy_multi_vae_l2_lpid"] = float(l2_lpid_t.item())
                        qual["proxy_vae_breakdown_lpid"] = bd_lpid
                        per_img_metrics.append(qual)

                mean_psnr = float(np.mean([m["psnr_db"] for m in per_img_metrics]))
                mean_ssim = float(np.mean([m["ssim"] for m in per_img_metrics]))
                mean_de_p95 = float(np.mean([m["flat_delta_e2000_p95"] for m in per_img_metrics]))
                mean_max_rgb = float(np.mean([m["max_abs_delta_rgb_255"] for m in per_img_metrics]))
                mean_rho_lf = float(np.mean([m["rho_lf"] for m in per_img_metrics]))
                mean_chk = float(np.mean([m.get("work_checkerboard_f05_peak_to_median", 0.0) for m in per_img_metrics]))
                cos_j = float(np.mean([m["proxy_trp_cos_jpeg75"] for m in per_img_metrics]))
                cos_l = float(np.mean([m["proxy_trp_cos_lpid"] for m in per_img_metrics]))
                l2_j = float(np.mean([m["proxy_multi_vae_l2_jpeg75"] for m in per_img_metrics]))
                l2_l = float(np.mean([m["proxy_multi_vae_l2_lpid"] for m in per_img_metrics]))

                def _avg_breakdown(key: str) -> dict[str, float]:
                    keys = sorted({k for m in per_img_metrics for k in m.get(key, {}).keys()})
                    return {k: float(np.mean([m[key][k] for m in per_img_metrics if k in m[key]])) for k in keys}

                lpips_vals = [m["lpips"] for m in per_img_metrics if m.get("lpips") is not None]
                ssim2_vals = [m["ssimulacra2"] for m in per_img_metrics if m.get("ssimulacra2") is not None]
                mean_lpips = float(np.mean(lpips_vals)) if lpips_vals else None
                mean_ssim2 = float(np.mean(ssim2_vals)) if ssim2_vals else None

                passes_inv = (mean_psnr >= float(gates["min_psnr_db"])) and (
                    mean_de_p95 <= float(gates["max_flat_delta_e2000_p95"])
                )
                if mean_lpips is not None:
                    passes_inv = passes_inv and (mean_lpips <= float(gates["max_lpips"]))
                if mean_ssim2 is not None:
                    passes_inv = passes_inv and (mean_ssim2 >= float(gates["min_ssimulacra2"]))

                # Illustration domain standalone gate check (prevent dilution by Photo/Painting)
                ill_items = [m for m in per_img_metrics if m.get("domain") == "illustration"]
                passes_ill = True
                if ill_items:
                    ill_psnr = float(np.mean([m["psnr_db"] for m in ill_items]))
                    ill_de = float(np.mean([m["flat_delta_e2000_p95"] for m in ill_items]))
                    passes_ill = (ill_psnr >= float(gates["min_psnr_db"])) and (
                        ill_de <= float(gates["max_flat_delta_e2000_p95"])
                    )
                    passes_inv = passes_inv and passes_ill

                proxy_score = 0.5 * (cos_j + cos_l) - alpha_vae * 0.5 * (l2_j + l2_l)
                all_records.append(
                    {
                        "scale_target": scale_target,
                        "lambda": lam,
                        "beta_chroma": beta_chroma,
                        "mean_psnr_db": mean_psnr,
                        "mean_ssim": mean_ssim,
                        "mean_flat_delta_e2000_p95": mean_de_p95,
                        "mean_max_abs_delta_rgb_255": mean_max_rgb,
                        "mean_rho_lf": mean_rho_lf,
                        "mean_work_checkerboard_peak_to_median": mean_chk,
                        "mean_lpips": mean_lpips,
                        "mean_ssimulacra2": mean_ssim2,
                        "proxy_trp_cos_jpeg75": cos_j,
                        "proxy_trp_cos_lpid": cos_l,
                        "proxy_multi_vae_l2_jpeg75": l2_j,
                        "proxy_multi_vae_l2_lpid": l2_l,
                        "proxy_vae_breakdown_jpeg75": _avg_breakdown("proxy_vae_breakdown_jpeg75"),
                        "proxy_vae_breakdown_lpid": _avg_breakdown("proxy_vae_breakdown_lpid"),
                        "proxy_protection_score": float(proxy_score),
                        "passes_illustration_gates": bool(passes_ill),
                        "passes_invisibility_gates": bool(passes_inv),
                    }
                )

    top_k = int(s2_cfg.get("top_k_pareto", 3))
    top_pareto = select_pareto_top_k(all_records, top_k=top_k)
    top_by_scale_target = {
        st: (select_pareto_top_k([r for r in all_records if r["scale_target"] == st], top_k=1) or [None])[0]
        for st in scale_targets
    }

    report = {
        "design_version": DESIGN_VERSION,
        "dry_run_synthetic": bool(dry_run_synthetic),
        "ell_c_px": int(ell_c_px),
        "optimizer": optimizer_name,
        "eot_always_include_identity": eot_always_identity,
        "snap_to_256_grid": bool(snap_to_256_grid),
        "num_images": len(images),
        "pgd_steps": pgd_steps,
        "alpha_vae": alpha_vae,
        "elapsed_sec": round(time.time() - t0, 2),
        "vae_load_status": getattr(surrogate, "vae_load_status", {}),
        "all_configs": all_records,
        "top_pareto": top_pareto,
        "top_by_scale_target": top_by_scale_target,
    }

    (output_dir / "step2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = _make_step2_markdown(report)
    (output_dir / "step2_report.md").write_text(md, encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_cfg = Path(__file__).resolve().parents[1] / "configs" / "phase0a_screen.yaml"
    parser = argparse.ArgumentParser(description="holoveil Phase 0a Target Verification & Pareto Screening")
    parser.add_argument("--config", type=Path, default=default_cfg, help="Path to phase0a_screen.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase0a"), help="Output directory")
    parser.add_argument("--step1-only", action="store_true", help="Run only Step 1 (stitch & autocorr gates)")
    parser.add_argument("--step2", action="store_true", help="Run Step 2 Pareto screening (with Step 1 unless --step1-report provided)")
    parser.add_argument("--step1-report", type=Path, default=None, help="Optional existing step1_report.json for Step 2")
    parser.add_argument("--ell-c", type=int, default=None, help="Override correlation length ell_c* (px) for Step 2")
    parser.add_argument("--image-dir", type=Path, default=None, help="Directory of 3-domain test images for Step 2")
    parser.add_argument(
        "--dry-run-synthetic",
        action="store_true",
        help="Run CPU synthetic verification without downloading/executing SD 1.5 weights",
    )
    parser.add_argument(
        "--allow-missing-perceptual-gates",
        action="store_true",
        help="Allow real Step 2 run to continue if optional LPIPS / SSIMULACRA2 packages are missing",
    )
    parser.add_argument("--device", type=str, default=None, help="Override torch device (e.g. cuda, cpu)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(int(cfg["seed"]))

    device_str = args.device or ("cpu" if args.dry_run_synthetic else ("cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_str)
    need_step2 = bool(args.step2 and not args.step1_only)

    if args.dry_run_synthetic:
        surrogate: SurrogateBackend = SyntheticDryRunSurrogate(include_extra_vaes=need_step2, device=device)
    else:
        m_cfg = cfg["model"]
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        req_dtype = dtype_map.get(str(m_cfg.get("torch_dtype", "bf16")).lower(), torch.bfloat16)
        if device.type == "cpu":
            req_dtype = torch.float32
        elif req_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            req_dtype = torch.float16
        surrogate = SD15MultiVaeSurrogate(
            sd15_model_id=m_cfg["sd15_model_id"],
            sd15_revision=m_cfg.get("sd15_revision"),
            sdxl_vae_id=m_cfg.get("sdxl_vae_id") if need_step2 else None,
            sdxl_original_vae_id=(
                m_cfg.get("sdxl_original_vae_id")
                if (need_step2 and m_cfg.get("eval_original_sdxl_vae", True))
                else None
            ),
            sdxl_original_vae_subfolder=str(m_cfg.get("sdxl_original_vae_subfolder", "vae")),
            flux_vae_id=(m_cfg.get("flux_vae_id") if (need_step2 and m_cfg.get("enable_flux_vae", True)) else None),
            flux_vae_subfolder=str(m_cfg.get("flux_vae_subfolder", "vae")),
            device=device,
            torch_dtype=req_dtype,
            load_extra_vaes=need_step2,
        )

    step1_report: dict[str, Any] | None = None
    if args.step1_report is not None:
        step1_report = json.loads(args.step1_report.read_text(encoding="utf-8"))
    elif args.step1_only or not args.step2 or args.ell_c is None:
        step1_report = run_step1(
            cfg=cfg,
            output_dir=args.output_dir,
            surrogate=surrogate,
            dry_run_synthetic=args.dry_run_synthetic,
        )
        print((args.output_dir / "step1_report.md").read_text(encoding="utf-8"))

    if args.step1_only:
        return 0

    if not args.step2:
        return 0

    ell_c_star = args.ell_c
    snap_256 = False
    if ell_c_star is None:
        assert step1_report is not None
        ell_c_star = step1_report.get("selected_ell_c_px")
        snap_256 = bool(step1_report.get("fallback_grid_snap_recommended", False))

    if ell_c_star is None:
        raise RuntimeError(
            "Step 1 did not find any ell_c candidate passing rho_rt(32px) >= 0.50; aborting Step 2 per DESIGN.md §7."
        )

    run_step2(
        cfg=cfg,
        output_dir=args.output_dir,
        surrogate=surrogate,
        ell_c_px=int(ell_c_star),
        snap_to_256_grid=snap_256,
        image_dir=args.image_dir,
        dry_run_synthetic=args.dry_run_synthetic,
        allow_missing_perceptual_gates=args.allow_missing_perceptual_gates,
    )
    print((args.output_dir / "step2_report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

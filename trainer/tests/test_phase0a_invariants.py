"""CPU unit tests & invariant checks for holoveil Phase 0a (DESIGN.md v0.5.3-design)."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from holoveil_train import DESIGN_VERSION, seed_everything
from holoveil_train.color_mask import (
    BASE_ETA_Y_255,
    DEFAULT_BLUE_NOISE_PATH,
    FIXED_CHROMA_RATIO,
    bilinear_resize,
    build_masks,
    catmull_rom_resize,
    compute_anti_clamp_mask,
    compute_checkerboard_metrics,
    compute_flat_ciede2000_p95,
    compute_psnr,
    compute_spectral_metrics,
    compute_work_grid,
    delta_ycbcr_to_rgb,
    exact_area_box_resize,
    flat_low_freq_chroma_loss,
    generate_blue_noise_64x64_ranks,
    get_blue_noise_64x64,
    inject_and_quantize,
    min_pool_exact,
    rgb_to_ycbcr_net,
    round_half_up,
    round_mult16,
)
from holoveil_train.poc_screen import (
    DiffJpegSte,
    apply_purification_with_targets,
    main as poc_main,
)
from holoveil_train.rewards import (
    ALL_SCALE_SPECS,
    DISCRETE_SCALES,
    MASTER_SIZE,
    RAW_SIZE,
    EmaNormalizedTrpLoss,
    SyntheticDryRunSurrogate,
    build_rt_master_for_ell_c,
    build_target_cache,
    compute_scaled_latent_slice_bounds,
    compute_stitch_coverage_stats,
    generate_stationary_field_raw,
    get_bucket1024_valid_scales,
    get_lpid_valid_scales,
    get_scale_spec,
    measure_rt_autocorrelation,
    prepare_eval_canvas_for_scale,
    select_shortest_valid_ell_c,
    stitch_latent_windows,
    verify_rt_stitch_fidelity,
)


# ---------------------------------------------------------------------------
# 1. All 9 Scales: Phase-Preserving Resize & Full Core Window Coverage
# ---------------------------------------------------------------------------

EXPECTED_SCALE_TABLE = {
    0.50: (1024, 0.500000, 1280, 16),
    1.00: (2048, 1.000000, 2304, 64),
    1.15: (2368, 1.156250, 2624, 100),
    1.30: (2688, 1.312500, 2944, 121),
    1.50: (3072, 1.500000, 3328, 144),
    1.75: (3584, 1.750000, 3840, 196),
    2.00: (4096, 2.000000, 4352, 256),
    2.25: (4608, 2.250000, 4864, 324),
    2.50: (5120, 2.500000, 5376, 400),
}


def test_all_9_scales_phase_preserving_and_window_coverage() -> None:
    total_windows = 0
    for s_nom in DISCRETE_SCALES:
        exp_ls, exp_seff, exp_leval, exp_wins = EXPECTED_SCALE_TABLE[s_nom]
        spec = get_scale_spec(s_nom)

        assert spec.L_s == exp_ls
        assert math.isclose(spec.s_eff, exp_seff, rel_tol=0.0, abs_tol=1e-12)
        assert spec.L_eval_s == exp_leval
        assert spec.window_count_total == exp_wins
        total_windows += exp_wins

        # Exact phase-preserving rational identity: L_total_s / 2560 == L_s / 2048 == s_eff
        assert spec.L_total_s * 4 == 5 * spec.L_s
        assert spec.pad_s * 8 == spec.L_s and spec.pad_s >= 128
        assert Fraction(spec.L_total_s, RAW_SIZE) == Fraction(spec.L_s, MASTER_SIZE)
        assert math.isclose(spec.L_total_s / float(RAW_SIZE), spec.s_eff, abs_tol=1e-15)

        cov = compute_stitch_coverage_stats(spec)
        assert cov["zero_missing_core_pixels"] is True, f"Uncovered core pixels at scale {s_nom}: {cov}"
        assert cov["core_min_active_windows"] >= 1
        assert cov["core_min_weight_sum"] >= 0.999
        assert cov["min_adjacent_effective_overlap_px"] >= 128

    assert total_windows == 1621


def test_constant_field_stitch_all_9_scales_exact() -> None:
    # Verify actual resize -> eval slice -> window stitching on a constant field produces zero boundary distortion
    x_raw = torch.full((1, 3, RAW_SIZE, RAW_SIZE), 0.37, dtype=torch.float32)

    def _const_eval(patches: torch.Tensor) -> torch.Tensor:
        return exact_area_box_resize(patches, patches.shape[-2] // 8, patches.shape[-1] // 8)[:, :1].repeat(1, 4, 1, 1)

    for s_nom, spec in ALL_SCALE_SPECS.items():
        x_eval = prepare_eval_canvas_for_scale(x_raw, spec)
        assert x_eval.shape == (1, 3, spec.L_eval_s, spec.L_eval_s)
        stitched = stitch_latent_windows(x_eval, spec, _const_eval, batch_size=32)
        assert stitched.shape == (1, 4, spec.L_s_lat, spec.L_s_lat)
        assert torch.allclose(stitched, torch.full_like(stitched, 0.37), atol=1e-5), f"Mismatch at s={s_nom}"


# ---------------------------------------------------------------------------
# 2. White-Background Black Line-Art: m_clip_native (3x3 min) & OR Guard
# ---------------------------------------------------------------------------


def test_white_bg_black_lineart_anti_clamp_and_or_guard() -> None:
    h, w = 128, 128
    x_u8 = torch.full((1, 3, h, w), 255, dtype=torch.uint8)
    # Draw crisp black lines and anti-aliased gray edges inside [20:108, 20:108]
    x_u8[:, :, 40:44, 24:104] = 0
    x_u8[:, :, 39, 24:104] = 220  # AA pixel adjacent to pure white (255)
    x_u8[:, :, 44, 24:104] = 180  # AA pixel adjacent to pure black (0)
    # Interior gray patch far from pure white/black (>= 24px away from clip boundary)
    x_u8[:, :, 70:100, 40:80] = 128

    x_01 = x_u8.to(torch.float32) / 255.0
    m_clip = compute_anti_clamp_mask(x_01)

    # Pure white and pure black must be strictly 0
    assert torch.all(m_clip[:, :, :35, :] == 0.0)
    assert torch.all(m_clip[:, :, 40:44, 24:104] == 0.0)
    # 1px AA neighbors adjacent to pure white or pure black must be zeroed by 3x3 min filter
    assert torch.all(m_clip[:, :, 39, 24:104] == 0.0)
    assert torch.all(m_clip[:, :, 44, 24:104] == 0.0)
    # Interior mid-gray pixels (>=2px inside [70:100, 40:80]) retain mask = 1.0
    assert torch.allclose(m_clip[:, :, 72:98, 42:78], torch.ones((1, 1, 26, 36)), atol=1e-6)

    # Test injection with strong negative AND positive perturbations + Blue-Noise dither
    alpha = torch.ones((1, 1, h, w), dtype=torch.float32)
    alpha[:, :, 80:90, 50:60] = 0.0  # Transparent cutout inside mid-gray region
    m_tex = torch.ones((1, 1, h, w), dtype=torch.float32)

    for sign in (-1.0, 1.0):
        hat_delta = torch.full((1, 3, h, w), sign, dtype=torch.float32)
        # Also test tiny delta below 1e-3 / 255 on a subset of mid-gray pixels
        hat_delta[:, :, 72:76, 42:46] = 1e-5
        res = inject_and_quantize(
            x_orig_01=x_01,
            hat_delta_work=hat_delta,
            m_tex_native=m_tex,
            m_clip_native=m_clip,
            lam=1.0,
            alpha_mask=alpha,
            blue_noise_dither=True,
            ste_quantize=False,
            x_orig_u8=x_u8,
        )
        out_u8 = res["x_u8"]

        # Pure white background must stay bit-exact 255 (no gray dots or dither)
        assert torch.equal(out_u8[:, :, :35, :], x_u8[:, :, :35, :])
        # AA edges adjacent to white/black must have zero gray halo (bit-exact preserved)
        assert torch.equal(out_u8[:, :, 39, 24:104], x_u8[:, :, 39, 24:104])
        assert torch.equal(out_u8[:, :, 44, 24:104], x_u8[:, :, 44, 24:104])
        # Pure black line must stay bit-exact 0
        assert torch.equal(out_u8[:, :, 40:44, 24:104], x_u8[:, :, 40:44, 24:104])
        # Alpha == 0 must stay bit-exact
        assert torch.equal(out_u8[:, :, 80:90, 50:60], x_u8[:, :, 80:90, 50:60])
        # |255 * delta_rgb| < 1e-3 must stay bit-exact (no blue-noise dither added)
        assert torch.equal(out_u8[:, :, 72:76, 42:46], x_u8[:, :, 72:76, 42:46])


# ---------------------------------------------------------------------------
# 3. Non-Integer Ratio 773x1159 -> 352x512: Area Box, Catmull-Rom, min_pool_exact
# ---------------------------------------------------------------------------


def _naive_area_box_1d(arr: np.ndarray, dst_len: int) -> np.ndarray:
    n_in = len(arr)
    r = n_in / float(dst_len)
    out = np.zeros(dst_len, dtype=np.float64)
    for o in range(dst_len):
        s0, s1 = o * r, min(float(n_in), (o + 1) * r)
        acc = 0.0
        wsum = 0.0
        for k in range(n_in):
            w = max(0.0, min(s1, k + 1.0) - max(s0, float(k)))
            acc += w * arr[k]
            wsum += w
        out[o] = acc / wsum
    return out


def _naive_catmull_rom_1d(arr: np.ndarray, dst_len: int) -> np.ndarray:
    n_in = len(arr)
    a = -0.5
    out = np.zeros(dst_len, dtype=np.float64)
    for o in range(dst_len):
        u = (o + 0.5) * (n_in / float(dst_len)) - 0.5
        base = math.floor(u)
        acc = 0.0
        wsum = 0.0
        for d in (-1, 0, 1, 2):
            k = base + d
            dist = abs(u - k)
            if dist <= 1.0:
                w = (a + 2.0) * dist**3 - (a + 3.0) * dist**2 + 1.0
            elif dist < 2.0:
                w = a * dist**3 - 5.0 * a * dist**2 + 8.0 * a * dist - 4.0 * a
            else:
                w = 0.0
            k_clamped = max(0, min(n_in - 1, k))
            acc += w * arr[k_clamped]
            wsum += w
        out[o] = acc / wsum
    return out


def _naive_min_pool_exact_2d(arr: np.ndarray, dst_h: int, dst_w: int) -> np.ndarray:
    h, w = arr.shape
    out = np.empty((dst_h, dst_w), dtype=arr.dtype)
    for y_w in range(dst_h):
        ys = math.floor(y_w * h / dst_h)
        ye = min(h, max(ys + 1, math.ceil((y_w + 1) * h / dst_h)))
        for x_w in range(dst_w):
            xs = math.floor(x_w * w / dst_w)
            xe = min(w, max(xs + 1, math.ceil((x_w + 1) * w / dst_w)))
            out[y_w, x_w] = arr[ys:ye, xs:xe].min()
    return out


def test_non_integer_resampling_and_min_pool_exact() -> None:
    seed_everything(42)
    # Exact 1D mathematical parity checks
    vec = np.linspace(0.1, 0.9, 17) + 0.2 * np.sin(np.arange(17))
    t_vec = torch.from_numpy(vec).view(1, 1, 1, 17).to(torch.float64)

    box_out = exact_area_box_resize(t_vec, 1, 7).squeeze().numpy()
    box_ref = _naive_area_box_1d(vec, 7)
    assert np.allclose(box_out, box_ref, atol=1e-12)

    cr_out = catmull_rom_resize(t_vec, 1, 41).squeeze().numpy()
    cr_ref = _naive_catmull_rom_1d(vec, 41)
    assert np.allclose(cr_out, cr_ref, atol=1e-12)

    # Confirm Catmull-Rom (a=-0.5) differs from PyTorch standard bicubic (a=-0.75) on high-contrast step
    step = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    step[:, :, :, 4:] = 1.0
    cr_up = catmull_rom_resize(step, 23, 29, clamp_unit=False)
    pt_bicubic = F.interpolate(step, size=(23, 29), mode="bicubic", align_corners=False)
    assert (cr_up - pt_bicubic).abs().max().item() > 0.015
    cr_clamped = catmull_rom_resize(step * 2.0 - 1.0, 23, 29, clamp_unit=True)
    assert cr_clamped.min().item() >= -1.0 and cr_clamped.max().item() <= 1.0

    # Full 773x1159 <-> 352x512 test
    img = torch.rand((1, 3, 773, 1159), dtype=torch.float32)
    down = exact_area_box_resize(img, 352, 512)
    assert down.shape == (1, 3, 352, 512)
    # Global mean conservation under exact area box filter
    assert abs(down.mean().item() - img.mean().item()) < 1e-5

    # Differentiable gradient check for Area Box and Catmull-Rom
    src_work = torch.randn((1, 3, 352, 512), dtype=torch.float32, requires_grad=True)
    up_nat = catmull_rom_resize(src_work, 773, 1159, clamp_unit=True)
    assert up_nat.shape == (1, 3, 773, 1159)
    up_nat.sum().backward()
    assert src_work.grad is not None and torch.isfinite(src_work.grad).all()

    # min_pool_exact check on 773x1159 -> 352x512 with sparse zero pixels
    rng = np.random.default_rng(123)
    sparse_map = rng.uniform(0.4, 1.0, size=(773, 1159)).astype(np.float32)
    zero_pts = [(0, 0), (17, 29), (386, 579), (772, 1158), (200, 900)]
    for y, x in zero_pts:
        sparse_map[y, x] = 0.0
    t_sparse = torch.from_numpy(sparse_map).unsqueeze(0).unsqueeze(0)
    pooled = min_pool_exact(t_sparse, 352, 512).squeeze().numpy()
    expected = _naive_min_pool_exact_2d(sparse_map, 352, 512)
    assert np.array_equal(pooled, expected)


# ---------------------------------------------------------------------------
# 4. BT.601 YCbCr (kappa_c = 0.40) RGB L_inf Theoretical Bounds
# ---------------------------------------------------------------------------


def test_bt601_ycbcr_rgb_linf_bounds() -> None:
    # Check all 8 extreme corners (-1, +1)^3
    corners = torch.tensor(
        [
            [sy, scb, scr]
            for sy in (-1.0, 1.0)
            for scb in (-1.0, 1.0)
            for scr in (-1.0, 1.0)
        ],
        dtype=torch.float64,
    ).view(8, 3, 1, 1)

    for lam, expected_max_255 in ((1.00, 20.5056), (0.85, 17.42976), (0.72, 14.764032)):
        delta_rgb = delta_ycbcr_to_rgb(corners, mask=1.0, lam=lam)
        max_255 = float(delta_rgb.abs().max().item() * 255.0)
        assert math.isclose(max_255, expected_max_255, abs_tol=1e-6)

    # Zero perturbation -> zero RGB change (no round-trip distortion)
    zero_delta = torch.zeros((1, 3, 16, 16), dtype=torch.float64)
    assert torch.all(delta_ycbcr_to_rgb(zero_delta, mask=1.0, lam=1.0) == 0.0)

    # Locking check: deviating from kappa_c=0.40 must raise ValueError
    with pytest.raises(ValueError):
        delta_ycbcr_to_rgb(zero_delta, chroma_ratio=0.50)

    # Sanity check BT.601 forward range
    rgb_corners = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float64).view(3, 3, 1, 1)
    y_net = rgb_to_ycbcr_net(rgb_corners)
    assert y_net.min().item() >= -1.0 - 1e-6 and y_net.max().item() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 5. EMA-Normalized TRP Gradient Geometry (Zero ||r_p||_2 Shrinkage Term)
# ---------------------------------------------------------------------------


def test_ema_trp_gradient_has_zero_norm_shrinkage_term() -> None:
    loss_fn = EmaNormalizedTrpLoss(decay=0.99)
    r_t = torch.randn((2, 4, 16, 16), dtype=torch.float32)

    # Prime EMA norm once
    r_p_init = torch.randn((2, 4, 16, 16), dtype=torch.float32)
    _ = loss_fn(r_p_init, r_t, update_ema=True)

    # Evaluate gradient at r_p and at scaled 3.0 * r_p (without changing EMA state)
    r_p1 = torch.randn((2, 4, 16, 16), dtype=torch.float32, requires_grad=True)
    loss1 = loss_fn(r_p1, r_t, update_ema=False)
    loss1.backward()
    grad1 = r_p1.grad.detach().clone()

    r_p2 = (3.0 * r_p1.detach()).requires_grad_(True)
    loss2 = loss_fn(r_p2, r_t, update_ema=False)
    loss2.backward()
    grad2 = r_p2.grad.detach().clone()

    # Pure linear projection: gradient is strictly independent of r_p's scale/direction!
    assert torch.allclose(grad1, grad2, atol=1e-7)

    # Cosine similarity between grad and -r_t must be +1.0 (zero orthogonal/shrinkage component along r_p)
    cos = F.cosine_similarity(grad1.flatten(1), (-r_t).flatten(1), dim=-1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-6)

    # Even when update_ema=True, bar_N_r is detached so gradient still has zero ||r_p||_2 shrinkage term
    r_p3 = (2.0 * r_p1.detach()).requires_grad_(True)
    loss3 = loss_fn(r_p3, r_t, update_ema=True)
    loss3.backward()
    grad3 = r_p3.grad.detach()
    cos3 = F.cosine_similarity(grad3.flatten(1), (-r_t).flatten(1), dim=-1)
    assert torch.allclose(cos3, torch.ones_like(cos3), atol=1e-6)


# ---------------------------------------------------------------------------
# 6. LPID_CropResize (352x512) & Bucket1024Scale (s=2.25, 2.50) s_eff Guards
# ---------------------------------------------------------------------------


def test_s_eff_coordinate_guards_lpid_and_bucket1024() -> None:
    # Case A: Portrait 352x512 LPID_CropResize must exclude s=1.15, 1.30 and keep {1.50, 1.75, 2.00}
    lpid_352 = get_lpid_valid_scales(352, 512)
    lpid_scales = [spec.s_nominal for spec, _ in lpid_352]
    assert lpid_scales == [1.50, 1.75, 2.00]
    for spec, s_crop in lpid_352:
        assert s_crop <= 352
        assert s_crop % 8 == 0

    # Case B: Bucket1024Scale on 352x512 includes s=2.25 and s=2.50; 512x512 stops at 2.00
    b1024_352 = get_bucket1024_valid_scales(352, 512)
    assert [spec.s_nominal for spec, _, _ in b1024_352] == [1.50, 1.75, 2.00, 2.25, 2.50]
    b1024_512 = get_bucket1024_valid_scales(512, 512)
    assert [spec.s_nominal for spec, _, _ in b1024_512] == [1.50, 1.75, 2.00]

    # Case C: Extreme boundary origin y_base = 2048 - H_b, x_base = 2048 - W_b must never slice out of bounds
    dummy_targets: dict[str, torch.Tensor] = {}
    for s_nom, spec in ALL_SCALE_SPECS.items():
        dummy_targets[f"rt.s_{spec.tag}"] = torch.zeros((1, 4, spec.L_s_lat, spec.L_s_lat), dtype=torch.float32)
        dummy_targets[f"zt.sd15.s_{spec.tag}"] = torch.zeros((1, 4, spec.L_s_lat, spec.L_s_lat), dtype=torch.float32)
        dummy_targets[f"zt.sdxl.s_{spec.tag}"] = torch.zeros((1, 4, spec.L_s_lat, spec.L_s_lat), dtype=torch.float32)
        dummy_targets[f"zt.flux.s_{spec.tag}"] = torch.zeros((1, 16, spec.L_s_lat, spec.L_s_lat), dtype=torch.float32)

    x_352 = torch.rand((1, 3, 352, 512), dtype=torch.float32)
    y_edge = MASTER_SIZE - 352  # 1696
    x_edge = MASTER_SIZE - 512  # 1536
    rng = torch.Generator(device="cpu").manual_seed(99)

    for s_scale in (2.25, 2.50):
        out = apply_purification_with_targets(
            x_work_01=x_352,
            op_name="Bucket1024Scale",
            y_base=y_edge,
            x_base=x_edge,
            target_tensors=dummy_targets,
            rng=rng,
            forced_scale=s_scale,
        )
        exp_h = out.metadata["h_scaled"] // 8
        exp_w = out.metadata["w_scaled"] // 8
        assert out.extra_vae_input.shape == (1, 3, out.metadata["h_scaled"], out.metadata["w_scaled"])
        assert out.r_t_aligned.shape == (1, 4, 352 // 8, 512 // 8)
        assert out.vae_targets_aligned["sdxl"].shape == (1, 4, exp_h, exp_w)
        assert out.vae_targets_aligned["flux"].shape == (1, 16, exp_h, exp_w)

    for s_scale in (1.50, 1.75, 2.00):
        out_lpid = apply_purification_with_targets(
            x_work_01=x_352,
            op_name="LPID_CropResize",
            y_base=y_edge,
            x_base=x_edge,
            target_tensors=dummy_targets,
            rng=rng,
            forced_scale=s_scale,
            forced_low_size=160,
        )
        assert out_lpid.sd15_input.shape == (1, 3, 512, 512)
        assert out_lpid.r_t_aligned.shape == (1, 4, 64, 64)
        assert out_lpid.vae_targets_aligned["flux"].shape == (1, 16, 64, 64)

    # Case D: Downscale05x_SD15 on 832x1216 at boundary
    x_1024 = torch.rand((1, 3, 832, 1216), dtype=torch.float32)
    out_down = apply_purification_with_targets(
        x_work_01=x_1024,
        op_name="Downscale05x_SD15",
        y_base=MASTER_SIZE - 832,
        x_base=MASTER_SIZE - 1216,
        target_tensors=dummy_targets,
        rng=rng,
    )
    assert out_down.sd15_input.shape == (1, 3, 416, 608)
    assert out_down.r_t_aligned.shape == (1, 4, 52, 76)
    assert out_down.vae_targets_aligned["sd15"].shape == (1, 4, 52, 76)
    assert out_down.vae_targets_aligned["sdxl"].shape == (1, 4, 104, 152)


# ---------------------------------------------------------------------------
# 7. Additional Invariants: Grid Planner, Blue Noise, DiffJPEG, Metrics, CLI Dry-Run
# ---------------------------------------------------------------------------


def test_work_grid_blue_noise_and_diff_jpeg() -> None:
    assert compute_work_grid(600, 3000, "Standard512") == (256, 1280)
    assert compute_work_grid(3000, 600, "Standard512") == (1280, 256)
    hw_b1024 = compute_work_grid(2000, 3000, "Bucket1024")
    assert hw_b1024[0] % 16 == 0 and hw_b1024[1] % 16 == 0
    assert abs(hw_b1024[0] * hw_b1024[1] - 1024 * 1024) < 80_000

    bn = get_blue_noise_64x64()
    assert bn.shape == (64, 64)
    assert bn.min() >= 0.0 and bn.max() < 1.0
    ranks = np.round(bn * 4096).astype(np.int32)
    assert np.array_equal(np.sort(ranks.ravel()), np.arange(4096))
    # Verify Blue-Noise spectral suppression at low frequencies vs high frequencies
    fft_bn = np.abs(np.fft.fftshift(np.fft.fft2(bn - bn.mean()))) ** 2
    center_power = fft_bn[28:36, 28:36].mean()
    corner_power = fft_bn[:8, :8].mean()
    assert corner_power > 3.0 * center_power

    # DiffJpegSte gradient flow check on non-16-multiple size
    jpeg = DiffJpegSte(quality=75)
    x = torch.rand((1, 3, 73, 101), dtype=torch.float32, requires_grad=True)
    y = jpeg(x)
    assert y.shape == x.shape
    y.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_metrics_and_autocorr_selection() -> None:
    # Verify stationary field autocorrelation scales monotonically with ell_c
    syn = SyntheticDryRunSurrogate()
    ac_results = {}
    for ell_c in (48, 96, 160):
        x_raw = generate_stationary_field_raw(ell_c_px=ell_c, seed=42)
        spec = get_scale_spec(1.00)
        x_eval = prepare_eval_canvas_for_scale(x_raw, spec)
        rt = stitch_latent_windows(
            x_eval,
            spec,
            lambda p: syn.compute_expected_rt(p, num_timesteps=2, num_noises=1),
            batch_size=16,
        )
        ac_results[ell_c] = {"autocorrelation": measure_rt_autocorrelation(rt)}

    rho32_48 = ac_results[48]["autocorrelation"]["rho_at_32px"]
    rho32_96 = ac_results[96]["autocorrelation"]["rho_at_32px"]
    rho32_160 = ac_results[160]["autocorrelation"]["rho_at_32px"]
    assert rho32_48 < rho32_96 < rho32_160
    selected = select_shortest_valid_ell_c(ac_results, min_rho_32px=0.50)
    assert selected in (48, 96, 160)

    # Verify quality & spectral metrics sanity
    clean = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    low_freq_noise = 0.02 * F.interpolate(
        torch.randn((1, 3, 64, 64)), size=(512, 512), mode="bilinear", align_corners=False
    )
    prot = torch.clamp(clean + low_freq_noise, 0.0, 1.0)
    assert compute_psnr(clean, prot) > 30.0
    spec_m = compute_spectral_metrics(prot - clean)
    assert spec_m["rho_lf"] > 0.85
    de = compute_flat_ciede2000_p95(clean, clean)
    assert de["flat_delta_e2000_p95"] == 0.0

    # Verify L_chroma scales quadratically with beta_chroma
    w_flat = torch.ones((1, 1, 64, 64))
    delta_eff = torch.ones((1, 3, 64, 64)) * 0.5
    l1 = flat_low_freq_chroma_loss(delta_eff, w_flat, beta_chroma=1.0)
    l4 = flat_low_freq_chroma_loss(delta_eff, w_flat, beta_chroma=4.0)
    assert math.isclose(l4.item() / l1.item(), 4.0, rel_tol=1e-5)


def test_poc_screen_cli_dry_run_step1_and_step2(tmp_path: Path) -> None:
    out_dir = tmp_path / "phase0a_dry"
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "phase0a_screen.yaml"

    # 1. Test --step1-only --dry-run-synthetic
    rc1 = poc_main(
        [
            "--config",
            str(cfg_path),
            "--output-dir",
            str(out_dir),
            "--step1-only",
            "--dry-run-synthetic",
        ]
    )
    assert rc1 == 0
    s1_json = json.loads((out_dir / "step1_report.json").read_text(encoding="utf-8"))
    assert s1_json["dry_run_synthetic"] is True
    assert s1_json["selected_ell_c_px"] in (48, 96, 160)
    assert (out_dir / "step1_report.md").exists()

    # 2. Test target cache build + metadata inspection
    ell_c_star = int(s1_json["selected_ell_c_px"])
    cache_dir = out_dir / "cache_test"
    syn = SyntheticDryRunSurrogate(include_extra_vaes=True)
    cache_res = build_target_cache(
        ell_c_px=ell_c_star,
        surrogate=syn,
        output_dir=cache_dir,
        scales=(0.50, 1.00, 1.15),
        num_timesteps=2,
        num_noises=1,
        batch_size=16,
    )
    manifest = json.loads(Path(cache_res["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["scale_eff_map"]["1.15"] == 1.15625
    with safe_open(cache_res["safetensors_path"], framework="pt") as sf:
        meta = sf.metadata()
        assert "rt_norm_mean" in meta and "rt_norm_std" in meta
        rt_master = sf.get_tensor("cond.target_residual_master")
        assert abs(rt_master.std(unbiased=False).item() - 1.0) < 1e-3

    # 3. Test --step2 --dry-run-synthetic across Standard512 and Bucket1024
    fast_cfg_data = json.loads(json.dumps(__import__("yaml").safe_load(cfg_path.read_text(encoding="utf-8"))))
    fast_cfg_data["step2_screen"]["pgd_steps"] = 3
    fast_cfg_data["step2_screen"]["lambda_values"] = [0.72]
    fast_cfg_data["step2_screen"]["beta_chroma_values"] = [1.0]
    fast_cfg_data["step2_screen"]["scale_targets"] = ["Standard512", "Bucket1024"]
    fast_cfg_data["target_field"]["discrete_scales"] = [0.50, 1.00, 1.15, 1.50, 1.75, 2.00]
    fast_cfg_path = tmp_path / "fast_phase0a.yaml"
    fast_cfg_path.write_text(__import__("yaml").safe_dump(fast_cfg_data), encoding="utf-8")

    rc2 = poc_main(
        [
            "--config",
            str(fast_cfg_path),
            "--output-dir",
            str(out_dir),
            "--step2",
            "--step1-report",
            str(out_dir / "step1_report.json"),
            "--dry-run-synthetic",
        ]
    )
    assert rc2 == 0
    s2_json = json.loads((out_dir / "step2_report.json").read_text(encoding="utf-8"))
    assert s2_json["design_version"] == DESIGN_VERSION
    assert s2_json["dry_run_synthetic"] is True
    assert s2_json["optimizer"] == "projected_adam"
    assert s2_json["eot_always_include_identity"] is True
    assert len(s2_json["all_configs"]) == 2
    assert set(s2_json["top_by_scale_target"].keys()) == {"Standard512", "Bucket1024"}
    for rec in s2_json["all_configs"]:
        assert "proxy_vae_breakdown_jpeg75" in rec
        assert "sd15" in rec["proxy_vae_breakdown_jpeg75"]
        assert "mean_work_checkerboard_peak_to_median" in rec
        assert "passes_illustration_gates" in rec
    assert (out_dir / "step2_report.md").exists()


def test_round_half_up_parity_and_work_grid_edge_cases() -> None:
    # Python round(22.5)==22 (banker's), whereas floor(22.5+0.5)==23 matching Rust f64::round()
    assert round_half_up(22.5) == 23
    assert round_mult16(360.0) == 368
    assert compute_work_grid(2000, 3000, "Bucket1024") == (832, 1248)
    assert compute_work_grid(120, 600, "Standard512") == (128, 640)
    assert compute_work_grid(104, 520, "Standard512") == (112, 560)

    # .5 latent boundary check under s_eff = 2.25: 2.25 * 10 / 8 = 2.8125, 2.25 * 2 / 8 = 0.5625,
    # 2.25 * 16 / 8 = 4.5 -> must round half up to 5
    spec_225 = get_scale_spec(2.25)
    start, end = compute_scaled_latent_slice_bounds(16, 64, spec_225)
    assert start == 5 and end == 69


def test_active_flat_ciede2000_excludes_white_background() -> None:
    # 96% pure white background (m_clip==0) + 4% interior skin patch with large color error (> 5 dE)
    h, w = 100, 100
    clean = torch.ones((1, 3, h, w), dtype=torch.float32)
    clean[:, :, 40:60, 40:60] = torch.tensor([0.90, 0.78, 0.70]).view(1, 3, 1, 1)

    prot = clean.clone()
    # Strong green tint only in the interior skin patch (white background untouched)
    prot[:, 0, 40:60, 40:60] = 0.80
    prot[:, 1, 40:60, 40:60] = 0.88
    prot[:, 2, 40:60, 40:60] = 0.60

    m_clip = compute_anti_clamp_mask(clean)
    w_flat = torch.ones((1, 1, h, w), dtype=torch.float32)

    res = compute_flat_ciede2000_p95(clean, prot, w_flat=w_flat, m_clip_native=m_clip)
    # Raw flat p95 is diluted by 96% zero white pixels, whereas active flat p95 catches the defect!
    assert res["raw_flat_delta_e2000_p95"] < 1.0
    assert res["flat_delta_e2000_p95"] > 5.0


def test_antithetic_variates_and_global_noise_sharing() -> None:
    # With nonzero Monte-Carlo noise perturbation, antithetic + global noise sharing cancels 1st-order noise
    syn_noisy = SyntheticDryRunSurrogate(mc_noise_sigma=0.35)
    x_raw, r_t_anti, _ = build_rt_master_for_ell_c(
        ell_c_px=96,
        surrogate=syn_noisy,
        seed=42,
        num_timesteps=4,
        num_noises=4,
        batch_size=16,
        antithetic=True,
        share_global_noise=True,
    )
    syn_clean = SyntheticDryRunSurrogate(mc_noise_sigma=0.0)
    _, r_t_exact, _ = build_rt_master_for_ell_c(
        ell_c_px=96,
        surrogate=syn_clean,
        seed=42,
        num_timesteps=4,
        num_noises=4,
        batch_size=16,
    )
    # Exact cancellation of linear +/-eps terms!
    assert torch.allclose(r_t_anti, r_t_exact, atol=1e-5)

    stitch = verify_rt_stitch_fidelity(
        x_raw=x_raw,
        r_t_master=r_t_anti,
        surrogate=syn_noisy,
        num_samples=10,
        seed=42,
        num_timesteps=4,
        num_noises=4,
        antithetic=True,
        share_global_noise=True,
    )
    assert stitch["full_64x64"]["mean"] >= 0.99


def test_l_chroma_512_ref_and_native_grid_checkerboard() -> None:
    # Same normalized low-frequency pattern at 512x512 and 1024x1024 yields matching L_chroma
    base = torch.ones((1, 3, 512, 512), dtype=torch.float32) * 0.4
    up = bilinear_resize(base, 1024, 1024)
    w_512 = torch.ones((1, 1, 512, 512), dtype=torch.float32)
    w_1024 = torch.ones((1, 1, 1024, 1024), dtype=torch.float32)
    l_512 = flat_low_freq_chroma_loss(base, w_512, beta_chroma=2.0)
    l_1024 = flat_low_freq_chroma_loss(up, w_1024, beta_chroma=2.0)
    assert math.isclose(l_512.item(), l_1024.item(), rel_tol=1e-3)

    # Non-square 352x512 2x2 checkerboard pattern must be caught directly on native working grid
    chk = torch.zeros((1, 3, 352, 512), dtype=torch.float32)
    yy = torch.arange(352).view(1, 1, 352, 1)
    xx = torch.arange(512).view(1, 1, 1, 512)
    chk[:, :, :, :] = ((yy + xx) % 2) * 2.0 - 1.0
    m_chk = compute_checkerboard_metrics(chk)
    assert m_chk["work_checkerboard_f05_exact_ratio"] > 0.95
    assert m_chk["work_checkerboard_f05_peak_to_median"] > 1e6


def test_blue_noise_8kb_u16le_asset_parity() -> None:
    assert DEFAULT_BLUE_NOISE_PATH.is_file()
    assert DEFAULT_BLUE_NOISE_PATH.stat().st_size == 8192
    bn_loaded = get_blue_noise_64x64()
    bn_gen = generate_blue_noise_64x64_ranks().astype(np.float32) / 4096.0
    assert np.array_equal(bn_loaded, bn_gen)

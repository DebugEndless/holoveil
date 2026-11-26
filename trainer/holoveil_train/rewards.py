"""Boundary-free 2048x2048 target field, 9-scale cache, Step-1 verification gates, and loss functions.

Implements DESIGN.md §0.2, §5.2, and §7 Phase 0a Step 1:
  - 2560x2560 stationary Gaussian-smoothed field with 256px margin around 2048x2048 logical master
  - Phase-preserving resize across 9 discrete scales with exact effective scale s_eff = L_s / 2048.0
  - 50% overlapping 512x512 window stitching (discard outer 8 latent px = 64px, raised-cosine blend)
  - Step 1 Gate 1: verify_rt_stitch_fidelity (average cosine >= 0.90)
  - Step 1 Gate 2: measure_rt_autocorrelation (rho(32px) >= 0.50, select shortest valid ell_c*)
  - EMA-normalized TRP loss (zero ||r_p||_2 shrinkage term) & Multi-VAE relative L2 loss
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import torch
import torch.nn as nn
from safetensors.torch import save_file as safetensors_save_file

from holoveil_train import DESIGN_VERSION
from holoveil_train.color_mask import (
    bilinear_resize,
    exact_area_box_resize,
    round_half_up,
    round_mult8,
    round_mult16,
)

# ---------------------------------------------------------------------------
# Section 5.2 Locked Constants
# ---------------------------------------------------------------------------

MASTER_SIZE: int = 2048
MARGIN_SIZE: int = 256
RAW_SIZE: int = MASTER_SIZE + 2 * MARGIN_SIZE  # 2560
EVAL_MARGIN: int = 128

WINDOW_SIZE: int = 512
WINDOW_STRIDE: int = 256
LATENT_DOWNSAMPLE: int = 8
DISCARD_LATENT_BORDER: int = 8  # 64px
RAMP_LATENT: int = 16           # 128px overlap ramp

DISCRETE_SCALES: tuple[float, ...] = (0.50, 1.00, 1.15, 1.30, 1.50, 1.75, 2.00, 2.25, 2.50)
ELL_C_CANDIDATES_PX: tuple[int, ...] = (48, 96, 160)
AUTOCORR_DELTAS_PX: tuple[int, ...] = (0, 8, 16, 32, 48, 64)
AUTOCORR_THRESHOLD_32PX: float = 0.50
STITCH_COSINE_THRESHOLD: float = 0.90

VAE_LOSS_WEIGHTS: dict[str, float] = {
    "sd15": 0.2,
    "sdxl": 0.5,
    "flux": 0.5,
}


@dataclass(frozen=True)
class ScaleSpec:
    """Exact rounded resolution and effective scale factor s_eff = L_s / 2048.0."""

    s_nominal: float
    L_s: int
    s_eff: float
    L_total_s: int
    pad_s: int
    L_eval_s: int
    L_s_lat: int
    eval_slice_start: int
    eval_slice_end: int
    core_latent_start: int
    core_latent_end: int
    window_count_1d: int
    window_count_total: int

    @property
    def tag(self) -> str:
        return f"{self.s_nominal:.2f}".replace(".", "_")

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def round_mult64(val: float) -> int:
    return max(64, round_half_up(val / 64.0) * 64)


def build_stitch_origins_1d(l_eval: int, win_size: int = WINDOW_SIZE, stride: int = WINDOW_STRIDE) -> list[int]:
    """Build 1D window start coordinates, appending an aligned end window l_eval - win_size when needed."""
    if l_eval < win_size:
        raise ValueError(f"Evaluation size {l_eval} must be >= window size {win_size}")
    origins = list(range(0, l_eval - win_size + 1, stride))
    last = l_eval - win_size
    if origins[-1] != last:
        origins.append(last)
    return origins


def build_stitch_windows(
    l_eval: int,
    win_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> list[tuple[int, int]]:
    """Return 2D (y, x) window origins covering an l_eval x l_eval canvas."""
    coords = build_stitch_origins_1d(l_eval, win_size=win_size, stride=stride)
    return [(y, x) for y in coords for x in coords]


def get_scale_spec(s_nominal: float) -> ScaleSpec:
    """Compute exact ScaleSpec for a discrete scale level per DESIGN.md §5.2.2."""
    l_s = round_mult64(MASTER_SIZE * float(s_nominal))
    s_eff = l_s / float(MASTER_SIZE)
    l_total_s = (5 * l_s) // 4
    pad_s = l_s // 8
    l_eval_s = l_s + 2 * EVAL_MARGIN
    l_s_lat = l_s // LATENT_DOWNSAMPLE
    eval_slice_start = pad_s - EVAL_MARGIN
    eval_slice_end = pad_s + l_s + EVAL_MARGIN
    core_latent_start = EVAL_MARGIN // LATENT_DOWNSAMPLE  # 16
    core_latent_end = core_latent_start + l_s_lat
    origins_1d = build_stitch_origins_1d(l_eval_s)
    return ScaleSpec(
        s_nominal=float(s_nominal),
        L_s=l_s,
        s_eff=s_eff,
        L_total_s=l_total_s,
        pad_s=pad_s,
        L_eval_s=l_eval_s,
        L_s_lat=l_s_lat,
        eval_slice_start=eval_slice_start,
        eval_slice_end=eval_slice_end,
        core_latent_start=core_latent_start,
        core_latent_end=core_latent_end,
        window_count_1d=len(origins_1d),
        window_count_total=len(origins_1d) ** 2,
    )


ALL_SCALE_SPECS: dict[float, ScaleSpec] = {s: get_scale_spec(s) for s in DISCRETE_SCALES}


# ---------------------------------------------------------------------------
# Stationary Field Generation & Phase-Preserving Multi-Scale Resizing
# ---------------------------------------------------------------------------


def generate_stationary_field_raw(
    ell_c_px: float,
    seed: int = 42,
    raw_size: int = RAW_SIZE,
    amplitude_scale: float = 0.40,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate stationary Gaussian-smoothed field X_t^raw in [-1, 1]^{1x3x2560x2560}.

    Uses 2D periodic FFT convolution with kernel sigma_corr = ell_c_px / 2.0 so that the
    spatial autocorrelation drops to 1/e at lag ell_c_px, with 256px stationary margins
    around the central [256:2304, 256:2304] 2048x2048 master core.
    """
    if ell_c_px <= 0:
        raise ValueError(f"ell_c_px must be positive, got {ell_c_px}")
    sigma_corr = float(ell_c_px) / 2.0

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    white = torch.randn((1, 3, raw_size, raw_size), generator=gen, dtype=torch.float32)

    fy = torch.fft.fftfreq(raw_size, d=1.0, dtype=torch.float64)
    fx = torch.fft.rfftfreq(raw_size, d=1.0, dtype=torch.float64)
    freq_sq = fy[:, None] ** 2 + fx[None, :] ** 2
    transfer = torch.exp(-2.0 * (math.pi**2) * (sigma_corr**2) * freq_sq)

    spec = torch.fft.rfft2(white.to(torch.float64), dim=(-2, -1))
    smoothed = torch.fft.irfft2(spec * transfer, s=(raw_size, raw_size), dim=(-2, -1))

    mean = smoothed.mean(dim=(-2, -1), keepdim=True)
    std = smoothed.std(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    normed = ((smoothed - mean) / std) * float(amplitude_scale)
    x_raw = torch.clamp(normed, -1.0, 1.0).to(dtype=torch.float32, device=device)
    return x_raw


def prepare_eval_canvas_for_scale(x_raw: torch.Tensor, spec: ScaleSpec) -> torch.Tensor:
    """Phase-preserving resize of 2560x2560 X_t^raw -> L_total_s, then slice (L_s + 256)^2 eval canvas."""
    if x_raw.shape[-2:] != (RAW_SIZE, RAW_SIZE):
        raise ValueError(f"Expected X_t^raw spatial size ({RAW_SIZE}, {RAW_SIZE}), got {tuple(x_raw.shape[-2:])}")

    if spec.s_nominal < 1.0:
        x_total = exact_area_box_resize(x_raw, spec.L_total_s, spec.L_total_s)
    elif spec.s_nominal == 1.0:
        x_total = x_raw
    else:
        x_total = bilinear_resize(x_raw, spec.L_total_s, spec.L_total_s)

    x_eval = x_total[
        :,
        :,
        spec.eval_slice_start : spec.eval_slice_end,
        spec.eval_slice_start : spec.eval_slice_end,
    ]
    if x_eval.shape[-2:] != (spec.L_eval_s, spec.L_eval_s):
        raise RuntimeError(
            f"Eval canvas shape mismatch: expected ({spec.L_eval_s}, {spec.L_eval_s}), got {tuple(x_eval.shape[-2:])}"
        )
    return x_eval


# ---------------------------------------------------------------------------
# Overlapping 512x512 Window Stitching (Discard Outer 8 Latent px = 64px)
# ---------------------------------------------------------------------------


def build_latent_cosine_window(
    win_lat: int = WINDOW_SIZE // LATENT_DOWNSAMPLE,
    discard_border: int = DISCARD_LATENT_BORDER,
    ramp_len: int = RAMP_LATENT,
    flat_top: bool = False,
    flat_bottom: bool = False,
    flat_left: bool = False,
    flat_right: bool = False,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build 2D raised-cosine stitching window [1, 1, 64, 64] with outer 8 latent px zeroed.

    Edges touching the outer boundary of the evaluation canvas use flat 1.0 weights
    (per v0.5.3) so every core pixel across all 9 scales achieves weight_sum >= 1.0.
    """
    inner = win_lat - 2 * discard_border
    if inner <= 0:
        raise ValueError("discard_border too large for window size")
    eff_ramp = min(ramp_len, inner // 2)
    flat_len = inner - 2 * eff_ramp

    idx = torch.arange(eff_ramp, dtype=torch.float64, device=device)
    sin_ramp = torch.sin(math.pi * (idx + 0.5) / (2.0 * eff_ramp)) ** 2
    cos_ramp = torch.cos(math.pi * (idx + 0.5) / (2.0 * eff_ramp)) ** 2
    ones_ramp = torch.ones(eff_ramp, dtype=torch.float64, device=device)
    flat = torch.ones(flat_len, dtype=torch.float64, device=device)
    zeros = torch.zeros(discard_border, dtype=torch.float64, device=device)

    def _make_1d(f_start: bool, f_end: bool) -> torch.Tensor:
        l_seg = ones_ramp if f_start else sin_ramp
        r_seg = ones_ramp if f_end else cos_ramp
        return torch.cat([zeros, l_seg, flat, r_seg, zeros], dim=0)

    wy = _make_1d(flat_top, flat_bottom)
    wx = _make_1d(flat_left, flat_right)
    w2d = (wy[:, None] * wx[None, :]).to(dtype=dtype)
    return w2d.unsqueeze(0).unsqueeze(0)


def compute_stitch_coverage_stats(spec: ScaleSpec) -> dict[str, Any]:
    """Verify full core coverage (>= 64px inside window, overlap >= 128px, weight >= 1.0) for a ScaleSpec."""
    eval_lat = spec.L_eval_s // LATENT_DOWNSAMPLE
    win_lat = WINDOW_SIZE // LATENT_DOWNSAMPLE
    last_px = spec.L_eval_s - WINDOW_SIZE
    weight_sum = torch.zeros((eval_lat, eval_lat), dtype=torch.float64)
    active_count = torch.zeros((eval_lat, eval_lat), dtype=torch.int32)

    origins = build_stitch_windows(spec.L_eval_s)
    for y_px, x_px in origins:
        w2d = build_latent_cosine_window(
            win_lat=win_lat,
            flat_top=(y_px == 0),
            flat_bottom=(y_px == last_px),
            flat_left=(x_px == 0),
            flat_right=(x_px == last_px),
            dtype=torch.float64,
        )[0, 0]
        yl = y_px // LATENT_DOWNSAMPLE
        xl = x_px // LATENT_DOWNSAMPLE
        weight_sum[yl : yl + win_lat, xl : xl + win_lat] += w2d
        active_count[yl : yl + win_lat, xl : xl + win_lat] += (w2d > 0).to(torch.int32)

    core_w = weight_sum[
        spec.core_latent_start : spec.core_latent_end,
        spec.core_latent_start : spec.core_latent_end,
    ]
    core_cnt = active_count[
        spec.core_latent_start : spec.core_latent_end,
        spec.core_latent_start : spec.core_latent_end,
    ]

    # Check 1D adjacent effective overlap in pixels
    origins_1d = build_stitch_origins_1d(spec.L_eval_s)
    eff_width_px = (win_lat - 2 * DISCARD_LATENT_BORDER) * LATENT_DOWNSAMPLE  # 384px
    overlaps_px = [eff_width_px - (b - a) for a, b in zip(origins_1d[:-1], origins_1d[1:])]
    min_overlap_px = min(overlaps_px) if overlaps_px else eff_width_px

    return {
        "s_nominal": spec.s_nominal,
        "window_count": len(origins),
        "core_min_weight_sum": float(core_w.min().item()),
        "core_max_weight_sum": float(core_w.max().item()),
        "core_min_active_windows": int(core_cnt.min().item()),
        "min_adjacent_effective_overlap_px": int(min_overlap_px),
        "zero_missing_core_pixels": bool((core_cnt >= 1).all().item() and (core_w >= 0.999).all().item()),
    }


def stitch_latent_windows(
    x_eval: torch.Tensor,
    spec: ScaleSpec,
    window_eval_fn: Callable[..., torch.Tensor],
    batch_size: int = 8,
    pass_origins: bool = False,
) -> torch.Tensor:
    """Evaluate 512x512 overlapping windows on x_eval, cosine-blend, and slice core [16 : 16 + L_s/8]."""
    if x_eval.shape[-2:] != (spec.L_eval_s, spec.L_eval_s):
        raise ValueError(f"Expected eval canvas ({spec.L_eval_s}, {spec.L_eval_s}), got {tuple(x_eval.shape[-2:])}")

    origins = build_stitch_windows(spec.L_eval_s)
    win_lat = WINDOW_SIZE // LATENT_DOWNSAMPLE
    eval_lat = spec.L_eval_s // LATENT_DOWNSAMPLE
    last_px = spec.L_eval_s - WINDOW_SIZE

    acc: torch.Tensor | None = None
    weight_sum = torch.zeros((1, 1, eval_lat, eval_lat), dtype=torch.float32, device=x_eval.device)

    for start in range(0, len(origins), batch_size):
        chunk_origins = origins[start : start + batch_size]
        patches = torch.cat(
            [x_eval[:, :, y : y + WINDOW_SIZE, x : x + WINDOW_SIZE] for y, x in chunk_origins],
            dim=0,
        )
        raw_out = window_eval_fn(patches, chunk_origins) if pass_origins else window_eval_fn(patches)
        out_lat = raw_out.to(device=x_eval.device, dtype=torch.float32)
        if acc is None:
            c_lat = out_lat.shape[1]
            acc = torch.zeros((1, c_lat, eval_lat, eval_lat), dtype=torch.float32, device=x_eval.device)

        for idx, (y_px, x_px) in enumerate(chunk_origins):
            w2d = build_latent_cosine_window(
                win_lat=win_lat,
                flat_top=(y_px == 0),
                flat_bottom=(y_px == last_px),
                flat_left=(x_px == 0),
                flat_right=(x_px == last_px),
                dtype=torch.float32,
                device=x_eval.device,
            )
            yl = y_px // LATENT_DOWNSAMPLE
            xl = x_px // LATENT_DOWNSAMPLE
            acc[:, :, yl : yl + win_lat, xl : xl + win_lat] += out_lat[idx : idx + 1] * w2d
            weight_sum[:, :, yl : yl + win_lat, xl : xl + win_lat] += w2d

    assert acc is not None
    core_acc = acc[:, :, spec.core_latent_start : spec.core_latent_end, spec.core_latent_start : spec.core_latent_end]
    core_w = weight_sum[:, :, spec.core_latent_start : spec.core_latent_end, spec.core_latent_start : spec.core_latent_end]
    if bool((core_w <= 1e-6).any().item()):
        raise RuntimeError("Found uncovered core latent pixels during window stitching")
    return core_acc / core_w


# ---------------------------------------------------------------------------
# Frozen Surrogate Protocol & Real Diffusers Implementation
# ---------------------------------------------------------------------------


class GlobalStep1NoiseBank:
    """Deterministic 320x320 global latent noise bank with antithetic (+eps, -eps) pairs for Step 1."""

    def __init__(
        self,
        seed: int = 42,
        num_timesteps: int = 20,
        num_noises: int = 6,
        antithetic: bool = True,
        latent_size: int = RAW_SIZE // LATENT_DOWNSAMPLE,
    ) -> None:
        self.seed = int(seed)
        self.num_timesteps = int(num_timesteps)
        self.num_noises = int(num_noises)
        self.antithetic = bool(antithetic)
        self.latent_size = int(latent_size)
        self._cache: dict[tuple[int, int], torch.Tensor] = {}

    def _get_base_field(self, t_idx: int, pair_idx: int) -> torch.Tensor:
        key = (t_idx, pair_idx)
        if key not in self._cache:
            field_seed = self.seed + t_idx * 1009 + pair_idx * 17
            gen = torch.Generator(device="cpu").manual_seed(field_seed)
            self._cache[key] = torch.randn((1, 4, self.latent_size, self.latent_size), generator=gen, dtype=torch.float32)
        return self._cache[key]

    def get_batch_slices(
        self,
        t_idx: int,
        noise_idx: int,
        raw_origins_px: list[tuple[int, int]],
        patch_lat: int = WINDOW_SIZE // LATENT_DOWNSAMPLE,
    ) -> torch.Tensor:
        if self.antithetic:
            pair_idx = noise_idx // 2
            sign = -1.0 if (noise_idx % 2 == 1 and (noise_idx < (self.num_noises // 2) * 2)) else 1.0
        else:
            pair_idx = noise_idx
            sign = 1.0
        field = self._get_base_field(t_idx, pair_idx)
        slices = []
        for ry, rx in raw_origins_px:
            yl = ry // LATENT_DOWNSAMPLE
            xl = rx // LATENT_DOWNSAMPLE
            slices.append(field[:, :, yl : yl + patch_lat, xl : xl + patch_lat])
        out = torch.cat(slices, dim=0)
        return out * sign if sign < 0 else out


class SurrogateBackend(Protocol):
    """Protocol for frozen SD 1.5 surrogate & VAE evaluation (real or synthetic dry-run)."""

    sd15_revision: str

    def encode_sd15_vae(
        self, x_01_or_m11: torch.Tensor, input_range: str = "01", use_checkpoint: bool = False
    ) -> torch.Tensor: ...
    def compute_expected_rt(
        self,
        x_m11: torch.Tensor,
        seed: int = 42,
        num_timesteps: int = 20,
        num_noises: int = 6,
        timestep_min: int = 25,
        timestep_max: int = 975,
        antithetic: bool = True,
        noise_bank: GlobalStep1NoiseBank | None = None,
        raw_origins_px: list[tuple[int, int]] | None = None,
    ) -> torch.Tensor: ...
    def forward_trp_residual(
        self,
        x_01: torch.Tensor,
        tau: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        use_checkpoint: bool = False,
    ) -> torch.Tensor: ...
    def encode_extra_vaes(self, x_01: torch.Tensor, use_checkpoint: bool = False) -> dict[str, torch.Tensor]: ...
    def encode_original_sdxl_vae(self, x_01: torch.Tensor) -> torch.Tensor | None: ...


def _resolve_hf_revision(model_id: str, obj: Any | None = None, explicit_rev: str | None = None) -> str:
    if explicit_rev:
        return str(explicit_rev)
    if obj is not None:
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            commit = getattr(cfg, "_commit_hash", None)
            if commit:
                return str(commit)
        commit = getattr(obj, "_commit_hash", None)
        if commit:
            return str(commit)
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id)
        if info.sha:
            return str(info.sha)
    except Exception:
        pass
    return "unresolved"


class SD15MultiVaeSurrogate:
    """Real frozen SD 1.5 (+ optional SDXL/Flux VAEs) surrogate evaluator."""

    def __init__(
        self,
        sd15_model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        sd15_revision: str | None = None,
        sdxl_vae_id: str | None = "madebyollin/sdxl-vae-fp16-fix",
        sdxl_original_vae_id: str | None = None,
        sdxl_original_vae_subfolder: str = "vae",
        flux_vae_id: str | None = "black-forest-labs/FLUX.1-schnell",
        flux_vae_subfolder: str = "vae",
        device: torch.device | str | None = None,
        torch_dtype: torch.dtype | None = None,
        load_extra_vaes: bool = False,
    ) -> None:
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        self.device = device

        if torch_dtype is None:
            if device.type == "cuda":
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                torch_dtype = torch.float32
        self.dtype = torch_dtype

        load_kwargs: dict[str, Any] = {"torch_dtype": self.dtype}
        if sd15_revision:
            load_kwargs["revision"] = sd15_revision

        self.vae = AutoencoderKL.from_pretrained(sd15_model_id, subfolder="vae", **load_kwargs).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(sd15_model_id, subfolder="unet", **load_kwargs).to(self.device)
        self.scheduler = DDPMScheduler.from_pretrained(
            sd15_model_id,
            subfolder="scheduler",
            **({"revision": sd15_revision} if sd15_revision else {}),
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            sd15_model_id,
            subfolder="tokenizer",
            **({"revision": sd15_revision} if sd15_revision else {}),
        )
        text_encoder = CLIPTextModel.from_pretrained(sd15_model_id, subfolder="text_encoder", **load_kwargs).to(
            self.device
        )

        self.sd15_revision = _resolve_hf_revision(sd15_model_id, obj=self.unet, explicit_rev=sd15_revision)

        for mod in (self.vae, self.unet, text_encoder):
            mod.eval()
            mod.requires_grad_(False)

        with torch.no_grad():
            tokens = tokenizer(
                "",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            self.empty_cond = text_encoder(tokens.input_ids)[0].to(dtype=self.dtype)
        del text_encoder, tokenizer

        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(device=self.device, dtype=torch.float32)

        self.extra_vaes: dict[str, Any] = {}
        self.vae_load_status: dict[str, str] = {"sd15": f"loaded:{self.sd15_revision}"}
        if load_extra_vaes:
            if sdxl_vae_id:
                try:
                    sdxl_vae = AutoencoderKL.from_pretrained(sdxl_vae_id, torch_dtype=self.dtype).to(self.device)
                    sdxl_vae.eval()
                    sdxl_vae.requires_grad_(False)
                    self.extra_vaes["sdxl"] = sdxl_vae
                    self.vae_load_status["sdxl"] = f"loaded:{sdxl_vae_id}"
                except Exception as exc:
                    self.vae_load_status["sdxl"] = f"unavailable:{type(exc).__name__}:{exc}"

            if flux_vae_id:
                try:
                    flux_vae = AutoencoderKL.from_pretrained(
                        flux_vae_id,
                        subfolder=flux_vae_subfolder,
                        torch_dtype=self.dtype,
                    ).to(self.device)
                    flux_vae.eval()
                    flux_vae.requires_grad_(False)
                    self.extra_vaes["flux"] = flux_vae
                    self.vae_load_status["flux"] = f"loaded:{flux_vae_id}"
                except Exception as exc:
                    self.vae_load_status["flux"] = f"unavailable:{type(exc).__name__}:{exc}"

        self.original_sdxl_vae: Any | None = None
        if load_extra_vaes and sdxl_original_vae_id:
            try:
                orig_vae = AutoencoderKL.from_pretrained(
                    sdxl_original_vae_id,
                    subfolder=sdxl_original_vae_subfolder,
                    torch_dtype=torch.float32,
                ).to(self.device)
                orig_vae.eval()
                orig_vae.requires_grad_(False)
                self.original_sdxl_vae = orig_vae
                self.vae_load_status["sdxl_original"] = f"loaded:{sdxl_original_vae_id}"
            except Exception as exc:
                self.vae_load_status["sdxl_original"] = f"unavailable:{type(exc).__name__}:{exc}"

    def encode_original_sdxl_vae(self, x_01: torch.Tensor) -> torch.Tensor | None:
        if self.original_sdxl_vae is None:
            return None
        x_m11 = self._to_m11(x_01, "01").to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            posterior = self.original_sdxl_vae.encode(x_m11).latent_dist
            scale = float(getattr(self.original_sdxl_vae.config, "scaling_factor", 0.13025))
            return posterior.mean * scale

    def _to_m11(self, x: torch.Tensor, input_range: str) -> torch.Tensor:
        if input_range == "01":
            return x * 2.0 - 1.0
        if input_range == "m11":
            return x
        raise ValueError(f"Unsupported input_range {input_range!r}")

    def _encode_single_vae(self, vae: Any, x_m11: torch.Tensor) -> torch.Tensor:
        x_in = x_m11.to(device=self.device, dtype=self.dtype)
        posterior = vae.encode(x_in).latent_dist
        shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
        scale = float(getattr(vae.config, "scaling_factor", 0.18215))
        return (posterior.mean - shift) * scale

    def encode_sd15_vae(
        self, x_01_or_m11: torch.Tensor, input_range: str = "01", use_checkpoint: bool = False
    ) -> torch.Tensor:
        x_m11 = self._to_m11(x_01_or_m11, input_range)
        if use_checkpoint and x_m11.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                lambda inp: self._encode_single_vae(self.vae, inp).to(torch.float32),
                x_m11,
                use_reentrant=False,
            )
        return self._encode_single_vae(self.vae, x_m11).to(torch.float32)

    def encode_extra_vaes(self, x_01: torch.Tensor, use_checkpoint: bool = False) -> dict[str, torch.Tensor]:
        x_m11 = self._to_m11(x_01, "01")
        out: dict[str, torch.Tensor] = {}
        for name, vae in self.extra_vaes.items():
            if use_checkpoint and x_m11.requires_grad:
                out[name] = torch.utils.checkpoint.checkpoint(
                    lambda inp, target_vae=vae: self._encode_single_vae(target_vae, inp).to(torch.float32),
                    x_m11,
                    use_reentrant=False,
                )
            else:
                out[name] = self._encode_single_vae(vae, x_m11).to(torch.float32)
        return out

    def compute_expected_rt(
        self,
        x_m11: torch.Tensor,
        seed: int = 42,
        num_timesteps: int = 20,
        num_noises: int = 6,
        timestep_min: int = 25,
        timestep_max: int = 975,
        antithetic: bool = True,
        noise_bank: GlobalStep1NoiseBank | None = None,
        raw_origins_px: list[tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        """Compute E_{tau, eps}[eps_SD15(z_tau, tau, c_empty) - eps] with antithetic variates (+eps, -eps)."""
        b = x_m11.shape[0]
        with torch.no_grad():
            z = self._encode_single_vae(self.vae, x_m11).to(torch.float32)
            timesteps = torch.linspace(timestep_min, timestep_max, num_timesteps, device=self.device)
            timesteps = torch.round(timesteps).to(torch.long)

            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            cond = self.empty_cond.expand(b, -1, -1)
            acc = torch.zeros_like(z, dtype=torch.float32)
            count = 0

            for t_idx, t_val in enumerate(timesteps.tolist()):
                alpha_bar = self.alphas_cumprod[t_val]
                sqrt_a = torch.sqrt(alpha_bar)
                sqrt_one_minus_a = torch.sqrt(1.0 - alpha_bar)
                t_tensor = torch.full((b,), int(t_val), device=self.device, dtype=torch.long)

                if noise_bank is not None and raw_origins_px is not None:
                    for n_idx in range(num_noises):
                        eps = noise_bank.get_batch_slices(
                            t_idx, n_idx, raw_origins_px, patch_lat=z.shape[-1]
                        ).to(self.device)
                        z_t = (sqrt_a * z + sqrt_one_minus_a * eps).to(self.dtype)
                        pred = self.unet(z_t, t_tensor, encoder_hidden_states=cond).sample.to(torch.float32)
                        acc += pred - eps
                        count += 1
                elif antithetic and num_noises >= 2:
                    n_pairs = num_noises // 2
                    for _ in range(n_pairs):
                        eps = torch.randn(z.shape, generator=gen, dtype=torch.float32).to(self.device)
                        for sign in (1.0, -1.0):
                            eps_s = eps * sign
                            z_t = (sqrt_a * z + sqrt_one_minus_a * eps_s).to(self.dtype)
                            pred = self.unet(z_t, t_tensor, encoder_hidden_states=cond).sample.to(torch.float32)
                            acc += pred - eps_s
                            count += 1
                    if num_noises % 2 == 1:
                        eps = torch.randn(z.shape, generator=gen, dtype=torch.float32).to(self.device)
                        z_t = (sqrt_a * z + sqrt_one_minus_a * eps).to(self.dtype)
                        pred = self.unet(z_t, t_tensor, encoder_hidden_states=cond).sample.to(torch.float32)
                        acc += pred - eps
                        count += 1
                else:
                    for _ in range(num_noises):
                        eps = torch.randn(z.shape, generator=gen, dtype=torch.float32).to(self.device)
                        z_t = (sqrt_a * z + sqrt_one_minus_a * eps).to(self.dtype)
                        pred = self.unet(z_t, t_tensor, encoder_hidden_states=cond).sample.to(torch.float32)
                        acc += pred - eps
                        count += 1

            return acc / float(count)

    def forward_trp_residual(
        self,
        x_01: torch.Tensor,
        tau: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        """Differentiable single-step SD 1.5 denoising residual r_p = eps_sur(z_tau, tau, c_empty) - eps."""
        x_m11 = self._to_m11(x_01, "01")
        b = x_m11.shape[0]

        def _encode_fn(inp: torch.Tensor) -> torch.Tensor:
            return self._encode_single_vae(self.vae, inp).to(torch.float32)

        if use_checkpoint and x_m11.requires_grad:
            z = torch.utils.checkpoint.checkpoint(_encode_fn, x_m11, use_reentrant=False)
        else:
            z = _encode_fn(x_m11)

        if tau is None:
            tau = torch.randint(50, 950, (b,), device=self.device, dtype=torch.long)
        else:
            tau = tau.to(device=self.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(z)
        else:
            noise = noise.to(device=self.device, dtype=torch.float32)

        alpha_bar = self.alphas_cumprod[tau].view(b, 1, 1, 1)
        z_tau = torch.sqrt(alpha_bar) * z + torch.sqrt(1.0 - alpha_bar) * noise
        cond = self.empty_cond.expand(b, -1, -1)

        def _unet_fn(zt_in: torch.Tensor) -> torch.Tensor:
            return self.unet(zt_in.to(self.dtype), tau, encoder_hidden_states=cond).sample.to(torch.float32)

        if use_checkpoint and z_tau.requires_grad:
            pred = torch.utils.checkpoint.checkpoint(_unet_fn, z_tau, use_reentrant=False)
        else:
            pred = _unet_fn(z_tau)
        return pred - noise


class SyntheticDryRunSurrogate:
    """Deterministic lightweight surrogate for CPU unit tests and --dry-run-synthetic CLI checks.

    Never used to fabricate real SD 1.5 measurements; outputs are clearly labeled synthetic.
    """

    def __init__(
        self,
        include_extra_vaes: bool = True,
        device: torch.device | str = "cpu",
        mc_noise_sigma: float = 0.0,
    ) -> None:
        self.device = torch.device(device)
        self.sd15_revision = "DRY_RUN_SYNTHETIC_NO_REAL_SD15"
        self.mc_noise_sigma = float(mc_noise_sigma)
        self.vae_load_status: dict[str, str] = {"sd15": "synthetic_dry_run"}
        self.extra_vaes: dict[str, Any] = {}
        if include_extra_vaes:
            self.extra_vaes = {"sdxl": 4, "flux": 16}
            self.vae_load_status["sdxl"] = "synthetic_dry_run"
            self.vae_load_status["flux"] = "synthetic_dry_run"
            self.vae_load_status["sdxl_original"] = "synthetic_dry_run"

    def encode_original_sdxl_vae(self, x_01: torch.Tensor) -> torch.Tensor | None:
        if "sdxl" not in self.extra_vaes:
            return None
        return self._conv_latent((x_01 * 2.0 - 1.0).to(self.device), out_ch=4)

    def _conv_latent(self, x_m11: torch.Tensor, out_ch: int = 4) -> torch.Tensor:
        # Smooth strided projection preserving spatial correlation of x_m11 -> 1/8 resolution
        pooled = exact_area_box_resize(x_m11, x_m11.shape[-2] // 8, x_m11.shape[-1] // 8)
        r, g, b = pooled[:, 0:1], pooled[:, 1:2], pooled[:, 2:3]
        c0 = 0.65 * r - 0.35 * g + 0.15 * b
        c1 = -0.25 * r + 0.70 * g - 0.30 * b
        c2 = 0.20 * r - 0.30 * g + 0.75 * b
        c3 = 0.40 * r + 0.35 * g - 0.50 * b
        base = torch.cat([c0, c1, c2, c3], dim=1)
        if out_ch == 4:
            return base
        reps = (out_ch + 3) // 4
        return base.repeat(1, reps, 1, 1)[:, :out_ch]

    def encode_sd15_vae(
        self, x_01_or_m11: torch.Tensor, input_range: str = "01", use_checkpoint: bool = False
    ) -> torch.Tensor:
        x_m11 = x_01_or_m11 * 2.0 - 1.0 if input_range == "01" else x_01_or_m11
        return self._conv_latent(x_m11.to(self.device), out_ch=4)

    def encode_extra_vaes(self, x_01: torch.Tensor, use_checkpoint: bool = False) -> dict[str, torch.Tensor]:
        x_m11 = (x_01 * 2.0 - 1.0).to(self.device)
        return {name: self._conv_latent(x_m11, out_ch=ch) for name, ch in self.extra_vaes.items()}

    def compute_expected_rt(
        self,
        x_m11: torch.Tensor,
        seed: int = 42,
        num_timesteps: int = 20,
        num_noises: int = 6,
        timestep_min: int = 25,
        timestep_max: int = 975,
        antithetic: bool = True,
        noise_bank: GlobalStep1NoiseBank | None = None,
        raw_origins_px: list[tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        z = self.encode_sd15_vae(x_m11, input_range="m11")
        pad = torch.nn.functional.pad(z, (1, 1, 1, 1), mode="replicate")
        local_avg = torch.nn.functional.avg_pool2d(pad, kernel_size=3, stride=1)
        base = 0.75 * z + 0.25 * local_avg
        if self.mc_noise_sigma <= 0.0:
            return base

        # Optional synthetic Monte-Carlo noise perturbation for testing antithetic cancellation
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        acc = torch.zeros_like(base)
        count = 0
        for t_idx in range(num_timesteps):
            if noise_bank is not None and raw_origins_px is not None:
                for n_idx in range(num_noises):
                    eps = noise_bank.get_batch_slices(t_idx, n_idx, raw_origins_px, patch_lat=z.shape[-1]).to(
                        self.device
                    )
                    acc += base + self.mc_noise_sigma * eps
                    count += 1
            elif antithetic and num_noises >= 2:
                for _ in range(num_noises // 2):
                    eps = torch.randn(z.shape, generator=gen, dtype=torch.float32).to(self.device)
                    acc += base + self.mc_noise_sigma * eps
                    acc += base - self.mc_noise_sigma * eps
                    count += 2
                if num_noises % 2 == 1:
                    eps = torch.randn(z.shape, generator=gen, dtype=torch.float32).to(self.device)
                    acc += base + self.mc_noise_sigma * eps
                    count += 1
            else:
                for _ in range(num_noises):
                    eps = torch.randn(z.shape, generator=gen, dtype=torch.float32).to(self.device)
                    acc += base + self.mc_noise_sigma * eps
                    count += 1
        return acc / float(max(1, count))

    def forward_trp_residual(
        self,
        x_01: torch.Tensor,
        tau: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        z = self.encode_sd15_vae(x_01, input_range="01")
        pad = torch.nn.functional.pad(z, (1, 1, 1, 1), mode="replicate")
        local_avg = torch.nn.functional.avg_pool2d(pad, kernel_size=3, stride=1)
        return 0.75 * z + 0.25 * local_avg


# ---------------------------------------------------------------------------
# Phase 0a Step 1 Gates: Stitch Fidelity & Spatial Autocorrelation
# ---------------------------------------------------------------------------


def verify_rt_stitch_fidelity(
    x_raw: torch.Tensor,
    r_t_master: torch.Tensor,
    surrogate: SurrogateBackend,
    num_samples: int = 20,
    seed: int = 42,
    min_cosine: float = STITCH_COSINE_THRESHOLD,
    num_timesteps: int = 20,
    num_noises: int = 6,
    batch_size: int = 8,
    antithetic: bool = True,
    share_global_noise: bool = True,
) -> dict[str, Any]:
    """Verify stitched r_t^(1.0) against direct 512x512 SD 1.5 evaluations at 20 random 8-aligned origins."""
    if r_t_master.shape != (1, 4, 256, 256):
        raise ValueError(f"Expected r_t_master shape (1, 4, 256, 256), got {tuple(r_t_master.shape)}")

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    max_lat_origin = (MASTER_SIZE - WINDOW_SIZE) // LATENT_DOWNSAMPLE  # 1536 / 8 = 192
    lat_coords = torch.randint(0, max_lat_origin + 1, (num_samples, 2), generator=gen)

    patches = []
    slice_refs = []
    origins_px: list[tuple[int, int]] = []
    raw_origins_px: list[tuple[int, int]] = []

    for i in range(num_samples):
        y_lat = int(lat_coords[i, 0].item())
        x_lat = int(lat_coords[i, 1].item())
        y_px = y_lat * LATENT_DOWNSAMPLE
        x_px = x_lat * LATENT_DOWNSAMPLE
        origins_px.append((y_px, x_px))

        raw_y = MARGIN_SIZE + y_px
        raw_x = MARGIN_SIZE + x_px
        raw_origins_px.append((raw_y, raw_x))
        patches.append(x_raw[:, :, raw_y : raw_y + WINDOW_SIZE, raw_x : raw_x + WINDOW_SIZE])
        slice_refs.append(r_t_master[:, :, y_lat : y_lat + 64, x_lat : x_lat + 64])

    patches_all = torch.cat(patches, dim=0)
    refs_all = torch.cat(slice_refs, dim=0).to(torch.float64)

    noise_bank = (
        GlobalStep1NoiseBank(seed=seed, num_timesteps=num_timesteps, num_noises=num_noises, antithetic=antithetic)
        if share_global_noise
        else None
    )

    direct_outs = []
    for start in range(0, num_samples, batch_size):
        chunk = patches_all[start : start + batch_size]
        chunk_raw_origins = raw_origins_px[start : start + batch_size] if share_global_noise else None
        direct_outs.append(
            surrogate.compute_expected_rt(
                chunk,
                seed=seed + 1000 + start,
                num_timesteps=num_timesteps,
                num_noises=num_noises,
                antithetic=antithetic,
                noise_bank=noise_bank,
                raw_origins_px=chunk_raw_origins,
            ).cpu()
        )
    direct_all = torch.cat(direct_outs, dim=0).to(torch.float64)

    def _cos_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[list[float], dict[str, float]]:
        a_flat = a.reshape(a.shape[0], -1)
        b_flat = b.reshape(b.shape[0], -1)
        cos = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=-1)
        vals = [float(v) for v in cos.tolist()]
        stats = {
            "mean": float(cos.mean().item()),
            "min": float(cos.min().item()),
            "median": float(cos.median().item()),
            "max": float(cos.max().item()),
            "std": float(cos.std(unbiased=False).item()),
        }
        return vals, stats

    full_vals, full_stats = _cos_stats(direct_all, refs_all)
    inner_vals, inner_stats = _cos_stats(direct_all[:, :, 8:56, 8:56], refs_all[:, :, 8:56, 8:56])

    passed = full_stats["mean"] >= min_cosine
    return {
        "gate_threshold": min_cosine,
        "passed": passed,
        "fallback_grid_snap_recommended": not passed,
        "num_samples": num_samples,
        "full_64x64": full_stats,
        "inner_48x48": inner_stats,
        "samples": [
            {
                "y_px": y_px,
                "x_px": x_px,
                "cosine_full_64x64": full_vals[i],
                "cosine_inner_48x48": inner_vals[i],
            }
            for i, (y_px, x_px) in enumerate(origins_px)
        ],
    }


def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    num = (a0 * b0).sum()
    den = torch.sqrt((a0 * a0).sum() * (b0 * b0).sum()).clamp_min(1e-12)
    return float((num / den).item())


def measure_rt_autocorrelation(
    r_t_master: torch.Tensor,
    deltas_px: Iterable[int] = AUTOCORR_DELTAS_PX,
    threshold_32px: float = AUTOCORR_THRESHOLD_32PX,
) -> dict[str, Any]:
    """Measure spatial autocorrelation rho_rt(Delta) over Delta in {0, 8, 16, 32, 48, 64}px."""
    r = r_t_master.detach().to(torch.float64).cpu()
    table: list[dict[str, float | int]] = []
    by_delta: dict[int, float] = {}

    for delta_px in deltas_px:
        if delta_px % LATENT_DOWNSAMPLE != 0:
            raise ValueError(f"Delta {delta_px}px must be a multiple of {LATENT_DOWNSAMPLE}")
        d = delta_px // LATENT_DOWNSAMPLE
        if d == 0:
            rho_x = 1.0
            rho_y = 1.0
            rho = 1.0
        else:
            rho_x = _pearson_corr(r[:, :, :, :-d], r[:, :, :, d:])
            rho_y = _pearson_corr(r[:, :, :-d, :], r[:, :, d:, :])
            rho = 0.5 * (rho_x + rho_y)
        by_delta[int(delta_px)] = rho
        table.append(
            {
                "delta_px": int(delta_px),
                "delta_latent": int(d),
                "rho": rho,
                "rho_x": rho_x,
                "rho_y": rho_y,
            }
        )

    rho_32 = by_delta.get(32)
    passes = (rho_32 is not None) and (rho_32 >= threshold_32px)
    return {
        "threshold_at_32px": threshold_32px,
        "rho_at_32px": rho_32,
        "passes_32px_gate": bool(passes),
        "table": table,
        "by_delta_px": by_delta,
    }


def select_shortest_valid_ell_c(
    candidates_results: dict[int, dict[str, Any]],
    min_rho_32px: float = AUTOCORR_THRESHOLD_32PX,
) -> int | None:
    """Select the shortest correlation length ell_c* whose rho_rt(32px) >= 0.50."""
    valid = [
        ell_c
        for ell_c, res in candidates_results.items()
        if (res["autocorrelation"]["rho_at_32px"] is not None)
        and (res["autocorrelation"]["rho_at_32px"] >= min_rho_32px)
    ]
    return min(valid) if valid else None


def normalize_rt_master(r_t_1_0: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Normalize r_t^(1.0) to zero mean and unit variance for UNet cond_proj injection (§2.1, §5.2.2)."""
    r_f64 = r_t_1_0.to(torch.float64)
    mean = float(r_f64.mean().item())
    std = float(r_f64.std(unbiased=False).item())
    normed = ((r_f64 - mean) / (std + 1e-8)).to(torch.float32)
    return normed, mean, std


def build_rt_master_for_ell_c(
    ell_c_px: int,
    surrogate: SurrogateBackend,
    seed: int = 42,
    num_timesteps: int = 20,
    num_noises: int = 6,
    batch_size: int = 8,
    antithetic: bool = True,
    share_global_noise: bool = True,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate X_t^raw, evaluate stitched boundary-free r_t^(1.0), and return (x_raw, r_t_1_0, r_t_norm)."""
    spec_1_0 = ALL_SCALE_SPECS[1.00]
    x_raw = generate_stationary_field_raw(ell_c_px=ell_c_px, seed=seed, device=device)
    x_eval = prepare_eval_canvas_for_scale(x_raw, spec_1_0)

    noise_bank = (
        GlobalStep1NoiseBank(seed=seed, num_timesteps=num_timesteps, num_noises=num_noises, antithetic=antithetic)
        if share_global_noise
        else None
    )

    def _eval_fn(patches: torch.Tensor, chunk_origins: list[tuple[int, int]]) -> torch.Tensor:
        raw_origins = (
            [(spec_1_0.eval_slice_start + y, spec_1_0.eval_slice_start + x) for y, x in chunk_origins]
            if share_global_noise
            else None
        )
        return surrogate.compute_expected_rt(
            patches,
            seed=seed,
            num_timesteps=num_timesteps,
            num_noises=num_noises,
            antithetic=antithetic,
            noise_bank=noise_bank,
            raw_origins_px=raw_origins,
        )

    r_t_1_0 = stitch_latent_windows(x_eval, spec_1_0, _eval_fn, batch_size=batch_size, pass_origins=True).cpu()
    r_t_norm, _, _ = normalize_rt_master(r_t_1_0)
    return x_raw, r_t_1_0, r_t_norm


# ---------------------------------------------------------------------------
# 9-Scale Target Cache Builder & s_eff Coordinate Guards
# ---------------------------------------------------------------------------


def build_target_cache(
    ell_c_px: int,
    surrogate: SurrogateBackend,
    output_dir: str | Path,
    scales: tuple[float, ...] = DISCRETE_SCALES,
    seed: int = 42,
    num_timesteps: int = 20,
    num_noises: int = 6,
    batch_size: int = 8,
    antithetic: bool = True,
    include_extra_vaes: bool = True,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Build and save the 9-scale discrete target cache + target_cache_manifest.json + safetensors."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    x_raw = generate_stationary_field_raw(ell_c_px=ell_c_px, seed=seed, device=device)
    tensors: dict[str, torch.Tensor] = {}
    scale_records: list[dict[str, Any]] = []

    for s_nom in scales:
        spec = get_scale_spec(s_nom)
        x_eval = prepare_eval_canvas_for_scale(x_raw, spec)

        def _eval_rt(patches: torch.Tensor) -> torch.Tensor:
            return surrogate.compute_expected_rt(
                patches,
                seed=seed,
                num_timesteps=num_timesteps,
                num_noises=num_noises,
                antithetic=antithetic,
            )

        r_t_s = stitch_latent_windows(x_eval, spec, _eval_rt, batch_size=batch_size).cpu().to(torch.float32)
        tensors[f"rt.s_{spec.tag}"] = r_t_s

        def _eval_sd15_vae(patches: torch.Tensor) -> torch.Tensor:
            return surrogate.encode_sd15_vae(patches, input_range="m11")

        z_sd15 = stitch_latent_windows(x_eval, spec, _eval_sd15_vae, batch_size=batch_size).cpu().to(torch.float32)
        tensors[f"zt.sd15.s_{spec.tag}"] = z_sd15

        if include_extra_vaes:
            extra_names = list(getattr(surrogate, "extra_vaes", {}).keys())
            for vae_name in extra_names:
                def _eval_extra(patches: torch.Tensor, target_name: str = vae_name) -> torch.Tensor:
                    outs = surrogate.encode_extra_vaes(patches * 0.5 + 0.5)
                    return outs[target_name]

                z_extra = stitch_latent_windows(x_eval, spec, _eval_extra, batch_size=batch_size).cpu().to(torch.float32)
                tensors[f"zt.{vae_name}.s_{spec.tag}"] = z_extra

        scale_records.append(spec.to_metadata())

    if "rt.s_1_00" not in tensors:
        raise RuntimeError("Discrete scale 1.00 must be present to normalize master target residual")
    r_t_norm, rt_mean, rt_std = normalize_rt_master(tensors["rt.s_1_00"])
    tensors["cond.target_residual_master"] = r_t_norm

    manifest = {
        "design_version": DESIGN_VERSION,
        "ell_c_px": int(ell_c_px),
        "seed": int(seed),
        "sd15_revision": getattr(surrogate, "sd15_revision", "unknown"),
        "vae_load_status": getattr(surrogate, "vae_load_status", {}),
        "rt_norm_mean": rt_mean,
        "rt_norm_std": rt_std,
        "scales": scale_records,
        "scale_eff_map": {f"{rec['s_nominal']:.2f}": rec["s_eff"] for rec in scale_records},
    }

    manifest_path = out_path / "target_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    st_meta = {
        "design_version": DESIGN_VERSION,
        "ell_c_px": str(ell_c_px),
        "sd15_revision": str(manifest["sd15_revision"]),
        "rt_norm_mean": repr(rt_mean),
        "rt_norm_std": repr(rt_std),
        "scales_json": json.dumps(scale_records),
    }
    st_path = out_path / "target_cache.safetensors"
    safetensors_save_file(tensors, str(st_path), metadata=st_meta)

    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "safetensors_path": str(st_path),
        "tensors": tensors,
    }


def get_lpid_valid_scales(
    h_b: int,
    w_b: int,
    candidate_scales: Iterable[float] = (1.15, 1.30, 1.50, 1.75, 2.00),
) -> list[tuple[ScaleSpec, int]]:
    """Return valid (ScaleSpec, s_crop) pairs where s_crop = round_mult8(512 / s_eff) <= min(H_b, W_b)."""
    min_side = min(h_b, w_b)
    valid: list[tuple[ScaleSpec, int]] = []
    for s_nom in candidate_scales:
        spec = get_scale_spec(s_nom)
        s_crop = round_mult8(512.0 / spec.s_eff)
        if s_crop <= min_side:
            valid.append((spec, s_crop))
    return valid


def get_bucket1024_valid_scales(
    h_b: int,
    w_b: int,
    candidate_scales: Iterable[float] = (1.50, 1.75, 2.00, 2.25, 2.50),
    max_scaled_pixels: int = 1_200_000,
) -> list[tuple[ScaleSpec, int, int]]:
    """Return valid (ScaleSpec, H_scaled, W_scaled) where s_eff^2 * H_b * W_b <= max_scaled_pixels."""
    valid: list[tuple[ScaleSpec, int, int]] = []
    for s_nom in candidate_scales:
        spec = get_scale_spec(s_nom)
        if (spec.s_eff * spec.s_eff) * float(h_b) * float(w_b) <= float(max_scaled_pixels):
            h_s = round_mult16(h_b * spec.s_eff)
            w_s = round_mult16(w_b * spec.s_eff)
            valid.append((spec, h_s, w_s))
    return valid


def compute_scaled_latent_slice_bounds(
    coord_px: int,
    patch_lat: int,
    spec: ScaleSpec,
) -> tuple[int, int]:
    """Compute guarded latent slice [start, start + patch_lat) using s_eff and floor(x+0.5) (§5.2.3)."""
    if patch_lat <= 0 or patch_lat > spec.L_s_lat:
        raise ValueError(f"patch_lat {patch_lat} out of valid range [1, {spec.L_s_lat}] for s={spec.s_nominal}")
    raw_idx = round_half_up(spec.s_eff * float(coord_px) / float(LATENT_DOWNSAMPLE))
    max_start = spec.L_s_lat - patch_lat
    start = max(0, min(raw_idx, max_start))
    return start, start + patch_lat


def slice_target_latent(
    target_tensor: torch.Tensor,
    y_px: int,
    x_px: int,
    h_lat: int,
    w_lat: int,
    spec: ScaleSpec,
) -> torch.Tensor:
    """Safely extract [1, C, h_lat, w_lat] slice from a cached target tensor using s_eff + boundary guard."""
    if target_tensor.shape[-2:] != (spec.L_s_lat, spec.L_s_lat):
        raise ValueError(
            f"Target tensor spatial shape {tuple(target_tensor.shape[-2:])} does not match ScaleSpec {spec.L_s_lat}"
        )
    ys, ye = compute_scaled_latent_slice_bounds(y_px, h_lat, spec)
    xs, xe = compute_scaled_latent_slice_bounds(x_px, w_lat, spec)
    return target_tensor[:, :, ys:ye, xs:xe]


# ---------------------------------------------------------------------------
# Section 0.2 & 5.2.4: EMA-Normalized TRP Loss & Multi-VAE Relative L2 Loss
# ---------------------------------------------------------------------------


class EmaNormalizedTrpLoss(nn.Module):
    """EMA-Normalized Target Residual Projection (TRP) loss per DESIGN.md §0.2.1.

    L_tilde_TRP(r_p, r_t) = -<r_p, r_t> / (||r_t||_2 * (bar_N_r + 1e-8))
    where bar_N_r = EMA(||r_p||_2.detach()) ensures O(1) loss scaling while strictly
    preserving linear projection gradient geometry (zero ||r_p||_2 shrinkage term).
    """

    def __init__(self, decay: float = 0.99, eps: float = 1e-8) -> None:
        super().__init__()
        self.decay = float(decay)
        self.eps = float(eps)
        self.register_buffer("ema_norm", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def forward(
        self,
        r_p: torch.Tensor,
        r_t: torch.Tensor,
        update_ema: bool = True,
    ) -> torch.Tensor:
        r_p_f32 = r_p.to(torch.float32)
        r_t_f32 = r_t.to(device=r_p.device, dtype=torch.float32)

        if update_ema:
            with torch.no_grad():
                curr_norm = r_p_f32.detach().flatten(1).norm(dim=1).mean()
                if not bool(self.initialized.item()):
                    self.ema_norm.copy_(curr_norm)
                    self.initialized.fill_(True)
                else:
                    self.ema_norm.mul_(self.decay).add_(curr_norm * (1.0 - self.decay))

        bar_n_r = self.ema_norm.detach()
        if not bool(self.initialized.item()):
            # Fallback if called with update_ema=False prior to initialization
            bar_n_r = r_p_f32.detach().flatten(1).norm(dim=1).mean()

        inner = (r_p_f32 * r_t_f32).flatten(1).sum(dim=1)
        rt_norm = r_t_f32.flatten(1).norm(dim=1).clamp_min(self.eps)
        loss = -(inner / (rt_norm * (bar_n_r + self.eps)))
        return loss.mean()


def multi_vae_relative_l2_loss(
    vae_latents: dict[str, torch.Tensor],
    aligned_targets: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted relative L2 latent distance sum_k w_k * ||E_k(X_p) - z_t,k||^2 / (||z_t,k||^2 + eps)."""
    w_map = weights if weights is not None else VAE_LOSS_WEIGHTS
    total: torch.Tensor | None = None
    breakdown: dict[str, float] = {}

    for name, z_p in vae_latents.items():
        if name not in aligned_targets:
            continue
        z_t = aligned_targets[name].to(device=z_p.device, dtype=z_p.dtype)
        diff_sq = (z_p - z_t).flatten(1).pow(2).sum(dim=1)
        ref_sq = z_t.flatten(1).pow(2).sum(dim=1) + eps
        rel_l2 = (diff_sq / ref_sq).mean()
        w_k = float(w_map.get(name, 0.0))
        breakdown[name] = float(rel_l2.detach().item())
        term = w_k * rel_l2
        total = term if total is None else (total + term)

    if total is None:
        first_tensor = next(iter(vae_latents.values())) if vae_latents else torch.tensor(0.0)
        return torch.zeros((), device=first_tensor.device, dtype=torch.float32), breakdown
    return total, breakdown

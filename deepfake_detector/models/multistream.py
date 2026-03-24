"""
Tri-stream deepfake detector:
 - RGB stream (EfficientNet)
 - Frequency stream (FFT log-magnitude, radial bands)
 - SRM-like stream (learnable noise residuals)

This is designed to be drop-in for the existing training/testing scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfake_detector.models.efficientnet import DeepFakeDetector


def _effnet_input_size(model_name: str) -> int:
    name = model_name.lower()
    # Matches efficientnet-pytorch defaults (common convention).
    if "b0" in name:
        return 224
    if "b1" in name:
        return 240
    if "b2" in name:
        return 260
    if "b3" in name:
        return 300
    if "b4" in name:
        return 380
    if "b5" in name:
        return 456
    if "b6" in name:
        return 528
    if "b7" in name:
        return 600
    # Fallback
    return 224


@dataclass
class TriStreamConfig:
    rgb_model: str = "efficientnet-b4"
    freq_model: str = "efficientnet-b2"
    srm_model: str = "efficientnet-b0"
    srm_filters: int = 30
    freq_band_r1: float = 0.33
    freq_band_r2: float = 0.66


class TriStreamDeepFakeDetector(nn.Module):
    """
    Forward takes RGB images only (as produced by dataset transforms),
    and internally derives frequency and SRM-like inputs.
    """

    def __init__(
        self,
        rgb_model: str = "efficientnet-b4",
        freq_model: str = "efficientnet-b2",
        srm_model: str = "efficientnet-b0",
        srm_filters: int = 30,
        freq_band_r1: float = 0.33,
        freq_band_r2: float = 0.66,
        pretrained: bool = True,
        num_classes: int = 2,
    ):
        super().__init__()

        self.cfg = TriStreamConfig(
            rgb_model=rgb_model,
            freq_model=freq_model,
            srm_model=srm_model,
            srm_filters=srm_filters,
            freq_band_r1=freq_band_r1,
            freq_band_r2=freq_band_r2,
        )

        # Branch classifiers.
        self.rgb_branch = DeepFakeDetector(
            model_name=self.cfg.rgb_model,
            num_classes=num_classes,
            pretrained=pretrained,
        )
        self.freq_branch = DeepFakeDetector(
            model_name=self.cfg.freq_model,
            num_classes=num_classes,
            pretrained=pretrained,
        )
        self.srm_branch = DeepFakeDetector(
            model_name=self.cfg.srm_model,
            num_classes=num_classes,
            pretrained=pretrained,
        )

        # Learnable SRM-like residual extractor (from grayscale).
        self._srm_conv = nn.Conv2d(1, self.cfg.srm_filters, kernel_size=5, padding=2, bias=False)
        self._srm_compress = nn.Conv2d(self.cfg.srm_filters, 3, kernel_size=1, bias=True)

        self._init_srm_kernels()

        # Gated fusion over logits.
        # Each branch outputs [B, 2], so concat => 6.
        self._gate = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3),
        )

        # ImageNet mean/std for denormalization and re-normalization.
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

        # Cached frequency radius grids keyed by size.
        self._radius_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def _init_srm_kernels(self) -> None:
        # Laplacian-like 5x5 kernel replicated across filters with small noise.
        base = torch.tensor(
            [
                [0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -2.0, 0.0, 0.0],
                [-1.0, -2.0, 16.0, -2.0, -1.0],
                [0.0, 0.0, -2.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        base = base / (base.abs().sum() + 1e-6)

        with torch.no_grad():
            weight = self._srm_conv.weight  # [K,1,5,5]
            for k in range(weight.shape[0]):
                noise = 0.02 * torch.randn_like(base)
                weight[k, 0].copy_(base + noise)

    def _denormalize_rgb(self, x_rgb: torch.Tensor) -> torch.Tensor:
        # Dataset uses ImageNet normalization, invert it for frequency/SRM computation.
        return x_rgb * self._std + self._mean

    def _normalize_to_imagenet(self, x_01: torch.Tensor) -> torch.Tensor:
        # x_01 assumed in [0,1] (approx); apply ImageNet normalization.
        return (x_01 - self._mean) / self._std

    def _get_radius_grid(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w)
        cached = self._radius_cache.get(key)
        if cached is not None and cached.device == device:
            return cached

        yy = torch.arange(h, device=device, dtype=torch.float32)
        xx = torch.arange(w, device=device, dtype=torch.float32)
        yy = yy - (h / 2.0)
        xx = xx - (w / 2.0)
        # radius in pixels
        rr = torch.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
        r_max = rr.max().clamp_min(1e-6)
        rr = rr / r_max  # normalized to [0,1]
        self._radius_cache[key] = rr
        return rr

    def _compute_freq_bands(self, x_rgb: torch.Tensor, out_size: int) -> torch.Tensor:
        # x_rgb: normalized tensor [B,3,H,W]
        x_rgb01 = self._denormalize_rgb(x_rgb).clamp(0.0, 1.0)

        # Convert to grayscale luminance.
        r, g, b = x_rgb01[:, 0:1], x_rgb01[:, 1:2], x_rgb01[:, 2:3]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  # [B,1,H,W]

        gray = F.interpolate(gray, size=(out_size, out_size), mode="bilinear", align_corners=False)

        # FFT -> log-magnitude -> radial bands
        fft = torch.fft.fft2(gray.squeeze(1))  # [B,H,W] complex
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        mag = torch.abs(fft)
        mag = torch.log1p(mag)  # [B,H,W]

        b, h, w = mag.shape
        rr = self._get_radius_grid(h, w, mag.device)  # [H,W] normalized radius

        r1 = self.cfg.freq_band_r1
        r2 = self.cfg.freq_band_r2
        mask_low = (rr < r1).to(mag.dtype)
        mask_mid = ((rr >= r1) & (rr < r2)).to(mag.dtype)
        mask_high = (rr >= r2).to(mag.dtype)

        low = mag * mask_low
        mid = mag * mask_mid
        high = mag * mask_high

        # Per-sample min-max scale to [0,1] to make EfficientNet input stable.
        def scale01(x: torch.Tensor) -> torch.Tensor:
            x_min = x.amin(dim=(-2, -1), keepdim=True)
            x_max = x.amax(dim=(-2, -1), keepdim=True)
            return (x - x_min) / (x_max - x_min + 1e-6)

        low = scale01(low)
        mid = scale01(mid)
        high = scale01(high)

        x_freq01 = torch.stack([low, mid, high], dim=1)  # [B,3,H,W]
        x_freq = self._normalize_to_imagenet(x_freq01)
        return x_freq

    def _compute_srm_like(self, x_rgb: torch.Tensor, out_size: int) -> torch.Tensor:
        x_rgb01 = self._denormalize_rgb(x_rgb).clamp(0.0, 1.0)
        r, g, b = x_rgb01[:, 0:1], x_rgb01[:, 1:2], x_rgb01[:, 2:3]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  # [B,1,H,W]
        gray = F.interpolate(gray, size=(out_size, out_size), mode="bilinear", align_corners=False)

        # Learnable noise residuals.
        noise = self._srm_conv(gray)
        noise = torch.abs(noise)
        noise = F.relu(self._srm_compress(noise))  # [B,3,H,W]

        # Normalize to [0,1] then to ImageNet normalization.
        noise_min = noise.amin(dim=(-2, -1), keepdim=True)
        noise_max = noise.amax(dim=(-2, -1), keepdim=True)
        noise01 = (noise - noise_min) / (noise_max - noise_min + 1e-6)
        x_srm = self._normalize_to_imagenet(noise01)
        return x_srm

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        # RGB logits.
        logits_rgb = self.rgb_branch(x_rgb)

        # Frequency logits.
        freq_size = _effnet_input_size(self.cfg.freq_model)
        x_freq = self._compute_freq_bands(x_rgb, out_size=freq_size)
        logits_freq = self.freq_branch(x_freq)

        # SRM-like logits.
        srm_size = _effnet_input_size(self.cfg.srm_model)
        x_srm = self._compute_srm_like(x_rgb, out_size=srm_size)
        logits_srm = self.srm_branch(x_srm)

        # Gated fusion.
        concat = torch.cat([logits_rgb, logits_freq, logits_srm], dim=1)  # [B,6]
        w = F.softmax(self._gate(concat), dim=1)  # [B,3]
        fused = (
            w[:, 0:1] * logits_rgb +
            w[:, 1:2] * logits_freq +
            w[:, 2:3] * logits_srm
        )
        return fused

    def save_checkpoint(
        self,
        checkpoint_path: str,
        epoch: Optional[int] = None,
        optimizer_state: Optional[dict] = None,
        scheduler_state: Optional[dict] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        checkpoint: Dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "tri_stream_config": self.cfg.__dict__,
            "num_classes": 2,
        }
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state
        if scheduler_state is not None:
            checkpoint["scheduler_state_dict"] = scheduler_state
        if metrics is not None:
            checkpoint["metrics"] = metrics
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str, device: str = "cpu") -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
            return checkpoint
        else:
            self.load_state_dict(checkpoint)
            return {}


"""
Tri-stream deepfake detector — SVG feature-vector pipeline.

Architecture (matching feature_vector_pipeline.svg):

  Input (H×W×3, ImageNet-normalised)
    ├─ Stream 1 – RGB:       original → EfficientNet-B4 (encoder only) → f_rgb  [B, D]
    ├─ Stream 2 – Frequency: FFT magnitude + phase + DCT → EfficientNet-B4 → f_freq [B, D]
    └─ Stream 3 – SRM:       30 high-pass filters → 3-ch → EfficientNet-B4 → f_srm  [B, D]
                                       ↓
             Channel Attention Fusion  (w_rgb·f_rgb + w_freq·f_freq + w_srm·f_srm)
                                       ↓
                            fused vector [B, D]
                                       ↓
              LayerNorm → Dropout(0.5) → FC(512) → GELU → FC(num_classes)

Key differences from the old logit-fusion implementation:
- All three branches share the same EfficientNet variant (default: B4).
- Branches act as *feature extractors* — the per-branch FC head is removed.
- Fusion happens at the 1792-d feature level (Channel Attention), not over 2-class logits.
- A single shared classifier head produces the final prediction.
- Frequency stream now encodes FFT magnitude + phase + DCT (3 complementary channels).
- SRM stream: 30 learnable Laplacian-like high-pass kernels → compressed to 3 channels.

Backward compatibility:
- The constructor accepts the old `freq_model`, `srm_model`, `freq_band_r1`, `freq_band_r2`
  arguments but ignores them (they are superseded by the shared `rgb_model` backbone).
- Output is still [B, num_classes] logits — `CrossEntropyLoss` in `train.py` is unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _effnet_input_size(model_name: str) -> int:
    sizes = {"b0": 224, "b1": 240, "b2": 260, "b3": 300,
             "b4": 380, "b5": 456, "b6": 528, "b7": 600}
    name = model_name.lower()
    for k, v in sizes.items():
        if k in name:
            return v
    return 224


def _effnet_feat_dim(model_name: str) -> int:
    """Feature dimension (before FC head) for each EfficientNet variant."""
    dims = {"b0": 1280, "b1": 1280, "b2": 1408, "b3": 1536,
            "b4": 1792, "b5": 2048, "b6": 2304, "b7": 2560}
    name = model_name.lower()
    for k, v in dims.items():
        if k in name:
            return v
    return 1280


# ──────────────────────────────────────────────────────────────────────────────
# Channel Attention Fusion
# ──────────────────────────────────────────────────────────────────────────────

class ChannelAttentionFusion(nn.Module):
    """
    Learns soft attention weights over N feature vectors and returns their
    weighted sum.

    w = softmax( MLP( concat(f_1, …, f_N) ) )   shape [B, N]
    out = sum_i  w_i * f_i                        shape [B, feat_dim]
    """

    def __init__(self, feat_dim: int, num_streams: int = 3, reduction: int = 16):
        super().__init__()
        hidden = max(feat_dim // reduction, 32)
        self.attn = nn.Sequential(
            nn.Linear(feat_dim * num_streams, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_streams),
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        cat = torch.cat(feats, dim=1)                        # [B, D*N]
        w = F.softmax(self.attn(cat), dim=1)                 # [B, N]
        return sum(w[:, i : i + 1] * f for i, f in enumerate(feats))  # [B, D]


# ──────────────────────────────────────────────────────────────────────────────
# Main detector
# ──────────────────────────────────────────────────────────────────────────────

class TriStreamDeepFakeDetector(nn.Module):
    """
    Tri-stream deepfake detector matching the SVG feature-vector pipeline.

    Args:
        rgb_model:   EfficientNet variant used as the shared backbone for all
                     three streams (e.g. 'efficientnet-b4').
        freq_model:  Deprecated — kept for backward compatibility, ignored.
        srm_model:   Deprecated — kept for backward compatibility, ignored.
        srm_filters: Number of learnable high-pass convolutional filters for
                     the SRM stream (default: 30).
        freq_band_r1, freq_band_r2: Deprecated — ignored.
        pretrained:  Load ImageNet pre-trained weights for the backbones.
        num_classes: Output classes (default: 2 for binary classification).
    """

    def __init__(
        self,
        rgb_model: str = "efficientnet-b4",
        freq_model: str = "efficientnet-b4",   # deprecated — ignored
        srm_model: str = "efficientnet-b4",    # deprecated — ignored
        srm_filters: int = 30,
        freq_band_r1: float = 0.33,            # deprecated — ignored
        freq_band_r2: float = 0.66,            # deprecated — ignored
        pretrained: bool = True,
        num_classes: int = 2,
    ):
        super().__init__()

        # All three streams use the same backbone variant.
        backbone = rgb_model
        self._backbone_name = backbone
        self._input_size = _effnet_input_size(backbone)
        self._feat_dim = _effnet_feat_dim(backbone)

        # ── Three feature encoders (EfficientNet without FC head) ────────────
        self.rgb_encoder = self._build_encoder(backbone, pretrained)
        self.freq_encoder = self._build_encoder(backbone, pretrained)
        self.srm_encoder = self._build_encoder(backbone, pretrained)

        # ── SRM block: learnable high-pass filters → 3 channels ─────────────
        # Conv(1 → srm_filters, 5×5)  then  Conv(srm_filters → 3, 1×1)
        self._srm_conv = nn.Conv2d(1, srm_filters, kernel_size=5, padding=2, bias=False)
        self._srm_to3 = nn.Conv2d(srm_filters, 3, kernel_size=1, bias=True)
        self._init_srm_kernels()

        # ── Channel Attention Fusion ─────────────────────────────────────────
        self.fusion = ChannelAttentionFusion(
            feat_dim=self._feat_dim, num_streams=3, reduction=16
        )

        # ── Classifier head ──────────────────────────────────────────────────
        # LayerNorm → Dropout → FC(512) → GELU → FC(num_classes)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self._feat_dim),
            nn.Dropout(0.5),
            nn.Linear(self._feat_dim, 512),
            nn.GELU(),
            nn.Linear(512, num_classes),
        )

        # ── ImageNet normalisation constants (registered as buffers) ─────────
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    # ── Construction helpers ──────────────────────────────────────────────────

    @staticmethod
    def _build_encoder(model_name: str, pretrained: bool) -> EfficientNet:
        """Return a bare EfficientNet (pre-trained weights, FC head unused)."""
        if pretrained:
            return EfficientNet.from_pretrained(model_name)
        return EfficientNet.from_name(model_name)

    def _init_srm_kernels(self) -> None:
        """Initialise SRM conv weights with Laplacian-like kernels + small noise."""
        base = torch.tensor(
            [[0.0,  0.0, -1.0,  0.0,  0.0],
             [0.0,  0.0, -2.0,  0.0,  0.0],
             [-1.0, -2.0, 16.0, -2.0, -1.0],
             [0.0,  0.0, -2.0,  0.0,  0.0],
             [0.0,  0.0, -1.0,  0.0,  0.0]],
            dtype=torch.float32,
        )
        base = base / (base.abs().sum() + 1e-6)
        with torch.no_grad():
            w = self._srm_conv.weight  # [K, 1, 5, 5]
            for k in range(w.shape[0]):
                w[k, 0].copy_(base + 0.02 * torch.randn_like(base))

    # ── ImageNet normalisation helpers ────────────────────────────────────────

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        """Undo ImageNet normalisation → [0, 1]."""
        return (x * self._std + self._mean).clamp(0.0, 1.0)

    def _renorm(self, x01: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalisation to a [0, 1] tensor."""
        return (x01 - self._mean) / (self._std + 1e-6)

    @staticmethod
    def _scale01(t: torch.Tensor) -> torch.Tensor:
        """Min-max scale each sample to [0, 1] over its spatial dimensions."""
        mn = t.amin(dim=(-2, -1), keepdim=True)
        mx = t.amax(dim=(-2, -1), keepdim=True)
        return (t - mn) / (mx - mn + 1e-6)

    # ── Feature extraction ────────────────────────────────────────────────────

    def _encode(self, encoder: EfficientNet, x: torch.Tensor) -> torch.Tensor:
        """
        Forward through EfficientNet up to global average-pool; skip the FC
        head.  Returns a feature vector of shape [B, feat_dim].
        """
        x = encoder.extract_features(x)   # [B, C, H', W']
        x = encoder._avg_pooling(x)        # [B, C, 1, 1]
        x = x.flatten(start_dim=1)         # [B, C]
        x = encoder._dropout(x)
        return x

    # ── Frequency stream ──────────────────────────────────────────────────────

    @staticmethod
    def _dct2d(x: torch.Tensor) -> torch.Tensor:
        """
        2-D DCT-II via FFT applied row-wise then column-wise.

        Input : x  [B, H, W]  (real-valued, any range)
        Output: [B, H, W]  DCT coefficients
        """
        def _dct1d(v: torch.Tensor, dim: int) -> torch.Tensor:
            N = v.shape[dim]
            # Mirror-extend so FFT gives DCT-II.
            v_ext = torch.cat([v, v.flip(dim)], dim=dim)
            V = torch.fft.rfft(v_ext, dim=dim)
            # Phase shift.
            k_shape = [1] * v.ndim
            k_shape[dim] = N
            k = torch.arange(N, device=v.device, dtype=torch.float32).view(k_shape)
            phase = torch.exp(
                -1j * torch.pi * k.to(torch.complex64) / (2 * N)
            )
            idx = [slice(None)] * v.ndim
            idx[dim] = slice(0, N)
            return (V[tuple(idx)] * phase).real

        return _dct1d(_dct1d(x, dim=-1), dim=-2)

    def _compute_freq_channels(self, x_rgb: torch.Tensor) -> torch.Tensor:
        """
        Derive 3-channel frequency representation from the RGB input:
          Ch 1 – FFT log-magnitude spectrum   (captures GAN frequency artefacts)
          Ch 2 – FFT phase spectrum            (captures geometric manipulations)
          Ch 3 – DCT log-magnitude             (complementary energy distribution)

        Returns an ImageNet-normalised tensor [B, 3, input_size, input_size].
        """
        x01 = self._denorm(x_rgb)
        # Luminance grayscale.
        gray = (0.2989 * x01[:, 0:1]
                + 0.5870 * x01[:, 1:2]
                + 0.1140 * x01[:, 2:3])           # [B, 1, H, W]
        gray = F.interpolate(
            gray, size=(self._input_size, self._input_size),
            mode="bilinear", align_corners=False
        )
        g = gray.squeeze(1)                        # [B, H, W]

        # FFT (centred).
        fft_s = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1))

        # Ch 1: log-magnitude.
        mag = self._scale01(torch.log1p(torch.abs(fft_s)))

        # Ch 2: phase → [0, 1].
        phase = (torch.angle(fft_s) / torch.pi + 1.0) * 0.5

        # Ch 3: DCT log-magnitude.
        dct = self._scale01(torch.log1p(self._dct2d(g).abs()))

        freq01 = torch.stack([mag, phase, dct], dim=1)   # [B, 3, H, W]
        return self._renorm(freq01)

    # ── SRM stream ────────────────────────────────────────────────────────────

    def _compute_srm_channels(self, x_rgb: torch.Tensor) -> torch.Tensor:
        """
        Apply 30 learnable Laplacian-like high-pass filters to grayscale image,
        then compress to 3 channels.

        Returns an ImageNet-normalised tensor [B, 3, input_size, input_size].
        """
        x01 = self._denorm(x_rgb)
        gray = (0.2989 * x01[:, 0:1]
                + 0.5870 * x01[:, 1:2]
                + 0.1140 * x01[:, 2:3])            # [B, 1, H, W]
        gray = F.interpolate(
            gray, size=(self._input_size, self._input_size),
            mode="bilinear", align_corners=False
        )
        noise = torch.abs(self._srm_conv(gray))    # [B, 30, H, W]
        out3 = F.relu(self._srm_to3(noise))         # [B, 3, H, W]
        mn = out3.amin(dim=(-2, -1), keepdim=True)
        mx = out3.amax(dim=(-2, -1), keepdim=True)
        out01 = (out3 - mn) / (mx - mn + 1e-6)
        return self._renorm(out01)

    # ── Feature extraction (for temporal model) ──────────────────────────────

    def encode_frame(self, x_rgb: torch.Tensor) -> torch.Tensor:
        """
        Extract fused feature vector for a single frame without classifying.
        Returns [B, feat_dim] — used by TemporalTriStreamDetector.
        """
        if x_rgb.shape[-1] != self._input_size or x_rgb.shape[-2] != self._input_size:
            x_rgb = F.interpolate(
                x_rgb, size=(self._input_size, self._input_size),
                mode="bilinear", align_corners=False,
            )
        f_rgb  = self._encode(self.rgb_encoder, x_rgb)
        x_freq = self._compute_freq_channels(x_rgb)
        f_freq = self._encode(self.freq_encoder, x_freq)
        x_srm  = self._compute_srm_channels(x_rgb)
        f_srm  = self._encode(self.srm_encoder, x_srm)
        return self.fusion([f_rgb, f_freq, f_srm])   # [B, feat_dim]

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_rgb: ImageNet-normalised face crop [B, 3, H, W].

        Returns:
            Class logits [B, num_classes].
        """
        # Resize input to backbone's expected resolution if necessary.
        if x_rgb.shape[-1] != self._input_size or x_rgb.shape[-2] != self._input_size:
            x_rgb = F.interpolate(
                x_rgb, size=(self._input_size, self._input_size),
                mode="bilinear", align_corners=False,
            )

        # ── 3 feature vectors ────────────────────────────────────────────────
        f_rgb = self._encode(self.rgb_encoder, x_rgb)

        x_freq = self._compute_freq_channels(x_rgb)
        f_freq = self._encode(self.freq_encoder, x_freq)

        x_srm = self._compute_srm_channels(x_rgb)
        f_srm = self._encode(self.srm_encoder, x_srm)

        # ── Channel Attention Fusion → fused vector ──────────────────────────
        fused = self.fusion([f_rgb, f_freq, f_srm])   # [B, feat_dim]

        # ── Classify ─────────────────────────────────────────────────────────
        return self.classifier(fused)                  # [B, num_classes]

    # ── Utility methods ───────────────────────────────────────────────────────

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def save_checkpoint(
        self,
        checkpoint_path: str,
        epoch: Optional[int] = None,
        optimizer_state: Optional[dict] = None,
        scheduler_state: Optional[dict] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        ckpt: Dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "backbone": self._backbone_name,
            "num_classes": self.classifier[-1].out_features,
        }
        if epoch is not None:
            ckpt["epoch"] = epoch
        if optimizer_state is not None:
            ckpt["optimizer_state_dict"] = optimizer_state
        if scheduler_state is not None:
            ckpt["scheduler_state_dict"] = scheduler_state
        if metrics is not None:
            ckpt["metrics"] = metrics
        torch.save(ckpt, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str, device: str = "cpu") -> dict:
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.load_state_dict(ckpt["model_state_dict"])
            return ckpt
        self.load_state_dict(ckpt)
        return {}

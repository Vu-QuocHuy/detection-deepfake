"""
TemporalTriStreamDetector — adds temporal Transformer on top of the
existing per-frame TriStream spatial encoder.

Architecture:
  Input: [B, T, 3, H, W]   (B=batch, T=num_frames per video)
      ↓  shared TriStream encoder  (RGB + Freq + SRM → Channel Attention Fusion)
  Frame features: [B, T, D]   D = feat_dim of backbone
      ↓  prepend learnable CLS token  → [B, T+1, D]
      ↓  sinusoidal positional encoding
      ↓  Transformer encoder  (Pre-LN, num_layers × num_heads)
  CLS output: [B, D]
      ↓  Classifier:  LayerNorm → Dropout → FC(512) → GELU → FC(num_classes)
  Output: [B, num_classes]

Why this is better than frame-level mean aggregation:
  - Captures flickering, temporal blending inconsistencies
  - CLS-token attention can selectively focus on suspicious frames
  - Pre-LN Transformer is more stable for fine-tuning

Training strategy (recommended):
  Phase 1 — freeze_backbone=True:  train only Temporal Transformer + head (5-10 ep)
  Phase 2 — freeze_backbone=False: end-to-end fine-tune with lower LR (10-20 ep)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfake_detector.models.multistream import (
    TriStreamDeepFakeDetector,
    _effnet_feat_dim,
    _effnet_input_size,
)


# ──────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal PE (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ──────────────────────────────────────────────────────────────────────────────
# Temporal Transformer
# ──────────────────────────────────────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    """
    Pre-LN Transformer encoder for temporal reasoning over frame features.
    Uses a CLS token to aggregate the sequence into a single vector.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_frames: int = 64,
    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=max_frames + 1, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,     # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_features: [B, T, D]  per-frame feature vectors
        Returns:
            cls_output: [B, D]  aggregated video representation
        """
        B = frame_features.size(0)
        cls = self.cls_token.expand(B, -1, -1)              # [B, 1, D]
        x = torch.cat([cls, frame_features], dim=1)          # [B, T+1, D]
        x = self.pos_enc(x)
        x = self.encoder(x)
        return x[:, 0]                                        # [B, D] — CLS token


# ──────────────────────────────────────────────────────────────────────────────
# Main Model
# ──────────────────────────────────────────────────────────────────────────────

class TemporalTriStreamDetector(nn.Module):
    """
    Temporal deepfake detector: TriStream spatial features + Temporal Transformer.

    2-Phase training strategy (recommended):
      Phase 1 — freeze Transformer, train spatial only:
        model.set_phase(1)
        → Spatial encoder learns frame-level artifact detection first.
        → Forward uses mean-pool instead of Transformer (stable gradients).
        → 5–10 epochs sufficient.

      Phase 2 — unfreeze all, end-to-end:
        model.set_phase(2)
        → Transformer learns temporal aggregation on stable spatial features.
        → 20–30 epochs for convergence.

    Args:
        backbone:              EfficientNet variant ('efficientnet-b1/b2/b4').
        n_frames:              Fixed T (e.g. T=16). Must match VideoSequenceDataset.
        num_heads:             Attention heads in Temporal Transformer.
        num_layers:            Transformer encoder layers.
        srm_filters:           Learnable SRM high-pass filter count.
        pretrained:            Load ImageNet weights for EfficientNet backbone.
        bce_output:            FC(1)+BCEWithLogitsLoss if True, FC(2)+CE if False.
        use_grad_checkpoint:   Gradient checkpointing on backbone.
                               Required for B4 + fine-tuning on ≤16 GB VRAM.
                               Reduces activation memory ~6x, ~30% slower.
        dropout:               Dropout in Transformer and classifier.
    """

    def __init__(
        self,
        backbone: str = 'efficientnet-b4',
        n_frames: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
        srm_filters: int = 30,
        pretrained: bool = True,
        bce_output: bool = True,
        use_grad_checkpoint: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.n_frames = n_frames
        self.bce_output = bce_output
        self.use_grad_checkpoint = use_grad_checkpoint
        feat_dim = _effnet_feat_dim(backbone)

        # Phase flag: 1 = spatial-only (mean-pool), 2 = full temporal
        self._phase: int = 2

        # ── Spatial feature extractor ─────────────────────────────────────────
        # 3 separate EfficientNet encoders (rgb, freq, srm) — intentionally
        # NOT weight-shared so each stream learns its own representations.
        # With B4 + grad_checkpoint: activation memory ≈ 1 frame at a time.
        self.spatial = TriStreamDeepFakeDetector(
            rgb_model=backbone,
            srm_filters=srm_filters,
            pretrained=pretrained,
            num_classes=2,
        )

        if use_grad_checkpoint:
            self._enable_grad_checkpointing()

        # ── Temporal Transformer ─────────────────────────────────────────────
        self.temporal = TemporalTransformer(
            d_model=feat_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=feat_dim * 2,
            dropout=dropout,
            max_frames=n_frames + 4,
        )

        # ── Classifier head ──────────────────────────────────────────────────
        # FC(1) + BCEWithLogitsLoss: standard for binary classification.
        out_dim = 1 if bce_output else 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Linear(512, out_dim),
        )

        self._feat_dim = feat_dim

    # ── Phase management ──────────────────────────────────────────────────────

    def set_phase(self, phase: int) -> None:
        """
        Switch between training phases.

        Phase 1: Spatial-only training (Transformer frozen + bypassed).
          → forward uses mean-pool of frame features → classifier
          → Only spatial encoder + channel attention + classifier update
          → Gradient signal is clean (no random Transformer interference)

        Phase 2: Full temporal training (all params active).
          → forward uses Temporal Transformer → CLS token → classifier
          → Spatial features are already good from Phase 1
        """
        assert phase in (1, 2), "phase must be 1 or 2"
        self._phase = phase
        if phase == 1:
            # Freeze Transformer: it's not used in phase 1 forward
            for p in self.temporal.parameters():
                p.requires_grad = False
            for p in self.spatial.parameters():
                p.requires_grad = True
            for p in self.classifier.parameters():
                p.requires_grad = True
        else:
            # Unfreeze everything for end-to-end fine-tuning
            for p in self.parameters():
                p.requires_grad = True

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Toggle spatial backbone gradient flow."""
        for p in self.spatial.parameters():
            p.requires_grad = not freeze

    def freeze_temporal(self, freeze: bool = True) -> None:
        """Toggle Temporal Transformer gradient flow."""
        for p in self.temporal.parameters():
            p.requires_grad = not freeze

    # ── Gradient checkpointing ────────────────────────────────────────────────

    def _enable_grad_checkpointing(self) -> None:
        """
        Enable gradient checkpointing on all 3 EfficientNet backbones.
        Reduces activation memory ~6x at cost of ~30% slower backward.
        Key insight: with checkpointing, VRAM usage is INDEPENDENT of T —
        only 1 frame's activations are live at a time regardless of T=8/16/32.
        """
        from torch.utils.checkpoint import checkpoint as ckpt_fn
        for encoder in [self.spatial.rgb_encoder,
                        self.spatial.freq_encoder,
                        self.spatial.srm_encoder]:
            if hasattr(encoder, 'set_grad_checkpointing'):
                encoder.set_grad_checkpointing(True)
            else:
                for block in encoder._blocks:
                    original_forward = block.forward
                    def make_ckpt(fn):
                        def _ckpt_forward(*args, **kwargs):
                            return ckpt_fn(fn, *args, use_reentrant=False,
                                           **kwargs)
                        return _ckpt_forward
                    block.forward = make_ckpt(original_forward)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 3, H, W]   T must equal self.n_frames

        Phase 1 forward (set_phase(1)):
            frame features [B, T, D]
            → mean pool → [B, D]   (no Transformer — stable gradients)
            → classifier → [B, out_dim]

        Phase 2 forward (set_phase(2)):
            frame features [B, T, D]
            → Temporal Transformer (CLS token) → [B, D]
            → classifier → [B, out_dim]
        """
        B, T, C, H, W = x.shape

        # Extract per-frame spatial features sequentially.
        # With grad_checkpoint: peak VRAM = 1 frame regardless of T.
        frame_features = []
        for t in range(T):
            feat = self.spatial.encode_frame(x[:, t])   # [B, D]
            frame_features.append(feat)

        features = torch.stack(frame_features, dim=1)    # [B, T, D]

        if self._phase == 1:
            # Phase 1: simple mean-pool — no Transformer interference
            # Trains spatial encoder to produce meaningful per-frame features
            video_repr = features.mean(dim=1)            # [B, D]
        else:
            # Phase 2: full temporal reasoning via Transformer
            video_repr = self.temporal(features)         # [B, D]

        logits = self.classifier(video_repr)               # [B, out_dim]
        # For BCE path (FC(1)), make logits shape [B] to match targets shape [B].
        # This avoids BCEWithLogitsLoss broadcasting/mismatch issues.
        if self.bce_output:
            logits = logits.squeeze(-1)                  # [B]
        return logits

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def save_checkpoint(
        self,
        path: str,
        epoch: Optional[int] = None,
        optimizer_state: Optional[dict] = None,
        scheduler_state: Optional[dict] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        ckpt: Dict[str, Any] = {
            'model_state_dict': self.state_dict(),
            'backbone': self.backbone_name,
            'n_frames': self.n_frames,
            'bce_output': self.bce_output,
            'out_dim': self.classifier[-1].out_features,
        }
        if epoch is not None:
            ckpt['epoch'] = epoch
        if optimizer_state is not None:
            ckpt['optimizer_state_dict'] = optimizer_state
        if scheduler_state is not None:
            ckpt['scheduler_state_dict'] = scheduler_state
        if metrics is not None:
            ckpt['metrics'] = metrics
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, device: str = 'cpu') -> dict:
        ckpt = torch.load(path, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        self.load_state_dict(state)
        return ckpt

    def load_spatial_weights(self, tristream_checkpoint: str, device: str = 'cpu') -> None:
        """
        Bootstrap from a pre-trained frame-level TriStream checkpoint.
        Only weights that shape-match are loaded; the temporal head starts fresh.
        """
        ckpt = torch.load(tristream_checkpoint, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        own = self.spatial.state_dict()
        loaded, skipped = 0, 0
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
                loaded += 1
            else:
                skipped += 1
        self.spatial.load_state_dict(own)
        print(f"Loaded {loaded} spatial weights, skipped {skipped} (shape mismatch / temporal head)")

"""Một luồng không gian + temporal giống TemporalTriStreamDetector; checkpoint có `spatial_forward` / `single_stream`."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from deepfake_detector.baselines.stream_frame_features import encode_frame_one_stream
from deepfake_detector.models.temporal import TemporalTriStreamDetector


class TemporalSingleStreamAblation(TemporalTriStreamDetector):
    """state_dict tương thích TemporalTriStreamDetector; forward một luồng."""

    def __init__(self, *args, single_stream: str = "rgb", **kwargs):
        if single_stream not in ("rgb", "freq", "srm"):
            raise ValueError("single_stream must be 'rgb', 'freq', or 'srm'")
        super().__init__(*args, **kwargs)
        self.single_stream = single_stream

    def save_checkpoint(
        self,
        path: str,
        epoch: Optional[int] = None,
        optimizer_state: Optional[dict] = None,
        scheduler_state: Optional[dict] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        ckpt: Dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "backbone": self.backbone_name,
            "n_frames": self.n_frames,
            "bce_output": self.bce_output,
            "out_dim": self.classifier[-1].out_features,
            "single_stream": self.single_stream,
            "spatial_forward": "single_stream",
        }
        if epoch is not None:
            ckpt["epoch"] = epoch
        if optimizer_state is not None:
            ckpt["optimizer_state_dict"] = optimizer_state
        if scheduler_state is not None:
            ckpt["scheduler_state_dict"] = scheduler_state
        if metrics is not None:
            ckpt["metrics"] = metrics
        torch.save(ckpt, path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _, _, _ = x.shape
        frame_features = []
        for t in range(T):
            feat = encode_frame_one_stream(self.spatial, x[:, t], self.single_stream)
            frame_features.append(feat)

        features = torch.stack(frame_features, dim=1)

        if self._phase == 1:
            video_repr = features.mean(dim=1)
        else:
            video_repr = self.temporal(features)

        logits = self.classifier(video_repr)
        if self.bce_output:
            logits = logits.squeeze(-1)
        return logits

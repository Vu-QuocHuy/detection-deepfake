"""Feature một frame: rgb | freq | srm (không fusion)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from deepfake_detector.models.multistream import TriStreamDeepFakeDetector

def encode_frame_one_stream(
    spatial: TriStreamDeepFakeDetector,
    x_rgb: torch.Tensor,
    stream: str,
) -> torch.Tensor:
    if stream not in ("rgb", "freq", "srm"):
        raise ValueError(f"stream must be rgb|freq|srm, got {stream!r}")

    inp = spatial._input_size
    if x_rgb.shape[-1] != inp or x_rgb.shape[-2] != inp:
        x_rgb = F.interpolate(
            x_rgb,
            size=(inp, inp),
            mode="bilinear",
            align_corners=False,
        )

    if stream == "rgb":
        return spatial._encode(spatial.rgb_encoder, x_rgb)

    if stream == "freq":
        x_freq = spatial._compute_freq_channels(x_rgb)
        return spatial._encode(spatial.freq_encoder, x_freq)

    x_srm = spatial._compute_srm_channels(x_rgb)
    return spatial._encode(spatial.srm_encoder, x_srm)

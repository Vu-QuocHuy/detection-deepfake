"""Model definitions for deepfake detection."""

from deepfake_detector.models.multistream import TriStreamDeepFakeDetector
from deepfake_detector.models.temporal import TemporalTriStreamDetector

__all__ = ["TriStreamDeepFakeDetector", "TemporalTriStreamDetector"]

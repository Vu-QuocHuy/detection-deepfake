"""Model definitions for deepfake detection."""

from deepfake_detector.models.efficientnet import DeepFakeDetector
from deepfake_detector.models.multistream import TriStreamDeepFakeDetector

__all__ = ["DeepFakeDetector", "TriStreamDeepFakeDetector"]

"""
Domain layer: core tracking algorithms and models.
"""

from mttrack.domain.models import Detection, Track, FrameResult, LabelResult
from mttrack.domain.tracker import BaseTracker, TrackerInfo
from mttrack.domain.kalman import KalmanBoxTracker
from mttrack.domain.bytetrack import ByteTrackTracker
from mttrack.domain.sort import SORTTracker
from mttrack.domain.appearance import AppearanceFeatureExtractor, AppearanceTracker
from mttrack.domain.adaptive_trigger import AdaptiveVLTrigger, TriggerDecision, TrackState
from mttrack.domain.association import MultiFeatureAssociation, AssociationConfig, AdaptiveThreshold

__all__ = [
    "Detection",
    "Track",
    "FrameResult",
    "LabelResult",
    "BaseTracker",
    "TrackerInfo",
    "KalmanBoxTracker",
    "ByteTrackTracker",
    "SORTTracker",
    # New: Appearance features
    "AppearanceFeatureExtractor",
    "AppearanceTracker",
    # New: Adaptive VL trigger
    "AdaptiveVLTrigger",
    "TriggerDecision",
    "TrackState",
    # New: Multi-feature association
    "MultiFeatureAssociation",
    "AssociationConfig",
    "AdaptiveThreshold",
]

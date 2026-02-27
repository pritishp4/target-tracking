"""
Enhanced tracker service with advanced features:
- Appearance feature extraction for robust re-identification
- Adaptive VL classification triggering
- Multi-feature fusion association
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from mttrack.domain import (
    BaseTracker,
    ByteTrackTracker,
    SORTTracker,
    AppearanceFeatureExtractor,
    AppearanceTracker,
    AdaptiveVLTrigger,
    MultiFeatureAssociation,
    AdaptiveThreshold,
    TrackerInfo,
)
from mttrack.infrastructure import BaseDetector, YoloDetector, DetectorResult


@dataclass
class TrackInfo:
    """Track information with enhanced features."""

    track_id: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    class_name: str
    class_id: int
    confidence: float
    label: Optional[str] = None
    label_confidence: float = 0.0
    # Enhanced features
    velocity: tuple[float, float] = (0.0, 0.0)
    appearance_feature: Optional[np.ndarray] = None
    appearance_change: float = 0.0


@dataclass
class FrameTracks:
    """Tracks for a single frame."""

    frame_id: int
    tracks: list[TrackInfo]


class EnhancedTrackerService:
    """Enhanced tracker service with advanced features.

    Features:
    - Appearance feature extraction for re-identification
    - Adaptive VL classification triggering
    - Multi-feature fusion data association
    """

    def __init__(
        self,
        detector: BaseDetector,
        tracker_type: str = "bytetrack",
        tracker_kwargs: Optional[dict] = None,
        # Appearance settings
        enable_appearance: bool = True,
        appearance_memory_size: int = 10,
        # Adaptive VL settings
        enable_adaptive_vl: bool = True,
        vl_min_interval: int = 30,
        vl_max_interval: int = 150,
        # Multi-feature association settings
        enable_multi_feature: bool = True,
        use_appearance_in_association: bool = True,
    ) -> None:
        """Initialize enhanced tracker service.

        Args:
            detector: Object detector
            tracker_type: Type of tracker ("bytetrack" or "sort")
            tracker_kwargs: Additional tracker arguments
            enable_appearance: Enable appearance feature extraction
            appearance_memory_size: Memory size for appearance features
            enable_adaptive_vl: Enable adaptive VL triggering
            vl_min_interval: Minimum frames between VL classifications
            vl_max_interval: Maximum frames between VL classifications
            enable_multi_feature: Enable multi-feature association
            use_appearance_in_association: Use appearance in data association
        """
        self.detector = detector
        self.tracker_kwargs = tracker_kwargs or {}
        self.tracker_type = tracker_type
        self._tracker: Optional[BaseTracker] = None
        self._frame_count = 0

        # Enhanced features flags
        self.enable_appearance = enable_appearance
        self.enable_adaptive_vl = enable_adaptive_vl
        self.enable_multi_feature = enable_multi_feature

        # Initialize appearance feature extractor
        if self.enable_appearance:
            self._appearance_extractor = AppearanceFeatureExtractor()
            self._appearance_tracker = AppearanceTracker(
                extractor=self._appearance_extractor,
                memory_size=appearance_memory_size,
            )
        else:
            self._appearance_extractor = None
            self._appearance_tracker = None

        # Initialize adaptive VL trigger
        if self.enable_adaptive_vl:
            self._vl_trigger = AdaptiveVLTrigger(
                min_interval_frames=vl_min_interval,
                max_interval_frames=vl_max_interval,
            )
        else:
            self._vl_trigger = None

        # Initialize multi-feature association
        if self.enable_multi_feature:
            self._mf_association = MultiFeatureAssociation()
            self._adaptive_threshold = AdaptiveThreshold()
        else:
            self._mf_association = None
            self._adaptive_threshold = None

        # Track info storage
        self._track_labels: dict[int, dict] = {}
        self._track_class_names: dict[int, str] = {}

    def _create_tracker(self) -> BaseTracker:
        """Create tracker instance."""
        if self.tracker_type == "bytetrack":
            return ByteTrackTracker(**self.tracker_kwargs)
        elif self.tracker_type == "sort":
            return SORTTracker(**self.tracker_kwargs)
        else:
            raise ValueError(f"Unknown tracker type: {self.tracker_type}")

    def process_frame(self, frame: np.ndarray) -> FrameTracks:
        """Process a single frame.

        Args:
            frame: BGR image

        Returns:
            FrameTracks with detected and tracked objects
        """
        self._frame_count += 1

        # Detect objects
        det_result = self.detector.detect(frame)

        if len(det_result.boxes) == 0:
            if self._tracker is None:
                self._tracker = self._create_tracker()
            self._tracker.update(
                np.array([]).reshape(0, 4),
                np.array([]),
                np.array([], dtype=int)
            )
            return FrameTracks(frame_id=self._frame_count, tracks=[])

        # Initialize tracker if needed
        if self._tracker is None:
            self._tracker = self._create_tracker()

        # Update tracker
        tracker_ids = self._tracker.update(
            det_result.boxes,
            det_result.confidences,
            det_result.class_ids
        )

        # Build result with enhanced features
        tracks = []
        active_track_ids = set()

        for i, (box, conf, cls_id, cls_name, trk_id) in enumerate(zip(
            det_result.boxes,
            det_result.confidences,
            det_result.class_ids,
            det_result.class_names,
            tracker_ids
        )):
            if trk_id < 0:
                continue

            track_id = int(trk_id)
            active_track_ids.add(track_id)

            # Get cached label
            label_info = self._track_labels.get(track_id, {})
            label = label_info.get("label")
            label_conf = label_info.get("confidence", 0.0)

            # Compute appearance feature if enabled
            appearance_feat = None
            appearance_change = 0.0

            if self.enable_appearance and self._appearance_tracker is not None:
                # Crop image
                crop = self._crop_bbox(frame, box)
                if crop is not None and crop.size > 0:
                    # Update appearance tracker
                    self._appearance_tracker.update(track_id, crop)
                    appearance_feat = self._appearance_tracker.get_feature(track_id)

                    # Compute appearance change
                    appearance_change = self._appearance_tracker.compute_appearance_change(
                        track_id, crop
                    )

            # Compute velocity (simple difference)
            velocity = self._compute_velocity(track_id, box)

            # Update class name
            self._track_class_names[track_id] = cls_name

            tracks.append(TrackInfo(
                track_id=track_id,
                bbox=tuple(box.tolist()),
                class_name=cls_name,
                class_id=int(cls_id),
                confidence=float(conf),
                label=label,
                label_confidence=label_conf,
                velocity=velocity,
                appearance_feature=appearance_feat,
                appearance_change=appearance_change,
            ))

        # Update motion history for multi-feature association
        if self.enable_multi_feature and self._mf_association is not None:
            for track in tracks:
                self._mf_association.update_motion_history(
                    track.track_id, track.velocity
                )

        # Cleanup old tracks
        self._cleanup_stale_tracks(active_track_ids)

        return FrameTracks(frame_id=self._frame_count, tracks=tracks)

    def _crop_bbox(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        margin: int = 5,
    ) -> Optional[np.ndarray]:
        """Crop bbox from frame with margin."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]

        # Add margin
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    def _compute_velocity(
        self,
        track_id: int,
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float]:
        """Compute velocity for a track."""
        # Simple velocity computation from bbox center movement
        # In a real implementation, you'd track this over time
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        # This is a placeholder - real velocity tracking would be more sophisticated
        return (0.0, 0.0)

    def should_classify_vl(
        self,
        track_id: int,
        bbox: tuple[float, float, float, float],
        current_confidence: float = 0.0,
    ) -> tuple[bool, str]:
        """Check if VL classification should be triggered using adaptive strategy.

        Args:
            track_id: Track ID
            bbox: Current bounding box
            current_confidence: Current classification confidence

        Returns:
            (should_trigger, reason) tuple
        """
        if not self.enable_adaptive_vl or self._vl_trigger is None:
            return False, "Adaptive VL disabled"

        # Get appearance change if available
        appearance_change = 0.0
        if self._appearance_tracker is not None:
            # We'd need to track this in process_frame
            pass

        # Make decision
        decision = self._vl_trigger.should_classify(
            track_id=track_id,
            frame_id=self._frame_count,
            bbox=bbox,
            appearance_change=appearance_change,
            current_confidence=current_confidence,
        )

        return decision.should_trigger, decision.reason

    def update_track_label(
        self,
        track_id: int,
        label: str,
        confidence: float,
    ) -> None:
        """Update label for a track.

        Args:
            track_id: Track ID
            label: Class label
            confidence: Label confidence
        """
        self._track_labels[track_id] = {
            "label": label,
            "confidence": confidence
        }

        # Update VL trigger state
        if self.enable_adaptive_vl and self._vl_trigger is not None:
            self._vl_trigger.update_classification_result(
                track_id, label, confidence
            )

    def get_track_label(self, track_id: int) -> Optional[dict]:
        """Get label for a track."""
        return self._track_labels.get(track_id)

    def get_appearance_feature(self, track_id: int) -> Optional[np.ndarray]:
        """Get appearance feature for a track."""
        if self._appearance_tracker is None:
            return None
        return self._appearance_tracker.get_feature(track_id)

    def get_track_state_info(self, track_id: int) -> Optional[any]:
        """Get detailed track state information."""
        if self._vl_trigger is None:
            return None
        return self._vl_trigger.get_track_info(track_id)

    def _cleanup_stale_tracks(self, active_track_ids: set[int]) -> None:
        """Clean up stale track data."""
        # Clean up appearance tracker
        if self._appearance_tracker is not None:
            self._appearance_tracker.cleanup(active_track_ids)

        # Clean up VL trigger
        if self._vl_trigger is not None:
            self._vl_trigger.cleanup(active_track_ids)

        # Clean up multi-feature association
        if self._mf_association is not None:
            self._mf_association.cleanup(active_track_ids)

        # Clean up track labels for inactive tracks
        stale_ids = set(self._track_labels.keys()) - active_track_ids
        for track_id in stale_ids:
            self._track_labels.pop(track_id, None)
            self._track_class_names.pop(track_id, None)

    def reset(self) -> None:
        """Reset tracker state."""
        if self._tracker:
            self._tracker.reset()
        self._frame_count = 0
        self._track_labels.clear()
        self._track_class_names.clear()

        if self._appearance_tracker:
            self._appearance_tracker._history.clear()
            self._appearance_tracker._last_features.clear()

        if self._vl_trigger:
            self._vl_trigger._track_states.clear()

    def warmup(self) -> None:
        """Warm up detector."""
        self.detector.warmup()


# Note: The EnhancedTrackerService class itself is the main service.
# No additional registry class needed - it's used directly.

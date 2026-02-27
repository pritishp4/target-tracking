"""
Multi-feature fusion data association.

This module provides enhanced data association by combining:
- IoU (Intersection over Union) similarity
- Appearance feature similarity
- Motion consistency (velocity similarity)
- Size similarity

This is more robust than traditional IoU-only association, especially
for objects with occlusions or similar appearance.
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class AssociationConfig:
    """Configuration for multi-feature association."""
    iou_weight: float = 0.4
    appearance_weight: float = 0.35
    motion_weight: float = 0.15
    size_weight: float = 0.1

    # Thresholds
    iou_threshold: float = 0.1
    appearance_threshold: float = 0.3
    motion_threshold: float = 0.5
    size_threshold: float = 0.5

    # Enable flags
    use_appearance: bool = True
    use_motion: bool = True
    use_size: bool = True


class MultiFeatureAssociation:
    """Multi-feature fusion for data association.

    Combines multiple similarity metrics for robust tracking,
    especially in challenging scenarios like occlusions.
    """

    def __init__(
        self,
        config: Optional[AssociationConfig] = None,
    ) -> None:
        """Initialize multi-feature association.

        Args:
            config: Association configuration
        """
        self.config = config or AssociationConfig()

        # Track motion history: track_id -> list of (velocity_x, velocity_y)
        self._motion_history: dict[int, list[tuple[float, float]]] = {}

    def compute_similarity_matrix(
        self,
        tracks: list[dict],
        detections: np.ndarray,
        appearance_features: Optional[dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """Compute similarity matrix between tracks and detections.

        Args:
            tracks: List of track info dicts with 'bbox' and 'velocity'
            detections: Array of detections (N, 4) in xyxy format
            appearance_features: Dict of track_id -> appearance feature

        Returns:
            Similarity matrix (M x N) where M = len(tracks), N = len(detections)
        """
        n_tracks = len(tracks)
        n_dets = len(detections)

        if n_tracks == 0 or n_dets == 0:
            return np.zeros((n_tracks, n_dets), dtype=np.float32)

        # Compute individual similarity matrices
        iou_sim = self._compute_iou_similarity(tracks, detections)

        appearance_sim = None
        if self.config.use_appearance and appearance_features:
            appearance_sim = self._compute_appearance_similarity(
                tracks, detections, appearance_features
            )

        motion_sim = None
        if self.config.use_motion:
            motion_sim = self._compute_motion_similarity(tracks, detections)

        size_sim = None
        if self.config.use_size:
            size_sim = self._compute_size_similarity(tracks, detections)

        # Combine similarity matrices
        combined = np.zeros((n_tracks, n_dets), dtype=np.float32)

        combined += iou_sim * self.config.iou_weight

        if appearance_sim is not None:
            combined += appearance_sim * self.config.appearance_weight
        else:
            combined += iou_sim * self.config.appearance_weight

        if motion_sim is not None:
            combined += motion_sim * self.config.motion_weight
        else:
            combined += iou_sim * self.config.motion_weight

        if size_sim is not None:
            combined += size_sim * self.config.size_weight

        return combined

    def _compute_iou_similarity(
        self,
        tracks: list[dict],
        detections: np.ndarray,
    ) -> np.ndarray:
        """Compute IoU-based similarity matrix."""
        n_tracks = len(tracks)
        n_dets = len(detections)

        similarity = np.zeros((n_tracks, n_dets), dtype=np.float32)

        for i, track in enumerate(tracks):
            track_bbox = track.get('bbox')
            if track_bbox is None:
                continue

            for j in range(n_dets):
                det_bbox = detections[j]
                iou = self._compute_iou(track_bbox, det_bbox)
                similarity[i, j] = iou

        return similarity

    def _compute_appearance_similarity(
        self,
        tracks: list[dict],
        detections: np.ndarray,
        appearance_features: dict[int, np.ndarray],
    ) -> np.ndarray:
        """Compute appearance similarity matrix.

        For each detection, we need to compute appearance from scratch.
        This is a placeholder - in practice you'd use a feature extractor.
        """
        n_tracks = len(tracks)
        n_dets = len(detections)

        similarity = np.zeros((n_tracks, n_dets), dtype=np.float32)

        for i, track in enumerate(tracks):
            track_id = track.get('track_id')
            if track_id is None or track_id not in appearance_features:
                continue

            track_feat = appearance_features[track_id]

            # For now, return uniform low similarity for detections
            # In practice, you'd extract features from detection crops
            for j in range(n_dets):
                # Use IoU as fallback for now
                similarity[i, j] = 0.2

        return similarity

    def _compute_motion_similarity(
        self,
        tracks: list[dict],
        detections: np.ndarray,
    ) -> np.ndarray:
        """Compute motion consistency similarity."""
        n_tracks = len(tracks)
        n_dets = len(detections)

        similarity = np.zeros((n_tracks, n_dets), dtype=np.float32)

        for i, track in enumerate(tracks):
            track_velocity = track.get('velocity', (0, 0))
            if track_velocity is None:
                track_velocity = (0.0, 0.0)

            # Predict next position based on velocity
            pred_x = track.get('bbox', [0, 0, 0, 0])[2] + track_velocity[0]
            pred_y = track.get('bbox', [0, 0, 0, 0])[3] + track_velocity[1]

            for j in range(n_dets):
                det_bbox = detections[j]
                det_x = (det_bbox[0] + det_bbox[2]) / 2
                det_y = (det_bbox[1] + det_bbox[3]) / 2

                # Distance from predicted position
                dist = np.sqrt((det_x - pred_x)**2 + (det_y - pred_y)**2)

                # Convert distance to similarity (exponential decay)
                sim = np.exp(-dist / 50.0)  # 50 pixels = ~0 similarity
                similarity[i, j] = sim

        return similarity

    def _compute_size_similarity(
        self,
        tracks: list[dict],
        detections: np.ndarray,
    ) -> np.ndarray:
        """Compute size similarity matrix."""
        n_tracks = len(tracks)
        n_dets = len(detections)

        similarity = np.zeros((n_tracks, n_dets), dtype=np.float32)

        for i, track in enumerate(tracks):
            track_bbox = track.get('bbox')
            if track_bbox is None:
                continue

            track_w = track_bbox[2] - track_bbox[0]
            track_h = track_bbox[3] - track_bbox[1]
            track_area = track_w * track_h

            for j in range(n_dets):
                det_bbox = detections[j]
                det_w = det_bbox[2] - det_bbox[0]
                det_h = det_bbox[3] - det_bbox[1]
                det_area = det_w * det_h

                # Size ratio similarity
                if track_area > 0 and det_area > 0:
                    ratio = min(track_area, det_area) / max(track_area, det_area)
                    similarity[i, j] = ratio

        return similarity

    def _compute_iou(
        self,
        box1: tuple[float, float, float, float],
        box2: tuple[float, float, float, float],
    ) -> float:
        """Compute IoU between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def update_motion_history(
        self,
        track_id: int,
        velocity: tuple[float, float],
    ) -> None:
        """Update motion history for a track."""
        if track_id not in self._motion_history:
            self._motion_history[track_id] = []

        self._motion_history[track_id].append(velocity)

        # Keep only recent history
        if len(self._motion_history[track_id]) > 10:
            self._motion_history[track_id] = self._motion_history[track_id][-10:]

    def remove_track(self, track_id: int) -> None:
        """Remove motion history for a track."""
        self._motion_history.pop(track_id, None)

    def cleanup(self, active_track_ids: set[int]) -> None:
        """Remove stale motion history."""
        stale_ids = set(self._motion_history.keys()) - active_track_ids
        for track_id in stale_ids:
            self.remove_track(track_id)


class AdaptiveThreshold:
    """Adaptive association threshold based on scene context.

    Adjusts IoU threshold dynamically based on:
    - Object density (more objects = lower threshold)
    - Frame-to-frame motion (faster motion = lower threshold)
    - Occlusion detection
    """

    def __init__(
        self,
        base_threshold: float = 0.3,
        min_threshold: float = 0.1,
        max_threshold: float = 0.5,
    ) -> None:
        """Initialize adaptive threshold.

        Args:
            base_threshold: Base IoU threshold
            min_threshold: Minimum threshold
            max_threshold: Maximum threshold
        """
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self._recent_densities: list[int] = []
        self._recent_motions: list[float] = []

    def compute_threshold(
        self,
        n_detections: int,
        avg_motion: float,
    ) -> float:
        """Compute adaptive threshold.

        Args:
            n_detections: Number of detections in current frame
            avg_motion: Average motion in pixels/frame

        Returns:
            Adjusted IoU threshold
        """
        # Track density
        self._recent_densities.append(n_detections)
        if len(self._recent_densities) > 30:
            self._recent_densities.pop(0)

        # Track motion
        self._recent_motions.append(avg_motion)
        if len(self._recent_motions) > 30:
            self._recent_motions.pop(0)

        # Compute adjustments
        avg_density = np.mean(self._recent_densities) if self._recent_densities else 1
        avg_motion = np.mean(self._recent_motions) if self._recent_motions else 0

        # Higher density -> lower threshold (more selective)
        density_factor = 1.0 / (1.0 + 0.01 * avg_density)

        # Higher motion -> lower threshold (more selective)
        motion_factor = 1.0 / (1.0 + 0.05 * avg_motion)

        # Compute final threshold
        threshold = self.base_threshold * density_factor * motion_factor

        return np.clip(threshold, self.min_threshold, self.max_threshold)

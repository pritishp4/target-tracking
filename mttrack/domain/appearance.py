"""
Appearance feature extractor for object re-identification.
Uses a lightweight CNN to extract appearance embeddings for robust data association.
"""

from __future__ import annotations

import numpy as np
from typing import Optional
import cv2


class AppearanceFeatureExtractor:
    """Extract appearance features for object re-identification.

    Uses a lightweight approach with color histograms and HOG features
    for fast and memory-efficient feature extraction.
    """

    def __init__(
        self,
        histogram_bins: int = 32,
        hsv_enabled: bool = True,
        feature_dim: int = 256,
    ) -> None:
        """Initialize appearance feature extractor.

        Args:
            histogram_bins: Number of bins for color histogram
            hsv_enabled: Whether to use HSV color space
            feature_dim: Target feature dimension
        """
        self.histogram_bins = histogram_bins
        self.hsv_enabled = hsv_enabled
        self.feature_dim = feature_dim

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract appearance features from cropped image.

        Args:
            image: Cropped image (BGR)

        Returns:
            Feature vector (feature_dim,)
        """
        if image is None or image.size == 0:
            return self._zero_feature()

        # Resize to consistent size
        h, w = image.shape[:2]
        if h < 10 or w < 10:
            return self._zero_feature()

        # Resize for faster processing
        resized = cv2.resize(image, (64, 64))

        # Extract color histogram features
        color_feat = self._extract_color_histogram(resized)

        # Extract edge/gradient features
        edge_feat = self._extract_edge_features(resized)

        # Combine features
        combined = np.concatenate([color_feat, edge_feat])

        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        # Pad or truncate to target dimension
        if len(combined) < self.feature_dim:
            combined = np.pad(combined, (0, self.feature_dim - len(combined)))
        else:
            combined = combined[:self.feature_dim]

        return combined.astype(np.float32)

    def _extract_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """Extract color histogram features."""
        features = []

        # BGR histogram
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [self.histogram_bins], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            features.append(hist)

        # HSV histogram (if enabled)
        if self.hsv_enabled:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            for i in range(3):
                if i == 0:  # Hue
                    hist = cv2.calcHist([hsv], [i], None, [self.histogram_bins], [0, 180])
                else:
                    hist = cv2.calcHist([hsv], [i], None, [self.histogram_bins], [0, 256])
                hist = hist.flatten() / (hist.sum() + 1e-6)
                features.append(hist)

        return np.concatenate(features)

    def _extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """Extract edge and gradient features using Sobel."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Sobel gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Gradient magnitude and direction
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx)

        # Compute histogram of gradient directions (HOG-like)
        hist_bins = 16
        hist, _ = np.histogram(direction, bins=hist_bins, range=(-np.pi, np.pi),
                              weights=magnitude)
        hist = hist / (hist.sum() + 1e-6)

        # Statistics
        stats = [
            magnitude.mean() / 255.0,
            magnitude.std() / 255.0,
            np.percentile(magnitude, 50) / 255.0,
            np.percentile(magnitude, 90) / 255.0,
        ]

        return np.concatenate([hist, np.array(stats)])

    def _zero_feature(self) -> np.ndarray:
        """Return zero feature vector."""
        return np.zeros(self.feature_dim, dtype=np.float32)

    def compute_similarity(
        self,
        feat1: np.ndarray,
        feat2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two features.

        Args:
            feat1: Feature vector 1
            feat2: Feature vector 2

        Returns:
            Similarity score in [0, 1]
        """
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(feat1, feat2) / (norm1 * norm2)
        return float(np.clip(similarity, 0.0, 1.0))


class AppearanceTracker:
    """Track appearance features over time for each object."""

    def __init__(
        self,
        extractor: Optional[AppearanceFeatureExtractor] = None,
        memory_size: int = 10,
    ) -> None:
        """Initialize appearance tracker.

        Args:
            extractor: Appearance feature extractor
            memory_size: Number of frames to remember
        """
        self.extractor = extractor or AppearanceFeatureExtractor()
        self.memory_size = memory_size

        # Track appearance history: track_id -> list of features
        self._history: dict[int, list[np.ndarray]] = {}

        # Track last appearance feature
        self._last_features: dict[int, np.ndarray] = {}

    def update(self, track_id: int, crop: np.ndarray) -> None:
        """Update appearance feature for a track.

        Args:
            track_id: Track ID
            crop: Cropped image
        """
        feature = self.extractor.extract(crop)

        if track_id not in self._history:
            self._history[track_id] = []

        self._history[track_id].append(feature)

        # Keep only recent history
        if len(self._history[track_id]) > self.memory_size:
            self._history[track_id] = self._history[track_id][-self.memory_size:]

        self._last_features[track_id] = feature

    def get_feature(self, track_id: int) -> Optional[np.ndarray]:
        """Get latest appearance feature for a track."""
        return self._last_features.get(track_id)

    def get_average_feature(self, track_id: int) -> Optional[np.ndarray]:
        """Get averaged appearance feature from history."""
        if track_id not in self._history or len(self._history[track_id]) == 0:
            return None

        features = np.array(self._history[track_id])
        return features.mean(axis=0)

    def compute_similarity(
        self,
        track_id1: int,
        track_id2: int
    ) -> float:
        """Compute appearance similarity between two tracks."""
        feat1 = self.get_average_feature(track_id1)
        feat2 = self.get_average_feature(track_id2)

        if feat1 is None or feat2 is None:
            return 0.0

        return self.extractor.compute_similarity(feat1, feat2)

    def compute_appearance_change(
        self,
        track_id: int,
        current_crop: np.ndarray
    ) -> float:
        """Compute appearance change score for a track.

        Higher score means more appearance change occurred.

        Args:
            track_id: Track ID
            current_crop: Current cropped image

        Returns:
            Change score in [0, 1]
        """
        current_feat = self.extractor.extract(current_crop)
        last_feat = self._last_features.get(track_id)

        if last_feat is None:
            return 0.0

        similarity = self.extractor.compute_similarity(current_feat, last_feat)
        return 1.0 - similarity

    def remove_track(self, track_id: int) -> None:
        """Remove track from history."""
        self._history.pop(track_id, None)
        self._last_features.pop(track_id, None)

    def cleanup(self, active_track_ids: set[int]) -> None:
        """Remove history for tracks not in active set."""
        stale_ids = set(self._history.keys()) - active_track_ids
        for track_id in stale_ids:
            self.remove_track(track_id)

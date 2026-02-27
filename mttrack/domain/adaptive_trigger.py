"""
Adaptive VL trigger strategy for intelligent classification.

Instead of fixed-interval triggering, this module makes intelligent
decisions about when to call the VL model based on:
- Object motion state (static vs moving)
- Appearance change magnitude
- Time since last classification
- Classification confidence history
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class MotionState:
    """Motion state of a tracked object."""
    velocity: float = 0.0  # Pixels per frame
    acceleration: float = 0.0  # Acceleration
    is_static: bool = True  # Whether object is nearly static
    motion_direction_changed: bool = False  # Direction change indicator


@dataclass
class TriggerDecision:
    """Decision from adaptive trigger."""
    should_trigger: bool
    reason: str
    priority: float  # 0-1, higher = more urgent
    confidence: float = 0.0


@dataclass
class TrackState:
    """State tracking for adaptive VL triggering."""
    track_id: int
    first_seen: float = 0.0
    last_classified: float = 0.0
    classification_count: int = 0
    last_confidence: float = 0.0

    # Motion state
    positions: list[tuple[float, float]] = field(default_factory=list)
    velocities: list[float] = field(default_factory=list)

    # Appearance state
    appearance_change_scores: list[float] = field(default_factory=list)

    # Classification history
    classification_history: list[str] = field(default_factory=list)

    # State
    is_mature: bool = False  # Has been tracked for enough frames
    is_static: bool = True
    last_trigger_decision: Optional[TriggerDecision] = None


class AdaptiveVLTrigger:
    """Adaptive VL classification trigger.

    Makes intelligent decisions about when to trigger VL classification
    based on multiple signals.
    """

    def __init__(
        self,
        # Timing parameters
        min_interval_frames: int = 30,  # Minimum frames between classifications
        max_interval_frames: int = 150,  # Maximum frames between classifications

        # Motion parameters
        static_velocity_threshold: float = 2.0,  # Pixels/frame to consider static
        motion_change_threshold: float = 0.5,  # Velocity change to trigger

        # Appearance parameters
        appearance_change_threshold: float = 0.3,  # Trigger on significant change
        appearance_window_size: int = 5,

        # Confidence parameters
        confidence_threshold_high: float = 0.8,  # Stop classifying if confident
        confidence_threshold_low: float = 0.4,  # Need classification if low

        # Maturity parameters
        maturity_frames: int = 10,  # Frames to consider track mature

        # Priority weights
        motion_weight: float = 0.3,
        appearance_weight: float = 0.4,
        time_weight: float = 0.3,
    ) -> None:
        """Initialize adaptive VL trigger.

        Args:
            min_interval_frames: Minimum frames between VL classifications
            max_interval_frames: Maximum frames between VL classifications
            static_velocity_threshold: Velocity threshold for static detection
            motion_change_threshold: Trigger on velocity change ratio
            appearance_change_threshold: Trigger on appearance change
            appearance_window_size: Number of frames to consider for appearance
            confidence_threshold_high: Stop classifying if confidence is high
            confidence_threshold_low: Always classify if confidence is low
            maturity_frames: Frames needed for track to be mature
            motion_weight: Weight for motion signal in priority
            appearance_weight: Weight for appearance signal in priority
            time_weight: Weight for time signal in priority
        """
        # Timing
        self.min_interval_frames = min_interval_frames
        self.max_interval_frames = max_interval_frames

        # Motion
        self.static_velocity_threshold = static_velocity_threshold
        self.motion_change_threshold = motion_change_threshold

        # Appearance
        self.appearance_change_threshold = appearance_change_threshold
        self.appearance_window_size = appearance_window_size

        # Confidence
        self.confidence_threshold_high = confidence_threshold_high
        self.confidence_threshold_low = confidence_threshold_low

        # Maturity
        self.maturity_frames = maturity_frames

        # Weights
        self.motion_weight = motion_weight
        self.appearance_weight = appearance_weight
        self.time_weight = time_weight

        # Track states
        self._track_states: dict[int, TrackState] = {}

    def should_classify(
        self,
        track_id: int,
        frame_id: int,
        bbox: tuple[float, float, float, float],
        appearance_change: float = 0.0,
        current_confidence: float = 0.0,
    ) -> TriggerDecision:
        """Determine if VL classification should be triggered.

        Args:
            track_id: Track ID
            frame_id: Current frame ID
            bbox: Current bounding box (x1, y1, x2, y2)
            appearance_change: Appearance change score [0, 1]
            current_confidence: Current classification confidence

        Returns:
            TriggerDecision with recommendation
        """
        # Get or create track state
        if track_id not in self._track_states:
            self._track_states[track_id] = TrackState(
                track_id=track_id,
                first_seen=time.time(),
            )

        state = self._track_states[track_id]

        # Update motion state
        self._update_motion_state(state, bbox, frame_id)

        # Update appearance state
        if appearance_change > 0:
            state.appearance_change_scores.append(appearance_change)
            if len(state.appearance_change_scores) > self.appearance_window_size:
                state.appearance_change_scores.pop(0)

        # Check maturity
        state.is_mature = len(state.positions) >= self.maturity_frames

        # Compute priority score
        priority = self._compute_priority(state, frame_id)

        # Make decision
        decision = self._make_decision(
            state, frame_id, priority, current_confidence
        )

        state.last_trigger_decision = decision

        # Update state if classified
        if decision.should_trigger:
            state.last_classified = time.time()
            state.classification_count += 1

        return decision

    def _update_motion_state(
        self,
        state: TrackState,
        bbox: tuple[float, float, float, float],
        frame_id: int,
    ) -> None:
        """Update motion state for a track."""
        # Get center position
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # Add position
        state.positions.append((cx, cy))
        if len(state.positions) > self.maturity_frames:
            state.positions.pop(0)

        # Compute velocity if we have enough positions
        if len(state.positions) >= 2:
            prev_cx, prev_cy = state.positions[-2]
            velocity = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            state.velocities.append(velocity)
            if len(state.velocities) > 10:
                state.velocities.pop(0)

            # Determine if static
            state.is_static = velocity < self.static_velocity_threshold

    def _compute_priority(
        self,
        state: TrackState,
        frame_id: int,
    ) -> float:
        """Compute priority score for classification.

        Returns:
            Priority score in [0, 1]
        """
        # Time priority
        time_score = self._compute_time_priority(state, frame_id)

        # Motion priority
        motion_score = self._compute_motion_priority(state)

        # Appearance priority
        appearance_score = self._compute_appearance_priority(state)

        # Weighted combination
        priority = (
            time_score * self.time_weight +
            motion_score * self.motion_weight +
            appearance_score * self.appearance_weight
        )

        return min(priority, 1.0)

    def _compute_time_priority(
        self,
        state: TrackState,
        frame_id: int,
    ) -> float:
        """Compute time-based priority."""
        if state.last_classified == 0:
            return 1.0  # Never classified

        # Frames since last classification
        frames_since = frame_id - state.last_classified

        # Normalize to [0, 1]
        if frames_since >= self.max_interval_frames:
            return 1.0
        elif frames_since <= self.min_interval_frames:
            return 0.0

        return (frames_since - self.min_interval_frames) / (
            self.max_interval_frames - self.min_interval_frames
        )

    def _compute_motion_priority(self, state: TrackState) -> float:
        """Compute motion-based priority.

        Higher priority when:
        - Object starts moving after being static
        - Object changes direction
        - Object is in transition
        """
        if len(state.velocities) < 2:
            return 0.5  # Unknown state

        # Check for motion change
        recent_velocities = state.velocities[-5:]
        if len(recent_velocities) >= 2:
            # Velocity change
            vel_change = abs(recent_velocities[-1] - recent_velocities[0])

            # Was static, now moving
            was_static = all(v < self.static_velocity_threshold
                           for v in recent_velocities[:-1])
            is_moving = recent_velocities[-1] > self.static_velocity_threshold

            if was_static and is_moving:
                return 1.0  # Object just started moving

            if vel_change > self.motion_change_threshold * 10:
                return 0.8  # Significant motion change

        # Normalize velocity to priority
        avg_velocity = np.mean(state.velocities) if state.velocities else 0
        return min(avg_velocity / 20.0, 1.0)

    def _compute_appearance_priority(self, state: TrackState) -> float:
        """Compute appearance-based priority.

        Higher priority when:
        - Significant appearance change detected
        - Appearance is inconsistent
        """
        if len(state.appearance_change_scores) == 0:
            return 0.0

        # Average recent appearance change
        recent_changes = state.appearance_change_scores[-self.appearance_window_size:]
        avg_change = np.mean(recent_changes)

        # Check if exceeds threshold
        if avg_change > self.appearance_change_threshold:
            return min(avg_change, 1.0)

        return 0.0

    def _make_decision(
        self,
        state: TrackState,
        frame_id: int,
        priority: float,
        current_confidence: float,
    ) -> TriggerDecision:
        """Make final classification decision."""

        # Check if we have enough history
        if len(state.positions) < 3:
            return TriggerDecision(
                should_trigger=False,
                reason="Track too new",
                priority=0.0,
                confidence=0.0,
            )

        # Check minimum interval
        frames_since = frame_id - (state.last_classified if state.last_classified > 0 else 0)
        if frames_since < self.min_interval_frames:
            return TriggerDecision(
                should_trigger=False,
                reason=f"Too soon since last classification ({frames_since} frames)",
                priority=priority,
                confidence=1.0 - frames_since / self.min_interval_frames,
            )

        # Check confidence
        if current_confidence >= self.confidence_threshold_high:
            return TriggerDecision(
                should_trigger=False,
                reason=f"High confidence ({current_confidence:.2f})",
                priority=0.0,
                confidence=current_confidence,
            )

        if current_confidence < self.confidence_threshold_low:
            return TriggerDecision(
                should_trigger=True,
                reason=f"Low confidence ({current_confidence:.2f})",
                priority=1.0,
                confidence=current_confidence,
            )

        # Priority-based decision
        should_trigger = priority > 0.5

        if should_trigger:
            reason = f"High priority ({priority:.2f})"
        else:
            reason = f"Low priority ({priority:.2f})"

        return TriggerDecision(
            should_trigger=should_trigger,
            reason=reason,
            priority=priority,
            confidence=0.5,
        )

    def update_classification_result(
        self,
        track_id: int,
        class_name: str,
        confidence: float,
    ) -> None:
        """Update state after VL classification."""
        if track_id not in self._track_states:
            return

        state = self._track_states[track_id]
        state.last_confidence = confidence
        state.classification_history.append(class_name)
        state.last_classified = time.time()

        # Keep history bounded
        if len(state.classification_history) > 10:
            state.classification_history.pop(0)

    def remove_track(self, track_id: int) -> None:
        """Remove track state."""
        self._track_states.pop(track_id, None)

    def cleanup(self, active_track_ids: set[int]) -> None:
        """Remove stale track states."""
        stale_ids = set(self._track_states.keys()) - active_track_ids
        for track_id in stale_ids:
            self.remove_track(track_id)

    def get_track_info(self, track_id: int) -> Optional[TrackState]:
        """Get track state information."""
        return self._track_states.get(track_id)

from .base_tracker import BaseTracker
from collections import deque
import numpy as np

class BallTracker(BaseTracker):
    def __init__(self, model_path, confidence_threshold=0.15):
        super().__init__(model_path, confidence_threshold, frame_interval=30)
        self.ball_buffer = {}
        self.frame_buffer = deque(maxlen=30)  # Store last 30 frames for interpolation

    def detect_and_track(self, frame, frame_idx):
        """Detect and track ball in the frame."""
        # Get detections from base tracker
        ball_tracks = super().detect_and_track(frame, frame_idx)
        
        # Sort by confidence and keep highest confidence ball
        if ball_tracks:
            ball_tracks = sorted(ball_tracks, key=lambda x: x[1], reverse=True)[:1]  # Sort by confidence (index 1)
            
        return ball_tracks

    def delayed_ball_interpolator(self, ball_buffer, target_idx, delay):
        """Interpolate ball position for a delayed frame."""
        # Find the closest past and future ball positions
        past = next(((i, b) for i, b in reversed(list(ball_buffer)) if i < target_idx and b is not None), None)
        future = next(((i, b) for i, b in ball_buffer if i > target_idx and b is not None), None)
        
        if past is None or future is None:
            return None
            
        past_idx, past_bbox = past
        future_idx, future_bbox = future
        
        # Calculate interpolation factor
        total_frames = future_idx - past_idx
        if total_frames == 0:
            return None
            
        factor = (target_idx - past_idx) / total_frames
        
        # Interpolate bbox
        interp_bbox = [
            past_bbox[0] + (future_bbox[0] - past_bbox[0]) * factor,
            past_bbox[1] + (future_bbox[1] - past_bbox[1]) * factor,
            past_bbox[2] + (future_bbox[2] - past_bbox[2]) * factor,
            past_bbox[3] + (future_bbox[3] - past_bbox[3]) * factor
        ]
        
        return {
            "bbox": interp_bbox,
            "confidence": 1.0,  # Interpolated position gets full confidence
            "class_id": 0,  # Ball class
            "track_id": -1  # Special track ID for interpolated positions
        }

    def get_ball_position(self, frame_idx):
        """Get the ball position for a specific frame."""
        return self.ball_buffer.get(frame_idx) 
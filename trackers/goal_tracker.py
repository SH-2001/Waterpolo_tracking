from .base_tracker import BaseTracker
import numpy as np

class GoalTracker(BaseTracker):
    def __init__(self, model_path, confidence_threshold=0.15):
        super().__init__(model_path, confidence_threshold, frame_interval=300)
        self.static_goals = None
        self.goal_history = {}  # track_id -> list of (frame_idx, bbox)
        self.movement_threshold = 50  # pixels
        self.min_detections_for_movement = 3  # number of consecutive detections needed to confirm movement
        self.goals_initialized = False

    def detect_and_track(self, frame, frame_idx):
        detections = super().detect_and_track(frame, frame_idx)
        
        # If we haven't initialized goals yet, try to detect them
        if not self.goals_initialized:
            if len(detections) >= 2:  # We found at least 2 goals
                current_goals = []
                for det in detections:
                    x1, y1, x2, y2 = det[0]
                    current_goals.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": det[1],
                        "class_id": int(det[2]),
                        "track_id": int(det[4])
                    })
                self.static_goals = current_goals
                self.goals_initialized = True
                print(f"Goals initialized at frame {frame_idx}")
            return detections

        # Only process goals every 300th frame after initialization
        if frame_idx % 300 == 0 and self.goals_initialized:
            if detections and self.static_goals:
                for det in detections:
                    track_id = int(det[4])
                    current_bbox = det[0]
                    
                    # Initialize history for new tracks
                    if track_id not in self.goal_history:
                        self.goal_history[track_id] = []
                    
                    # Add current detection to history
                    self.goal_history[track_id].append((frame_idx, current_bbox))
                    
                    # Keep only recent history
                    self.goal_history[track_id] = self.goal_history[track_id][-self.min_detections_for_movement:]
                    
                    # Check for movement if we have enough history
                    if len(self.goal_history[track_id]) >= self.min_detections_for_movement:
                        if self._detect_movement(track_id):
                            # Update static goals with new position
                            self._update_static_goals(track_id, current_bbox)
                            print(f"Goal {track_id} movement detected at frame {frame_idx}")
        
        # After initialization, always return the static goals
        if self.goals_initialized:
            static_detections = []
            for goal in self.static_goals:
                static_detections.append((
                    goal["bbox"],
                    goal["confidence"],
                    goal["class_id"],
                    "goal",
                    goal["track_id"]
                ))
            return static_detections
        
        return detections

    def _detect_movement(self, track_id):
        """Detect if a goal has moved significantly."""
        if track_id not in self.goal_history:
            return False
            
        history = self.goal_history[track_id]
        if len(history) < self.min_detections_for_movement:
            return False
            
        # Get the original position from static goals
        original_goal = next((g for g in self.static_goals if g["track_id"] == track_id), None)
        if not original_goal:
            return False
            
        # Calculate center points
        orig_bbox = original_goal["bbox"]
        orig_center = np.array([
            (orig_bbox[0] + orig_bbox[2]) / 2,
            (orig_bbox[1] + orig_bbox[3]) / 2
        ])
        
        # Get current position
        current_bbox = history[-1][1]
        current_center = np.array([
            (current_bbox[0] + current_bbox[2]) / 2,
            (current_bbox[1] + current_bbox[3]) / 2
        ])
        
        # Calculate movement distance
        movement = np.linalg.norm(current_center - orig_center)
        
        return movement > self.movement_threshold

    def _update_static_goals(self, track_id, new_bbox):
        """Update the static goals with new position."""
        for goal in self.static_goals:
            if goal.get("track_id") == track_id:
                if not np.allclose(goal["bbox"], new_bbox, atol=5):
                    goal["bbox"] = new_bbox.tolist()
            break

    def get_static_goals(self):
        """Get the static goal positions."""
        return self.static_goals if self.static_goals is not None else [] 
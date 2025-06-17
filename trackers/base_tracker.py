import supervision as sv
from ultralytics import YOLO
import numpy as np


class BaseTracker:
    def __init__(self, model_path, confidence_threshold=0.20, frame_interval=5):
        self.model = YOLO(model_path)
        # Initialize ByteTracker with default parameters
        self.tracker = sv.ByteTrack()
        self.annotator = sv.BoxAnnotator()
        self.confidence_threshold = confidence_threshold
        self.frame_interval = frame_interval
        self.track_memory = {}  # track_id -> {xyxy, class_id, confidence, class_name, last_seen, velocity}
        self.id_to_label = {}  # track_id -> class_name
        self.max_missed_frames = 30
        self.last_processed_frame = -1
        self.last_positions = {}  # For velocity calculation

    def should_process_frame(self, frame_idx):
        """Check if this frame should be processed based on frame interval."""
        return frame_idx % self.frame_interval == 0

    def predict_next_position(self, track_id, current_xyxy):
        """Predict next position based on velocity."""
        if track_id in self.track_memory:
            info = self.track_memory[track_id]
            if "velocity" in info:
                # Apply velocity to current position
                vx, vy = info["velocity"]
                x1, y1, x2, y2 = current_xyxy
                width = x2 - x1
                height = y2 - y1
                
                # Predict next position
                next_x1 = x1 + vx
                next_y1 = y1 + vy
                next_x2 = next_x1 + width
                next_y2 = next_y1 + height
                
                return [next_x1, next_y1, next_x2, next_y2]
        return current_xyxy

    def calculate_velocity(self, track_id, current_xyxy, frame_idx):
        """Calculate smoothed velocity based on last known position."""
        if track_id in self.last_positions:
            last_xyxy, last_frame = self.last_positions[track_id]
            frames_diff = frame_idx - last_frame
            if frames_diff > 0:
                dx = (current_xyxy[0] - last_xyxy[0]) / frames_diff
                dy = (current_xyxy[1] - last_xyxy[1]) / frames_diff

                # Clamp velocity to avoid large jumps
                max_velocity = 50  # pixels per frame
                dx = np.clip(dx, -max_velocity, max_velocity)
                dy = np.clip(dy, -max_velocity, max_velocity)

                # Smooth with exponential moving average
                if track_id in self.track_memory and "velocity" in self.track_memory[track_id]:
                    old_vx, old_vy = self.track_memory[track_id]["velocity"]
                    alpha = 0.3  # smoothing factor
                    dx = (1 - alpha) * old_vx + alpha * dx
                    dy = (1 - alpha) * old_vy + alpha * dy

                velocity = (dx, dy)
                self.last_positions[track_id] = (current_xyxy, frame_idx)
                return velocity

        # No prior velocity — initialize
        self.last_positions[track_id] = (current_xyxy, frame_idx)
        return (0, 0)

    def detect_and_track(self, frame, frame_idx):
        """Detect and track objects in the frame."""
        detections = []
        if not self.should_process_frame(frame_idx):
            # For frames we don't process, return predicted positions
            if self.track_memory:
                for tid, info in self.track_memory.items():
                    # Predict next position based on velocity
                    predicted_xyxy = self.predict_next_position(tid, info["xyxy"])
                    detections.append((
                        predicted_xyxy,
                        info["confidence"],
                        info["class_id"],
                        info["class_name"],
                        tid
                    ))
                return detections
            return []

        self.last_processed_frame = frame_idx
        results = self.model.predict(frame, conf=self.confidence_threshold)
        result = results[0]
        boxes = result.boxes

        # No detections
        if len(boxes) == 0:
            print(f"No detections in frame {frame_idx}")
            if self.track_memory:
                for tid, info in self.track_memory.items():
                    predicted_xyxy = self.predict_next_position(tid, info["xyxy"])
                    detections.append((
                        predicted_xyxy,
                        info["confidence"],
                        info["class_id"],
                        info["class_name"],
                        tid
                    ))
                return detections
            return []
    
        det_sv = sv.Detections.from_ultralytics(result)
        detection_with_tracks = self.tracker.update_with_detections(det_sv)
        
        # Update track memory with new detections
        current_tracks = set()
        for det in detection_with_tracks:
            xyxy = det[0]
            confidence = float(det[2])
            class_id = int(det[3])
            track_id = int(det[4])
            class_name = self.model.names[class_id]

            # Calculate velocity
            velocity = self.calculate_velocity(track_id, xyxy, frame_idx)

            # Store in track memory
            self.track_memory[track_id] = {
                "xyxy": xyxy,
                "class_id": class_id,
                "confidence": confidence,
                "class_name": class_name,
                "last_seen": frame_idx,
                "velocity": velocity
            }
            self.id_to_label[track_id] = class_name
            current_tracks.add(track_id)

        # Update last_seen for existing tracks and remove old ones
        to_delete = []
        for tid, info in self.track_memory.items():
            if tid not in current_tracks:
                if frame_idx - info["last_seen"] > self.max_missed_frames:
                    to_delete.append(tid)
                    if tid in self.last_positions:
                        del self.last_positions[tid]

        for tid in to_delete:
            del self.track_memory[tid]
            if tid in self.id_to_label:
                del self.id_to_label[tid]

        return detection_with_tracks

    def get_last_detections(self):
        """Return the last known detections for frames that shouldn't be processed."""
        # If no track memory exists yet, return empty list
        if not self.track_memory:
            return []
            
        # Convert track memory to detections format
        detections = []
        for tid, info in self.track_memory.items():
            if tid in self.id_to_label:
                detections.append((
                    info["xyxy"],
                    info["confidence"],  
                    info["class_id"],
                    info["class_name"],
                    tid
                ))
        return detections

    def get_track_info(self, track_id):
        """Get information about a specific track."""
        if track_id in self.track_memory:
            return self.track_memory[track_id]
        return None 
from .base_tracker import BaseTracker
from sklearn.cluster import KMeans
import cv2
import numpy as np

class GoalkeeperTracker(BaseTracker):
    def __init__(self, model_path, confidence_threshold=0.15):
        super().__init__(model_path, confidence_threshold, frame_interval=5)
        self.id_to_team = {}
        self.id_to_hist = {}  # track_id -> color histogram (used for re-ID)
        self.track_id_to_side = {}  # track_id -> "left" or "right"
        self.team_hist_refs = {"blue": None, "white": None}
        self.last_detections = None

    def detect_and_track(self, frame, frame_idx):
        goalkeeper_tracks = super().detect_and_track(frame, frame_idx)

        self.last_frame = frame.copy()

        # Prune stale tracks
        self.prune_tracks(frame_idx)
            
        # Sort by confidence (index 2) and keep top 2
        # Each detection is a tuple: (xyxy, confidence, class_id, track_id)
        goalkeeper_tracks = sorted(goalkeeper_tracks, key=lambda x: x[1] if x[1] is not None else 0, reverse=True)[:2]

        # If 2 goalkeepers detected, assign left/right
        if len(goalkeeper_tracks) == 2:
            # Sort by x-center to assign left/right
            def center_x(det): return (det[0][0] + det[0][2]) / 2
            sorted_by_x = sorted(goalkeeper_tracks, key=center_x)
            self.track_id_to_side[int(sorted_by_x[0][4])] = "left"
            self.track_id_to_side[int(sorted_by_x[1][4])] = "right"
                
        # Store for next fallback
        self.last_detections = goalkeeper_tracks[:2]

        # If only one keeper is visible, attempt to reconstruct the missing one
        if len(goalkeeper_tracks) == 1:
            visible_track = goalkeeper_tracks[0]
            visible_id = int(visible_track[4])
            visible_side = self.track_id_to_side.get(visible_id)

            # Try to find the missing side
            if visible_side == "left":
                missing_side = "right"
            elif visible_side == "right":
                missing_side = "left"
            else:
                missing_side = None  # Not enough info yet

            # Try to find a track ID for the missing side
            if missing_side:
                for track_id, side in self.track_id_to_side.items():
                    if side == missing_side and track_id in self.track_memory:
                        info = self.track_memory[track_id]
                        # Predict next location using velocity
                        predicted_xyxy = self.predict_next_position(track_id, info["xyxy"])
                        confidence = info["confidence"]
                        class_id = info["class_id"]
                        class_name = info["class_name"]

                        reconstructed_det = (predicted_xyxy, confidence, class_id, class_name, track_id)
                        goalkeeper_tracks.append(reconstructed_det)
                        break

        # Only return 2 at most
        goalkeeper_tracks = goalkeeper_tracks[:2]
                
        # Assign teams to detected goalkeepers
        matched_old_ids = set()
        updated_tracks = []

        for det in goalkeeper_tracks:
            new_id = int(det[4])
            matched_id, hist = self.try_reid_goalkeeper(det, matched_old_ids)

            if matched_id is not None and matched_id != new_id:
                # Replace new ID with matched old ID
                print(f"[ID Merge] Replacing new ID {new_id} with old ID {matched_id}")
                det = (
                    det[0], det[1], det[2], det[3], matched_id
                )
                track_id = matched_id
            else:
                track_id = new_id

            # Update histogram and team info
            self.id_to_hist[track_id] = hist
            self.assign_team_with_reid(frame, det)

            # Update track memory
            self.track_memory[track_id] = {
                "xyxy": det[0],
                "confidence": det[1],
                "class_id": det[2],
                "class_name": det[3],
                "last_seen": frame_idx,
            }

            updated_tracks.append(det)
            
        return goalkeeper_tracks

    def assign_team_with_reid(self, frame, det):
        track_id = int(det[4])
        x1, y1, x2, y2 = map(int, det[0])
        center_x = (x1 + x2) // 2
        width = frame.shape[1]

        # Extract HSV histogram from cap area
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist = cv2.normalize(hist, hist).flatten()

        self.id_to_hist[track_id] = hist

        # If no references yet, use position to bootstrap
        if self.team_hist_refs["blue"] is None or self.team_hist_refs["white"] is None:
            if center_x < width // 2:
                self.id_to_team[track_id] = "blue"
                self.team_hist_refs["blue"] = hist
            else:
                self.id_to_team[track_id] = "white"
                self.team_hist_refs["white"] = hist
            return

        # Compare to both team reference histograms
        blue_dist = cv2.compareHist(hist.astype(np.float32), self.team_hist_refs["blue"].astype(np.float32), cv2.HISTCMP_CORREL)
        white_dist = cv2.compareHist(hist.astype(np.float32), self.team_hist_refs["white"].astype(np.float32), cv2.HISTCMP_CORREL)

        # Choose the closest match (higher correlation = more similar)
        new_team = "blue" if blue_dist > white_dist else "white"
        new_conf = max(blue_dist, white_dist)

        # Reassign only if new team is more confident (as you requested)
        old_team = self.id_to_team.get(track_id)
        if old_team != new_team:
            if old_team is None or new_conf > 0.9:
                self.id_to_team[track_id] = new_team

    def try_reid_goalkeeper(self, new_det, matched_old_ids):
        """Try to re-identify a newly detected goalkeeper using histogram similarity."""
        x1, y1, x2, y2 = map(int, new_det[0])
        new_track_id = new_det[4]
        roi = self.last_frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist = cv2.normalize(hist, hist).flatten()

        best_match = None
        best_score = -1

        for old_id, old_hist in self.id_to_hist.items():
            if old_id in matched_old_ids:
                continue
            score = cv2.compareHist(hist.astype(np.float32), old_hist.astype(np.float32), cv2.HISTCMP_CORREL)
            if score > best_score:
                best_score = score
                best_match = old_id

        if best_score > 0.95 and best_match in self.id_to_team:
            matched_old_ids.add(best_match)
            print(f"[Re-ID] Matched new track {new_track_id} to old track {best_match} (score={best_score:.2f})")
            return best_match, hist
        else:
            return None, hist

    def prune_tracks(self, current_frame_idx, max_age=30):
        to_delete = []
        for track_id, info in self.track_memory.items():
            last_seen = info.get("last_seen", current_frame_idx)
            if current_frame_idx - last_seen > max_age:
                to_delete.append(track_id)

        for track_id in to_delete:
            print(f"[Prune] Removing stale track {track_id}")  
            self.track_memory.pop(track_id, None)
            self.id_to_team.pop(track_id, None)
            self.id_to_hist.pop(track_id, None)
            self.track_id_to_side.pop(track_id, None)
            
    def get_team(self, track_id):
        """Get the team assignment for a specific track."""
        return self.id_to_team.get(track_id, "unknown") 
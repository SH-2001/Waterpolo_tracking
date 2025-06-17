from .base_tracker import BaseTracker
from sklearn.cluster import KMeans
import cv2
import numpy as np

class PlayerTracker(BaseTracker):
    def __init__(self, model_path, confidence_threshold=0.15):
        super().__init__(model_path, confidence_threshold, frame_interval=30)
        self.id_to_team = {}
        self.cap_features = []  # List of (track_id, feature)
        self.cap_feature_map = {}  # track_id -> feature
        self.kmeans = None
        self.cluster_ready = False
        self.min_samples_for_clustering = 20
        self.team_confidence = {}
        self.cluster_id_to_name = {
            0: "dark",
            1: "white"
        }

    def detect_and_track(self, frame, frame_idx):
        detections = super().detect_and_track(frame, frame_idx)
        
        # Sort by confidence and keep top 12 players
        player_tracks = [(det[2], det) for det in detections]  # (confidence, detection)
        player_tracks = sorted(player_tracks, key=lambda x: x[0], reverse=True)[:12]
        
        # Assign teams to detected players
        for _, det in player_tracks:
            self.assign_team(frame, det)
            
        return [d for _, d in player_tracks]

    def assign_team(self, frame, det):
        tid = int(det[4])
        box = det[0]

        # flatten box if needed
        if isinstance(box, np.ndarray) and box.ndim > 1:
            box = box.flatten()
        x1, y1, x2, y2 = map(int, box)
        h = y2 - y1

        # Define cap region: top 10% to 35% of the bounding box height
        cap_top = y1 + int(h * 0.10)
        cap_bottom = y1 + int(h * 0.35)
        cap_crop = frame[cap_top:cap_bottom, x1:x2]

        if cap_crop.size == 0:
            return self.id_to_team.get(tid, "unknown")

        # Extract feature
        hsv = cv2.cvtColor(cap_crop, cv2.COLOR_BGR2HSV)
        mean_hsv = hsv.mean(axis=(0, 1))
        self.cap_feature_map[tid] = mean_hsv

        # Already clustered?
        if self.cluster_ready:
            label = self.kmeans.predict([mean_hsv])[0]
            cluster_center = self.kmeans.cluster_centers_[label]
            dist = np.linalg.norm(mean_hsv - cluster_center)  # Smaller = better match
            confidence = -dist  # Higher = better (invert distance)

            if tid not in self.team_confidence or confidence > self.team_confidence[tid]:
                self.id_to_team[tid] = self.cluster_id_to_name[label]
                self.team_confidence[tid] = confidence
                return self.id_to_team[tid]

        # Collect samples
        if tid not in self.cap_feature_map:
            self.cap_features.append((tid, mean_hsv))
            self.cap_feature_map[tid] = mean_hsv

        if len(self.cap_features) >= self.min_samples_for_clustering:
            # Run clustering once
            features = [f for _, f in self.cap_features]
            self.kmeans = KMeans(n_clusters=2, random_state=42)
            self.kmeans.fit(features)
            self.cluster_ready = True
            print("[INFO] KMeans color clustering completed")

            for tid_i, feat in self.cap_features:
                label = self.kmeans.predict([feat])[0]
                center = self.kmeans.cluster_centers_[label]
                dist = np.linalg.norm(feat - center)
                self.id_to_team[tid_i] = self.cluster_id_to_name[label]
                self.team_confidence[tid_i] = -dist
        else:
            # TEMPORARY fallback: HSV threshold
            avg_hue = mean_hsv[0]
            if avg_hue < 30 or avg_hue > 160:  # likely white
                self.id_to_team[tid] = "white"
                self.team_confidence[tid] = 0  # Low confidence
            else:
                self.id_to_team[tid] = "dark"
                self.team_confidence[tid] = 0  # Low confidence

    def get_team(self, track_id):
        """Get the team assignment for a specific track."""
        return self.id_to_team.get(track_id, "unknown") 
import supervision as sv
from ultralytics import YOLO
import numpy as np

model = YOLO("models/player/best_player_v4.pt")
source_path="D:/Sander/Waterpolo_Tracking/videos/raw_video/Hasselt-Antwerpen/Q4.mp4"
tracker = sv.ByteTrack()

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

video_info = sv.VideoInfo.from_video_path(source_path)
fps = video_info.fps
duration_seconds = 70
max_frame = int(fps * duration_seconds)

def callback(frame: np.ndarray, index: int) -> np.ndarray:
    if index >= max_frame:
        raise StopIteration
    
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

    annotated_frame = bounding_box_annotator.annotate(
        scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame



sv.process_video(
    source_path=source_path,
    target_path="output/tracker_test_player.mp4",
    callback=callback
)
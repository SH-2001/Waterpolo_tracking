import cv2
import os
import json
from trackers import PlayerTracker, GoalkeeperTracker, BallTracker, GoalTracker
import numpy as np
import time


def read_save_video(input_video, output_video, timeline_path):
    """Read video, process frames, and save output video with annotations."""
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_video}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate number of frames for x minutes
    x = 1
    frames_xmin = int(x * 60 * fps)
    print(f"Processing first {x} minutes ({frames_xmin} frames) of video...")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Initialize trackers with their respective models
    player_tracker = PlayerTracker("models/player/best_player_v3.pt")
    goalkeeper_tracker = GoalkeeperTracker("models/goalkeeper/best_goalkeeper_v3.pt")
    ball_tracker = BallTracker("models/ball/best_ball_v3.pt")
    goal_tracker = GoalTracker("models/goal/best_goal_v2.pt")

    frame_count = 0

    while cap.isOpened() and frame_count < frames_xmin:
        ret, frame = cap.read()
        if not ret:
            break

        # Get detections from each tracker
        player_tracks = player_tracker.detect_and_track(frame, frame_count)
        goalkeeper_tracks = goalkeeper_tracker.detect_and_track(frame, frame_count)
        frame = goalkeeper_tracker.draw_debug_info(frame, goalkeeper_tracks)
        ball_tracks = ball_tracker.detect_and_track(frame, frame_count)
        goal_tracks = goal_tracker.detect_and_track(frame, frame_count)

        # Convert all tracks to list format for consistent handling
        all_tracks = []
        if isinstance(player_tracks, list):
            all_tracks.extend([(det, "player") for det in player_tracks])
        if isinstance(goalkeeper_tracks, list):
            all_tracks.extend([(det, "goalkeeper") for det in goalkeeper_tracks])
        if isinstance(ball_tracks, list):
            all_tracks.extend([(det, "ball") for det in ball_tracks])
        if isinstance(goal_tracks, list):
           all_tracks.extend([(det, "goal") for det in goal_tracks])

        # Annotate frame with all tracks and goals
        annotated_frame = annotate(frame, [det[0] for det in all_tracks], goal_tracks, player_tracks, goalkeeper_tracker) 

        # Save frame
        out.write(annotated_frame)    

        # Create timeline entry
        timeline_entry = []
        
        # Add detections to timeline
        for det, class_name in all_tracks:
            if len(det) >= 5:  # Ensure we have all required elements
                # Convert all values to proper types
                bbox = det[0].tolist() if isinstance(det[0], np.ndarray) else det[0]
                confidence = float(det[1]) if det[1] is not None else 0.0
                track_id = int(det[4]) if det[4] is not None else None

                detection_data = {
                    "frame": int(frame_count),
                    "timestamp": float(frame_count / fps),
                    "bbox": bbox,
                    "confidence": confidence,
                    "class_name": class_name,
                    "track_id": track_id
                }
                timeline_entry.append(detection_data)

        # Save timeline entry
        append_to_json(timeline_entry, timeline_path)
        
        frame_count += 1

        # Print progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    # Release resources
    cap.release()
    out.release()

    print(f"Video processing complete. Output saved to {output_video}")
    print(f"Timeline data saved to {timeline_path}")


def append_to_json(data, file_path):
    """Append data to a JSON file, creating it if it doesn't exist."""
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # If file doesn't exist, create it with empty list
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump([], f)

    try:
        # Read existing data
        with open(file_path, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []

        # Append new data
        if isinstance(data, list):
            existing_data.extend(data)
        else:
            existing_data.append(data)

        serializable_data = convert_to_python_types(existing_data)

        # Write back to file
        with open(file_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)

    except PermissionError:
        print(f"Warning: Could not write to {file_path}. File may be in use.")
        # Create a backup file with timestamp
        backup_path = f"{file_path}.backup_{int(time.time())}"
        try:
            with open(backup_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Data saved to backup file: {backup_path}")
        except Exception as e:
            print(f"Failed to create backup file: {e}")
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")


def convert_to_python_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    

def annotate(frame, tracks, goals, player_tracker, goalkeeper_tracker):
    annotated = frame.copy()
    
    for det in tracks:
        x1, y1, x2, y2 = map(int, det[0])
        track_id = int(det[4])
        
        # Determine if it's a player, goalkeeper, or ball
        if track_id in player_tracker.id_to_team:
            team = player_tracker.get_team(track_id)
            if team == "white":
                color = (255, 255, 255)
            else:  # dark
                color = (255, 0, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"ID:{track_id} Player"
            
        elif track_id in goalkeeper_tracker.id_to_team:
            team = goalkeeper_tracker.get_team(track_id)
            if team == "white":
                color = (255, 255, 255)
            else:  # dark
                color = (255, 0, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"ID:{track_id} GK"
            
        else:  # ball
            color = (0, 255, 255)
            cv2.ellipse(annotated, center=(int((x1 + x2) / 2), y2),
                       axes=(int(x2 - x1), int(0.35 * (x2 - x1))), angle=0.0,
                       startAngle=45, endAngle=235, color=color, thickness=2)
            text = f"ID:{track_id} Ball"
            
        cv2.putText(annotated, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Annotate goals
    for goal in goals:
        x1, y1, x2, y2 = map(int, goal[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 0), 1)
        #cv2.putText(annotated, "Goal", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return annotated
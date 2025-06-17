import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time

def analyze_video(video_path, model_path, frame_interval=60, confidence_threshold=0.25):
    # Check if video file exists
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")
    
    # Initialize model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"Video file has no frames: {video_path}")
    
    print(f"Processing video with {total_frames} total frames")
    
    # Initialize analysis variables
    frame_indices = []
    detection_counts = []
    confidences = []
    
    # Process only every nth frame
    for frame_idx in range(0, total_frames, frame_interval):
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}")
            continue
            
        # Process frame
        results = model(frame, conf=confidence_threshold)[0]
        detections = results.boxes.data.cpu().numpy()
        
        # Record analysis data
        frame_indices.append(frame_idx)
        detection_counts.append(len(detections))
        if len(detections) > 0:
            confidences.append(float(detections[:, 4].max()))  # Max confidence in frame
        else:
            confidences.append(0)
    
    # Release video capture
    cap.release()
    
    if not frame_indices:
        raise ValueError("No frames were successfully processed. Check video file and model.")
    
    return frame_indices, detection_counts, confidences

def plot_analysis(frame_indices, detection_counts, confidences, confidence_threshold=0.25):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Calculate percentage of frames with exactly 2 detection
    frames_with_two = sum(1 for count in detection_counts if count == 2)
    percentage_two = (frames_with_two / len(detection_counts)) * 100
    
    # Plot detection counts
    ax1.plot(frame_indices, detection_counts, 'b-', label='Number of Detections')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Number of Detections')
    ax1.set_title(f'Detection Counts Over Time (Exactly 2 detections: {percentage_two:.1f}%)')
    ax1.grid(True)
    
    # Calculate moving average for confidence (window size of 5)
    window_size = 5
    moving_avg = np.convolve(confidences, np.ones(window_size)/window_size, mode='valid')
    moving_avg_indices = frame_indices[window_size-1:]
    
    # Plot confidences with moving average
    ax2.plot(frame_indices, confidences, 'r-', alpha=0.3, label='Raw Confidence')
    ax2.plot(moving_avg_indices, moving_avg, 'b-', label='Moving Average')
    ax2.axhline(y=confidence_threshold, color='g', linestyle='--', label='Confidence Threshold')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Confidence')
    ax2.set_title('Detection Confidence Over Time')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    total_frames = len(frame_indices)
    frames_with_detections = sum(1 for count in detection_counts if count > 0)
    avg_confidence = np.mean([c for c in confidences if c > 0]) if any(c > 0 for c in confidences) else 0
    
    print(f"\nAnalysis Statistics:")
    print(f"Total frames analyzed: {total_frames}")
    print(f"Frames with detections: {frames_with_detections}")
    print(f"Detection rate: {frames_with_detections/total_frames:.2%}")
    print(f"Frames with exactly 2 detections: {frames_with_two} ({percentage_two:.1f}%)")
    print(f"Average confidence: {avg_confidence:.2f}")
    print(f"Max detections in a frame: {max(detection_counts)}")

def main():
    # Configuration
    video_path = "D:/Sander/Waterpolo_Tracking/videos/raw_video/Hasselt-Antwerpen/Q4.mp4"
    model_path = "models/goalkeeper/best_goalkeeper_v5.pt"
    frame_interval = 60
    confidence_threshold = 0.25
    
    try:
        # Run analysis with timing
        print("Starting video analysis...")
        start_time = time.time()
        
        frame_indices, detection_counts, confidences = analyze_video(
            video_path=video_path,
            model_path=model_path,
            frame_interval=frame_interval,
            confidence_threshold=confidence_threshold
        )
        
        analysis_time = time.time() - start_time
        print(f"\nAnalysis completed in {analysis_time:.2f} seconds")
        print(f"Average time per analyzed frame: {analysis_time/len(frame_indices):.3f} seconds")
        
        # Plot results
        plot_analysis(frame_indices, detection_counts, confidences, confidence_threshold)
        print("\nAnalysis complete!")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please check if the video file exists at the specified path.")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
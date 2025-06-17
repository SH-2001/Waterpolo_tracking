from utils.video_utils import read_save_video
import time

def main():
    # Set your video paths here
    input_video = "D:/Sander/Waterpolo_Tracking/videos/raw_video/Hasselt-Antwerpen/Q4.mp4" 
    output_video = "output/new_tracker/output.mp4"  
    timeline_path = "output/new_tracker/timeline.json"  
    
    # Process the video
    start_time = time.time()
    read_save_video(input_video, output_video, timeline_path)
    end_time = time.time()
    
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"Output video saved to: {output_video}")
    if timeline_path:
        print(f"Timeline saved to: {timeline_path}")

if __name__ == "__main__":
    main()
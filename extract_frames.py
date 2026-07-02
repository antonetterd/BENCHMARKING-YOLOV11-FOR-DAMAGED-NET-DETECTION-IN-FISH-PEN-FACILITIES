#FRAME EXTRACTION FROM MP4 TO JPG

import cv2
import os

os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "8192"
VIDEO_PATH ="clean_video.mp4"
OUTPUT_FOLDER = r"C:\Users\ANTONETTE\Videos\TEST_VIDE0\clean"
TARGET_FPS = 10  # Number of frames to extract per second 

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Could not open video file {VIDEO_PATH}")
    exit(1)

original_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video Info: {original_fps:.2f} FPS, {total_frames} total frames")
frame_interval = max(1, int(original_fps / TARGET_FPS))
print(f"Extracting 1 frame every {frame_interval} frames (target {TARGET_FPS} FPS)")


padding_length = len(str(total_frames))

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  

    if frame_count % frame_interval == 0:
        filename = os.path.join(OUTPUT_FOLDER, f"frame_{saved_count:0{padding_length}d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        saved_count += 1

    frame_count += 1

cap.release()
print(f"\nDONE! Extracted {saved_count} frames to '{OUTPUT_FOLDER}'")

# This is only applicable in Google Colab. Running this in other environment may cause error.
# This script is to run a detection test on image or videos of underwater fish pen nets and produce and output video or image with bounding box.

!pip install -q ultralytics pandas opencv-python

from google.colab import drive
drive.mount("/content/drive")

from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import time
import shutil
import os



SOURCE_PATH = "/content/drive/MyDrive/thesis_training2/raw_test_inputs/v (83).MP4"
#SOURCE_PATH = "/content/drive/MyDrive/thesis_training2/raw_test_inputs/4.jpg"

#MODEL_PATH = "/content/drive/MyDrive/thesis_training2/final_benchmark_clean_v5/training_results/yolo11s_training/weights/best.pt"
#MODEL_PATH = "/content/drive/MyDrive/thesis_training2/final_benchmark_clean_v5/lightweight_quantized_results_standard/02_exported_models/YOLOv11s/YOLOv11s_fp16.engine"
MODEL_PATH = "/content/drive/MyDrive/thesis_training2/final_benchmark_clean_v5/lightweight_quantized_results_standard/02_exported_models/YOLOv11m/YOLOv11m_int8.engine"

RESULTS_ROOT = "/content/drive/MyDrive/thesis_training2/final_benchmark_clean_v5/Video_Image_Test_Result"


IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


PRINT_EVERY_FRAME = True
PRINT_EVERY_N_FRAMES = 1


SAVE_SAMPLE_FRAMES = True
MAX_DETECTED_SAMPLE_FRAMES = 30
MAX_NO_DEFECT_SAMPLE_FRAMES = 10

results_root = Path(RESULTS_ROOT)
results_root.mkdir(parents=True, exist_ok=True)

def create_next_test_folder(root):
    existing = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("Test_"):
            try:
                existing.append(int(p.name.split("_")[1]))
            except:
                pass

    next_num = max(existing) + 1 if existing else 1
    test_dir = root / f"Test_{next_num:03d}"
    test_dir.mkdir(parents=True, exist_ok=False)
    return test_dir

TEST_DIR = create_next_test_folder(results_root)

ANNOTATED_OUTPUT_DIR = TEST_DIR / "annotated_outputs"
ANNOTATED_VIDEO_DIR = TEST_DIR / "annotated_video_with_boxes"
DETECTED_SAMPLE_DIR = TEST_DIR / "sample_detected_frames"
NO_DEFECT_SAMPLE_DIR = TEST_DIR / "sample_no_defect_frames"

ANNOTATED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
DETECTED_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
NO_DEFECT_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATED_VIDEO_PATH = ANNOTATED_VIDEO_DIR / "annotated_video_defect.mp4"

print("New test folder created:")
print(TEST_DIR)


source_path = Path(SOURCE_PATH.strip())
model_path = Path(MODEL_PATH.strip())

if not source_path.exists():
    raise FileNotFoundError(f"Source file/folder not found: {source_path}")

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

print("\nSource found:")
print(source_path)

print("\nModel found:")
print(model_path)

video_exts = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]
image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

if source_path.is_dir():
    SOURCE_TYPE = "image_folder"
elif source_path.suffix.lower() in video_exts:
    SOURCE_TYPE = "video"
elif source_path.suffix.lower() in image_exts:
    SOURCE_TYPE = "image"
else:
    SOURCE_TYPE = "unknown"

print("\nSource type:", SOURCE_TYPE)

video_fps = None
video_frame_count = None
video_duration = None

if SOURCE_TYPE == "video":
    cap = cv2.VideoCapture(str(source_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = video_frame_count / video_fps if video_fps and video_fps > 0 else None
    cap.release()

    print("\nVideo information:")
    print(f"Original video FPS: {video_fps:.2f}" if video_fps else "Original video FPS: unavailable")
    print(f"Total video frames: {video_frame_count}")
    print(f"Video duration: {video_duration:.2f} seconds" if video_duration else "Video duration: unavailable")


settings_text = f"""
RAW VIDEO / IMAGE TEST SETTINGS

Test folder:
{TEST_DIR}

Source path:
{source_path}

Source type:
{SOURCE_TYPE}

Model path:
{model_path}

Image size:
{IMG_SIZE}

Confidence threshold:
{CONF_THRESHOLD}

IoU threshold:
{IOU_THRESHOLD}

Print every frame:
{PRINT_EVERY_FRAME}

Print every N frames:
{PRINT_EVERY_N_FRAMES}

Manual annotated video output:
{ANNOTATED_VIDEO_PATH if SOURCE_TYPE == "video" else "N/A"}

Notes:
- DEFECT means at least one bounding box was detected in the frame/image.
- NO DEFECT means no bounding box passed the confidence threshold.
- For videos, detection is performed per frame.
- Detected instances refer to the number of detected bounding boxes.
- Actual raw video FPS is computed from the complete timed prediction loop.
- The manual annotated video contains bounding boxes and the label "defect".
"""

with open(TEST_DIR / "run_settings.txt", "w") as f:
    f.write(settings_text)


print("\nLoading model...")
model = YOLO(str(model_path))


try:
    model.names = {0: "defect"}
except Exception:
    pass

print("Model loaded successfully.")
print("Class label used for output:", {0: "defect"})



print("\nStarting detection...")
print("Terminal output format:")
print("Frame/Image | STATUS | instances | max_conf | avg_conf | inference_ms")
print("-" * 80)

start_time = time.time()

results_generator = model.predict(
    source=str(source_path),
    imgsz=IMG_SIZE,
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    save=True,
    save_txt=True,
    save_conf=True,
    project=str(ANNOTATED_OUTPUT_DIR),
    name="ultralytics_output",
    stream=True,
    verbose=False
)

records = []

processed_count = 0
total_defect_instances = 0
frames_or_images_with_defect = 0
frames_or_images_no_defect = 0

saved_detected_samples = 0
saved_no_defect_samples = 0


video_writer = None
video_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video_fps = video_fps if video_fps and video_fps > 0 else 30

for result in results_generator:
    processed_count += 1

    result.names = {0: "defect"}

    boxes = result.boxes
    num_instances = 0 if boxes is None else len(boxes)

    if num_instances > 0:
        confs = boxes.conf.cpu().numpy().tolist()
        class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
        class_names = ["defect" for _ in class_ids]

        max_conf = float(max(confs))
        avg_conf = float(sum(confs) / len(confs))
        status = "DEFECT"

        total_defect_instances += num_instances
        frames_or_images_with_defect += 1
    else:
        confs = []
        class_ids = []
        class_names = []
        max_conf = 0.0
        avg_conf = 0.0
        status = "NO DEFECT"

        frames_or_images_no_defect += 1

    speed = result.speed if hasattr(result, "speed") else {}
    preprocess_ms = speed.get("preprocess", np.nan)
    inference_ms = speed.get("inference", np.nan)
    postprocess_ms = speed.get("postprocess", np.nan)

    if pd.notna(preprocess_ms) and pd.notna(inference_ms) and pd.notna(postprocess_ms):
        total_ms = preprocess_ms + inference_ms + postprocess_ms
    else:
        total_ms = np.nan

    if SOURCE_TYPE == "video" and video_fps and video_fps > 0:
        timestamp_sec = (processed_count - 1) / video_fps
    else:
        timestamp_sec = np.nan


    should_print = PRINT_EVERY_FRAME and (processed_count % PRINT_EVERY_N_FRAMES == 0)

    if should_print:
        item_label = "Frame" if SOURCE_TYPE == "video" else "Image"
        print(
            f"{item_label} {processed_count:06d} | "
            f"{status:<9} | "
            f"instances={num_instances:<2d} | "
            f"max_conf={max_conf:.3f} | "
            f"avg_conf={avg_conf:.3f} | "
            f"inference={inference_ms:.3f} ms"
        )


    annotated_image = result.plot(
        labels=True,
        conf=True,
        boxes=True
    )


    if SOURCE_TYPE == "video":
        if video_writer is None:
            h, w = annotated_image.shape[:2]
            video_writer = cv2.VideoWriter(
                str(ANNOTATED_VIDEO_PATH),
                video_fourcc,
                output_video_fps,
                (w, h)
            )

        video_writer.write(annotated_image)

 
    if SAVE_SAMPLE_FRAMES:
        if status == "DEFECT" and saved_detected_samples < MAX_DETECTED_SAMPLE_FRAMES:
            out_img = DETECTED_SAMPLE_DIR / f"detected_{processed_count:06d}.jpg"
            cv2.imwrite(str(out_img), annotated_image)
            saved_detected_samples += 1

        elif status == "NO DEFECT" and saved_no_defect_samples < MAX_NO_DEFECT_SAMPLE_FRAMES:
            out_img = NO_DEFECT_SAMPLE_DIR / f"no_defect_{processed_count:06d}.jpg"
            cv2.imwrite(str(out_img), annotated_image)
            saved_no_defect_samples += 1

    records.append({
        "index": processed_count,
        "source_type": SOURCE_TYPE,
        "source_path": str(source_path),
        "model_path": str(model_path),
        "timestamp_sec": timestamp_sec,
        "status": status,
        "detected_instances": num_instances,
        "class_names": ",".join(class_names),
        "max_confidence": max_conf,
        "average_confidence": avg_conf,
        "preprocess_ms": preprocess_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "total_ms": total_ms
    })



if video_writer is not None:
    video_writer.release()

end_time = time.time()


total_runtime_sec = end_time - start_time
actual_processing_fps = processed_count / total_runtime_sec if total_runtime_sec > 0 else 0

log_df = pd.DataFrame(records)

mean_preprocess_ms = log_df["preprocess_ms"].mean()
mean_inference_ms = log_df["inference_ms"].mean()
mean_postprocess_ms = log_df["postprocess_ms"].mean()
mean_total_ms = log_df["total_ms"].mean()

inference_fps_from_mean = 1000 / mean_inference_ms if mean_inference_ms > 0 else 0
total_fps_from_mean = 1000 / mean_total_ms if mean_total_ms > 0 else 0

avg_conf_detected = log_df.loc[log_df["status"] == "DEFECT", "average_confidence"].mean()
max_conf_detected = log_df.loc[log_df["status"] == "DEFECT", "max_confidence"].max()

if pd.isna(avg_conf_detected):
    avg_conf_detected = 0.0

if pd.isna(max_conf_detected):
    max_conf_detected = 0.0


csv_path = TEST_DIR / "detection_log.csv"
log_df.to_csv(csv_path, index=False)

summary_text = f"""
RAW VIDEO / IMAGE DETECTION SUMMARY

Test folder:
{TEST_DIR}

Source:
{source_path}

Source type:
{SOURCE_TYPE}

Model:
{model_path}

Detection threshold:
Confidence = {CONF_THRESHOLD}
IoU = {IOU_THRESHOLD}

OVERALL DETECTION COUNTS
Processed frames/images: {processed_count}
Frames/images with DEFECT: {frames_or_images_with_defect}
Frames/images with NO DEFECT: {frames_or_images_no_defect}
Total detected defect instances: {total_defect_instances}

CONFIDENCE SUMMARY
Average confidence on detected frames/images: {avg_conf_detected:.4f}
Maximum confidence detected: {max_conf_detected:.4f}

SPEED SUMMARY
Total runtime: {total_runtime_sec:.4f} seconds
Actual processing FPS: {actual_processing_fps:.4f}

Mean preprocess time: {mean_preprocess_ms:.4f} ms
Mean inference time: {mean_inference_ms:.4f} ms
Mean postprocess time: {mean_postprocess_ms:.4f} ms
Mean total model processing time: {mean_total_ms:.4f} ms

Inference FPS from mean inference time: {inference_fps_from_mean:.4f}
Total FPS from mean total time: {total_fps_from_mean:.4f}

VIDEO INFORMATION
Original video FPS: {video_fps if video_fps is not None else "N/A"}
Original video frame count: {video_frame_count if video_frame_count is not None else "N/A"}
Original video duration: {video_duration if video_duration is not None else "N/A"}

OUTPUT FILES
Detection log CSV:
{csv_path}

Ultralytics annotated output folder:
{ANNOTATED_OUTPUT_DIR / "ultralytics_output"}

Manual annotated video with bounding boxes:
{ANNOTATED_VIDEO_PATH if SOURCE_TYPE == "video" else "N/A"}

Sample detected frames:
{DETECTED_SAMPLE_DIR}

Sample no-defect frames:
{NO_DEFECT_SAMPLE_DIR}

INTERPRETATION NOTE
DEFECT means at least one damaged net hole bounding box was detected in the frame/image.
NO DEFECT means no bounding box passed the confidence threshold.
The manual annotated video contains bounding boxes labeled "defect".
The actual processing FPS includes the complete prediction loop and is different from the test-set benchmark FPS.
"""

summary_path = TEST_DIR / "summary_report.txt"

with open(summary_path, "w") as f:
    f.write(summary_text)

summary_df = pd.DataFrame([{
    "source_type": SOURCE_TYPE,
    "source_path": str(source_path),
    "model_path": str(model_path),
    "processed_frames_or_images": processed_count,
    "with_defect": frames_or_images_with_defect,
    "no_defect": frames_or_images_no_defect,
    "total_detected_instances": total_defect_instances,
    "average_confidence_detected": avg_conf_detected,
    "maximum_confidence_detected": max_conf_detected,
    "total_runtime_sec": total_runtime_sec,
    "actual_processing_fps": actual_processing_fps,
    "mean_preprocess_ms": mean_preprocess_ms,
    "mean_inference_ms": mean_inference_ms,
    "mean_postprocess_ms": mean_postprocess_ms,
    "mean_total_ms": mean_total_ms,
    "inference_fps_from_mean": inference_fps_from_mean,
    "total_fps_from_mean": total_fps_from_mean,
    "original_video_fps": video_fps,
    "original_video_frame_count": video_frame_count,
    "original_video_duration": video_duration,
    "manual_annotated_video_path": str(ANNOTATED_VIDEO_PATH) if SOURCE_TYPE == "video" else "N/A"
}])

summary_csv_path = TEST_DIR / "summary_report.csv"
summary_df.to_csv(summary_csv_path, index=False)


print("-" * 80)
print("Detection completed.")
print(f"Test folder: {TEST_DIR}")
print(f"Processed frames/images: {processed_count}")
print(f"Frames/images with DEFECT: {frames_or_images_with_defect}")
print(f"Frames/images with NO DEFECT: {frames_or_images_no_defect}")
print(f"Total detected defect instances: {total_defect_instances}")
print(f"Actual processing FPS: {actual_processing_fps:.2f}")
print(f"Mean inference time: {mean_inference_ms:.3f} ms")
print(f"Inference FPS from mean inference time: {inference_fps_from_mean:.2f}")
print(f"Mean total model processing time: {mean_total_ms:.3f} ms")
print(f"Total FPS from mean total time: {total_fps_from_mean:.2f}")
print("\nSaved files:")
print(f"Detection log: {csv_path}")
print(f"Summary TXT: {summary_path}")
print(f"Summary CSV: {summary_csv_path}")
print(f"Annotated outputs: {ANNOTATED_OUTPUT_DIR / 'ultralytics_output'}")

if SOURCE_TYPE == "video":
    print(f"Manual annotated video with boxes: {ANNOTATED_VIDEO_PATH}")

print(f"Sample detected frames: {DETECTED_SAMPLE_DIR}")
print(f"Sample no-defect frames: {NO_DEFECT_SAMPLE_DIR}")
print("-" * 80)

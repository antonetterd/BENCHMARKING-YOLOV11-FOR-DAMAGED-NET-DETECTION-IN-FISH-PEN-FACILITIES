
# FALSE POSITIVE / FALSE NEGATIVE IMAGE CHECKING
# For one-class damaged net hole detection


from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import zipfile
import cv2
import numpy as np
import pandas as pd
import gc
import torch
from tqdm import tqdm
from ultralytics import YOLO


SPLIT = "val"   # change to "test" if needed


ZIP_PATH = Path("/content/drive/MyDrive/thesis_training2/dataset.zip")

EXTRACT_DIR = Path("/content/dataset_unzipped")
DATASET_ROOT = EXTRACT_DIR / "dataset"

MODEL_PATH = Path(
    "/content/drive/MyDrive/thesis_training2/final_benchmark_clean_v5/"
    "benchmark_results/yolo11s_best.pt"
)

OUTPUT_DIR = Path(
    f"/content/drive/MyDrive/thesis_training2/FNFP_VAL_error_analysis_yolov11s_{SPLIT}"
)

FP_DIR = OUTPUT_DIR / "false_positives"
FN_DIR = OUTPUT_DIR / "false_negatives"
BOTH_DIR = OUTPUT_DIR / "both_false_positive_and_false_negative"
SUMMARY_DIR = OUTPUT_DIR / "summary"

for folder in [FP_DIR, FN_DIR, BOTH_DIR, SUMMARY_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_MATCH_THRESHOLD = 0.50

print("Split to inspect:", SPLIT)
print("Model path:", MODEL_PATH)
print("Output folder:", OUTPUT_DIR)


if not DATASET_ROOT.exists():
    print("Extracting dataset.zip...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

print("Dataset root:", DATASET_ROOT)

IMAGE_DIR = DATASET_ROOT / SPLIT / "images"
LABEL_DIR = DATASET_ROOT / SPLIT / "labels"

print("Image directory:", IMAGE_DIR)
print("Label directory:", LABEL_DIR)

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

if not LABEL_DIR.exists():
    raise FileNotFoundError(f"Label directory not found: {LABEL_DIR}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def yolo_label_to_xyxy(label_line, img_w, img_h):
    """
    Convert YOLO label format:
    class x_center y_center width height
    into pixel xyxy format:
    x1 y1 x2 y2 class
    """
    parts = label_line.strip().split()

    if len(parts) < 5:
        return None

    cls = int(float(parts[0]))
    x_center = float(parts[1]) * img_w
    y_center = float(parts[2]) * img_h
    width = float(parts[3]) * img_w
    height = float(parts[4]) * img_h

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return [x1, y1, x2, y2, cls]


def load_ground_truth(label_path, img_w, img_h):
    """
    Load ground-truth boxes from YOLO label file.
    Empty .txt file means no defect.
    """
    boxes = []

    if not label_path.exists():
        return boxes

    text = label_path.read_text().strip()

    if text == "":
        return boxes

    for line in text.splitlines():
        box = yolo_label_to_xyxy(line, img_w, img_h)
        if box is not None:
            boxes.append(box)

    return boxes


def compute_iou(box_a, box_b):
    """
    Compute IoU between two boxes in xyxy format.
    box: [x1, y1, x2, y2]
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold=0.50):
    """
    Match predicted boxes to ground-truth boxes.

    pred_boxes format:
    [x1, y1, x2, y2, confidence]

    gt_boxes format:
    [x1, y1, x2, y2, class]

    Returns:
    false_positive_boxes, false_negative_boxes, true_positive_matches
    """
    matched_gt = set()
    matched_pred = set()
    true_positive_matches = []

    for p_idx, pred in enumerate(pred_boxes):
        pred_xyxy = pred[:4]

        best_iou = 0.0
        best_gt_idx = -1

        for g_idx, gt in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue

            gt_xyxy = gt[:4]
            iou = compute_iou(pred_xyxy, gt_xyxy)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_threshold:
            matched_pred.add(p_idx)
            matched_gt.add(best_gt_idx)
            true_positive_matches.append((p_idx, best_gt_idx, best_iou))

    false_positive_boxes = [
        pred_boxes[i] for i in range(len(pred_boxes)) if i not in matched_pred
    ]

    false_negative_boxes = [
        gt_boxes[i] for i in range(len(gt_boxes)) if i not in matched_gt
    ]

    return false_positive_boxes, false_negative_boxes, true_positive_matches


def draw_error_boxes(image, gt_boxes, pred_boxes, fp_boxes, fn_boxes):
    """
    Draw error analysis boxes.

    Green  = ground truth defect
    Red    = false positive prediction
    Yellow = false negative / missed defect
    """
    img = image.copy()


    for box in gt_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            "GT defect",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )


    for box in fp_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        conf = box[4]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            img,
            f"FP {conf:.2f}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )


    for box in fn_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(
            img,
            "FN missed",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

    return img



image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]

image_paths = []
for ext in image_extensions:
    image_paths.extend(list(IMAGE_DIR.glob(ext)))
    image_paths.extend(list(IMAGE_DIR.glob(ext.upper())))

image_paths = sorted(image_paths)

print("Total images found:", len(image_paths))

if len(image_paths) == 0:
    raise RuntimeError("No images found. Check IMAGE_DIR path.")

model = YOLO(str(MODEL_PATH))



summary_rows = []

summary = {
    "images_checked": 0,
    "images_with_false_positive": 0,
    "images_with_false_negative": 0,
    "images_with_both_fp_and_fn": 0,
    "total_ground_truth_boxes": 0,
    "total_predicted_boxes": 0,
    "total_false_positive_boxes": 0,
    "total_false_negative_boxes": 0,
    "total_true_positive_matches": 0,
}

for img_path in tqdm(image_paths, desc=f"Checking YOLOv11n {SPLIT} errors"):
    image = cv2.imread(str(img_path))

    if image is None:
        continue

    img_h, img_w = image.shape[:2]
    label_path = LABEL_DIR / f"{img_path.stem}.txt"

    gt_boxes = load_ground_truth(label_path, img_w, img_h)

    results = model.predict(
        source=str(img_path),
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    pred_boxes = []

    for r in results:
        if r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for box, conf in zip(xyxy, confs):
                x1, y1, x2, y2 = box.tolist()
                pred_boxes.append([x1, y1, x2, y2, float(conf)])

    fp_boxes, fn_boxes, tp_matches = match_predictions_to_gt(
        pred_boxes,
        gt_boxes,
        iou_threshold=IOU_MATCH_THRESHOLD
    )

    has_fp = len(fp_boxes) > 0
    has_fn = len(fn_boxes) > 0

    summary["images_checked"] += 1
    summary["total_ground_truth_boxes"] += len(gt_boxes)
    summary["total_predicted_boxes"] += len(pred_boxes)
    summary["total_false_positive_boxes"] += len(fp_boxes)
    summary["total_false_negative_boxes"] += len(fn_boxes)
    summary["total_true_positive_matches"] += len(tp_matches)

    if has_fp:
        summary["images_with_false_positive"] += 1

    if has_fn:
        summary["images_with_false_negative"] += 1

    if has_fp and has_fn:
        summary["images_with_both_fp_and_fn"] += 1

    if has_fp or has_fn:
        drawn = draw_error_boxes(
            image=image,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            fp_boxes=fp_boxes,
            fn_boxes=fn_boxes
        )

        if has_fp and has_fn:
            save_dir = BOTH_DIR
            error_type = "both_fp_fn"
        elif has_fp:
            save_dir = FP_DIR
            error_type = "false_positive"
        else:
            save_dir = FN_DIR
            error_type = "false_negative"

        save_path = save_dir / img_path.name
        cv2.imwrite(str(save_path), drawn)

        summary_rows.append({
            "image_name": img_path.name,
            "image_path": str(img_path),
            "label_path": str(label_path),
            "error_type": error_type,
            "ground_truth_boxes": len(gt_boxes),
            "predicted_boxes": len(pred_boxes),
            "true_positive_matches": len(tp_matches),
            "false_positive_boxes": len(fp_boxes),
            "false_negative_boxes": len(fn_boxes),
            "saved_visual_path": str(save_path)
        })


summary_df = pd.DataFrame(summary_rows)
summary_csv_path = SUMMARY_DIR / f"yolov11n_{SPLIT}_error_images_summary.csv"
summary_df.to_csv(summary_csv_path, index=False)

overall_summary_df = pd.DataFrame([summary])
overall_summary_csv_path = SUMMARY_DIR / f"yolov11n_{SPLIT}_overall_summary.csv"
overall_summary_df.to_csv(overall_summary_csv_path, index=False)


print("\n============================================================")
print("YOLOv11n ERROR ANALYSIS SUMMARY")
print("============================================================")

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nSaved visual error images to:")
print("False positives:", FP_DIR)
print("False negatives:", FN_DIR)
print("Both FP and FN:", BOTH_DIR)

print("\nSaved CSV summaries:")
print("Error images summary:", summary_csv_path)
print("Overall summary:", overall_summary_csv_path)

print("\nBox color guide:")
print("Green  = ground truth defect")
print("Red    = false positive prediction")
print("Yellow = false negative / missed defect")

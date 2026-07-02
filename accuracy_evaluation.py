# EVALUATION OF PT, TENSORRT FP16, AND  TENSORRT INT8
# Uses internal TEST dataset.

from pathlib import Path
import shutil
import gc
import torch
import pandas as pd
import ultralytics
import tensorrt as trt
from ultralytics import YOLO


BASE_DIR = Path("/content/drive/MyDrive/thesis_training2/final_benchmark_clean_v5")
OUTPUT_DIR = BASE_DIR / "lightweight_quantized_results_standard"

PT_MODELS_DIR = OUTPUT_DIR / "01_original_pt_models"
EXPORT_DIR = OUTPUT_DIR / "02_exported_models"


FINAL_EVAL_DIR = OUTPUT_DIR / "03_evaluation_results_train_calib_INT8"

IMG_SIZE = 640
DEVICE = 0

DATA_YAML = Path("/content/dataset_unzipped/dataset/data.yaml")

MODEL_NAMES = ["YOLOv11n", "YOLOv11s", "YOLOv11m", "YOLOv11l", "YOLOv11x"]

print("Software versions:")
print("Ultralytics:", ultralytics.__version__)
print("TensorRT:", trt.__version__)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("DATA_YAML:", DATA_YAML, DATA_YAML.exists())

if not DATA_YAML.exists():
    raise FileNotFoundError(f"DATA_YAML not found: {DATA_YAML}")

# ------------------------------------------------------------
# 2. Make a clean evaluation folder
# ------------------------------------------------------------

if FINAL_EVAL_DIR.exists():
    print(f"[CLEAN] Removing old corrected evaluation folder: {FINAL_EVAL_DIR}")
    shutil.rmtree(FINAL_EVAL_DIR)

FINAL_EVAL_DIR.mkdir(parents=True, exist_ok=True)



export_records = []

for model_name in MODEL_NAMES:
    model_export_dir = EXPORT_DIR / model_name

    pt_path = PT_MODELS_DIR / f"{model_name}_best.pt"
    fp16_path = model_export_dir / f"{model_name}_fp16.engine"
    int8_path = model_export_dir / f"{model_name}_int8.engine"

    if not pt_path.exists():
        raise FileNotFoundError(f"Missing PT model: {pt_path}")

    if not fp16_path.exists():
        raise FileNotFoundError(f"Missing FP16 engine: {fp16_path}")

    if not int8_path.exists():
        raise FileNotFoundError(f"Missing INT8 engine: {int8_path}")

    export_records.append({
        "model": model_name,
        "pt_path": str(pt_path),
        "fp16_engine_path": str(fp16_path),
        "int8_engine_path": str(int8_path)
    })

export_df = pd.DataFrame(export_records)

print("\nFiles to evaluate:")
display(export_df)



def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def file_size_mb(path):
    path = Path(path)
    return path.stat().st_size / (1024 * 1024) if path.exists() else None

def compute_f1(p, r):
    return 2 * p * r / (p + r + 1e-9)

def evaluate_model(model_path, model_name, version_name, data_yaml):
    model_path = Path(model_path)

    print("\n" + "-" * 80)
    print(f"EVALUATING: {model_name} | {version_name}")
    print(model_path)
    print("-" * 80)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    clear_memory()

   model = YOLO(str(model_path), task="detect")

    results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMG_SIZE,
        batch=1,
        device=DEVICE,
        plots=True,
        project=str(FINAL_EVAL_DIR),
        name=f"{model_name}_{version_name}_internal_test",
        exist_ok=False
    )

    p = float(results.box.mp)
    r = float(results.box.mr)
    f1 = compute_f1(p, r)
    map50 = float(results.box.map50)
    map5095 = float(results.box.map)

    speed = results.speed
    preprocess_ms = float(speed.get("preprocess", 0))
    inference_ms = float(speed.get("inference", 0))
    postprocess_ms = float(speed.get("postprocess", 0))
    total_ms = preprocess_ms + inference_ms + postprocess_ms

    fps_inference_only = 1000 / inference_ms if inference_ms > 0 else 0
    fps_total = 1000 / total_ms if total_ms > 0 else 0

    metrics = {
        "model": model_name,
        "version": version_name,
        "model_path": str(model_path),
        "model_size_mb": round(file_size_mb(model_path), 2),

        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map5095, 4),

        "preprocess_ms": round(preprocess_ms, 3),
        "inference_ms": round(inference_ms, 3),
        "postprocess_ms": round(postprocess_ms, 3),
        "total_ms": round(total_ms, 3),

        "fps_inference_only": round(fps_inference_only, 2),
        "fps_total": round(fps_total, 2)
    }

    del model
    clear_memory()

    return metrics


eval_results = []

for _, row in export_df.iterrows():
    model_name = row["model"]

    model_versions = [
        ("PT_FP32", row["pt_path"]),
        ("TensorRT_FP16", row["fp16_engine_path"]),
        ("TensorRT_INT8_train_calib", row["int8_engine_path"]),
    ]

    for version_name, model_path in model_versions:
        try:
            eval_results.append(
                evaluate_model(model_path, model_name, version_name, DATA_YAML)
            )
        except Exception as e:
            print(f"[EVAL ERROR] {model_name} {version_name}: {e}")

eval_df = pd.DataFrame(eval_results)


EVAL_CSV = OUTPUT_DIR / "internal_test_PT_FP16_INT8_results_train_calib_INT8.csv"
eval_df.to_csv(EVAL_CSV, index=False)

print("\nFINAL CORRECTED EVALUATION RESULTS:")
display(eval_df)

print("\nSaved evaluation CSV:")
print(EVAL_CSV)

print("\nEvaluation plots saved in:")
print(FINAL_EVAL_DIR)

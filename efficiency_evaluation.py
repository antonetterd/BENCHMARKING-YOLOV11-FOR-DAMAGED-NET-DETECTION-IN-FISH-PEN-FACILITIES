
# EFFICIENCY EVALUATION ONLY
# PT, TENSORRT FP16, AND TENSORRT INT8
# Uses internal TEST dataset.
# Runs each model 10 times and reports average speed/FPS.


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

EFFICIENCY_DIR = OUTPUT_DIR / "Efficiency result final"
VAL_LOGS_DIR = EFFICIENCY_DIR / "ultralytics_val_logs"

IMG_SIZE = 640
DEVICE = 0
BATCH_SIZE = 1
N_RUNS = 10

DATA_YAML = Path("/content/dataset/dataset/data.yaml")

MODEL_NAMES = ["YOLOv11n", "YOLOv11s", "YOLOv11m", "YOLOv11l", "YOLOv11x"]

print("Software versions:")
print("Ultralytics:", ultralytics.__version__)
print("TensorRT:", trt.__version__)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("DATA_YAML:", DATA_YAML, DATA_YAML.exists())

if not DATA_YAML.exists():
    raise FileNotFoundError(f"DATA_YAML not found: {DATA_YAML}")


if EFFICIENCY_DIR.exists():
    print(f"[CLEAN] Removing old efficiency folder: {EFFICIENCY_DIR}")
    shutil.rmtree(EFFICIENCY_DIR)

EFFICIENCY_DIR.mkdir(parents=True, exist_ok=True)
VAL_LOGS_DIR.mkdir(parents=True, exist_ok=True)

ENV_TXT = EFFICIENCY_DIR / "environment_info.txt"

with open(ENV_TXT, "w") as f:
    f.write("FINAL EFFICIENCY EVALUATION ENVIRONMENT\n")
    f.write("=" * 50 + "\n")
    f.write(f"Ultralytics: {ultralytics.__version__}\n")
    f.write(f"TensorRT: {trt.__version__}\n")
    f.write(f"Torch: {torch.__version__}\n")
    f.write(f"CUDA available: {torch.cuda.is_available()}\n")
    if torch.cuda.is_available():
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
    f.write(f"Image size: {IMG_SIZE}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Number of runs per model: {N_RUNS}\n")
    f.write(f"Dataset YAML: {DATA_YAML}\n")
    f.write(f"Split used: test\n")

print(f"\nEnvironment info saved to: {ENV_TXT}")

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

    export_records.extend([
        {
            "model": model_name,
            "version": "PT_FP32",
            "model_path": str(pt_path)
        },
        {
            "model": model_name,
            "version": "TensorRT_FP16",
            "model_path": str(fp16_path)
        },
        {
            "model": model_name,
            "version": "TensorRT_INT8_train_calib",
            "model_path": str(int8_path)
        }
    ])

export_df = pd.DataFrame(export_records)

print("\nAll 15 model variants for efficiency evaluation:")
display(export_df)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def file_size_mb(path):
    path = Path(path)
    return path.stat().st_size / (1024 * 1024) if path.exists() else None

def evaluate_efficiency(model_path, model_name, version_name, run_idx, data_yaml):
    model_path = Path(model_path)

    print("\n" + "-" * 90)
    print(f"EFFICIENCY RUN {run_idx}/{N_RUNS}: {model_name} | {version_name}")
    print(model_path)
    print("-" * 90)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    clear_memory()

    model = YOLO(str(model_path), task="detect")

    results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        plots=False,
        project=str(VAL_LOGS_DIR),
        name=f"{model_name}_{version_name}_run{run_idx}",
        exist_ok=False,
        verbose=False
    )

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
        "run": run_idx,
        "model_path": str(model_path),
        "model_size_mb": round(file_size_mb(model_path), 2),

        "preprocess_ms": round(preprocess_ms, 4),
        "inference_ms": round(inference_ms, 4),
        "postprocess_ms": round(postprocess_ms, 4),
        "total_ms": round(total_ms, 4),

        "fps_inference_only": round(fps_inference_only, 4),
        "fps_total": round(fps_total, 4)
    }

    del model
    clear_memory()

    return metrics

efficiency_results = []

for run_idx in range(1, N_RUNS + 1):
    print("\n" + "=" * 100)
    print(f"STARTING EFFICIENCY EVALUATION RUN {run_idx}/{N_RUNS}")
    print("=" * 100)

    for _, row in export_df.iterrows():
        try:
            efficiency_results.append(
                evaluate_efficiency(
                    model_path=row["model_path"],
                    model_name=row["model"],
                    version_name=row["version"],
                    run_idx=run_idx,
                    data_yaml=DATA_YAML
                )
            )
        except Exception as e:
            print(f"[EFFICIENCY ERROR] Run {run_idx} | {row['model']} {row['version']}: {e}")

raw_efficiency_df = pd.DataFrame(efficiency_results)



RAW_EFFICIENCY_CSV = EFFICIENCY_DIR / "raw_efficiency_results_3runs_all15_models.csv"
raw_efficiency_df.to_csv(RAW_EFFICIENCY_CSV, index=False)

print("\nRAW EFFICIENCY RESULTS:")
display(raw_efficiency_df)

print("\nSaved raw efficiency CSV:")
print(RAW_EFFICIENCY_CSV)



summary_df = (
    raw_efficiency_df
    .groupby(["model", "version"], as_index=False)
    .agg({
        "model_size_mb": "mean",

        "preprocess_ms": ["mean", "std"],
        "inference_ms": ["mean", "std"],
        "postprocess_ms": ["mean", "std"],
        "total_ms": ["mean", "std"],

        "fps_inference_only": ["mean", "std"],
        "fps_total": ["mean", "std"]
    })
)


summary_df.columns = [
    "_".join(col).strip("_") if isinstance(col, tuple) else col
    for col in summary_df.columns
]


summary_df = summary_df.round(4)

SUMMARY_EFFICIENCY_CSV = EFFICIENCY_DIR / "summary_efficiency_average_3runs_all15_models.csv"
summary_df.to_csv(SUMMARY_EFFICIENCY_CSV, index=False)

print("\nAVERAGED FINAL EFFICIENCY SUMMARY:")
display(summary_df)

print("\nSaved average efficiency summary CSV:")
print(SUMMARY_EFFICIENCY_CSV)

print("\nEfficiency results saved in:")
print(EFFICIENCY_DIR)

print("\nUltralytics validation logs saved in:")
print(VAL_LOGS_DIR)

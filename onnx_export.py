
# EXPORT ONNX INTERMEDIATE FILES
# ONNX is saved for conversion

export_records = []

for model_name, pt_path in MODEL_PATHS.items():
    print("\n" + "=" * 80)
    print(f"EXPORTING ONNX: {model_name}")
    print("=" * 80)

    model_export_dir = EXPORT_DIR / model_name
    model_export_dir.mkdir(parents=True, exist_ok=True)

    onnx_dst = model_export_dir / f"{model_name}.onnx"
    fp16_dst = model_export_dir / f"{model_name}_fp16.engine"
    int8_dst = model_export_dir / f"{model_name}_int8.engine"

    record = {
        "model": model_name,
        "pt_path": str(pt_path),
        "onnx_path": str(onnx_dst) if onnx_dst.exists() else "",
        "fp16_engine_path": str(fp16_dst) if fp16_dst.exists() else "",
        "int8_engine_path": str(int8_dst) if int8_dst.exists() else ""
    }

    if onnx_dst.exists():
        print(f"[SKIP] ONNX already exists: {onnx_dst}")
        export_records.append(record)
        continue

    model = YOLO(str(pt_path))

    try:
        onnx_path = model.export(
            format="onnx",
            imgsz=IMG_SIZE,
            device=DEVICE,
            simplify=True
        )

        onnx_path = Path(onnx_path)
        shutil.copy2(onnx_path, onnx_dst)

        record["onnx_path"] = str(onnx_dst)
        print(f"[ONNX DONE] {onnx_dst}")

    except Exception as e:
        print(f"[ONNX ERROR] {model_name}: {e}")

    export_records.append(record)

    del model
    clear_memory()

export_df = pd.DataFrame(export_records)

EXPORT_CSV = OUTPUT_DIR / "export_records_with_onnx.csv"
export_df.to_csv(EXPORT_CSV, index=False)

display(export_df)
print("Saved:", EXPORT_CSV)



# contiuation in next cell (Google Colab)
# RECREATE export records AFTER RUNTIME RESTART
from pathlib import Path
import pandas as pd
import shutil
import gc
import torch
from ultralytics import YOLO

BASE_DIR = Path("/content/drive/MyDrive/thesis_training2/final_benchmark_clean_v5")
OUTPUT_DIR = BASE_DIR / "lightweight_quantized_results_standard"

PT_MODELS_DIR = OUTPUT_DIR / "01_original_pt_models"
EXPORT_DIR = OUTPUT_DIR / "02_exported_models"
EVAL_DIR = OUTPUT_DIR / "03_evaluation_results"
GRAPH_DIR = OUTPUT_DIR / "04_graphs_no_onnx"

PT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 640
DEVICE = 0

DATA_YAML = Path("/content/dataset_unzipped/dataset/data.yaml")

MODEL_NAMES = ["YOLOv11n", "YOLOv11s", "YOLOv11m", "YOLOv11l", "YOLOv11x"]

export_records = []

for model_name in MODEL_NAMES:
    model_export_dir = EXPORT_DIR / model_name
    model_export_dir.mkdir(parents=True, exist_ok=True)

    pt_path = PT_MODELS_DIR / f"{model_name}_best.pt"
    onnx_path = model_export_dir / f"{model_name}.onnx"
    fp16_path = model_export_dir / f"{model_name}_fp16.engine"
    int8_path = model_export_dir / f"{model_name}_int8.engine"

    export_records.append({
        "model": model_name,
        "pt_path": str(pt_path) if pt_path.exists() else "",
        "onnx_path": str(onnx_path) if onnx_path.exists() else "",
        "fp16_engine_path": str(fp16_path) if fp16_path.exists() else "",
        "int8_engine_path": str(int8_path) if int8_path.exists() else ""
    })

export_df = pd.DataFrame(export_records)

display(export_df)

print("\nChecking files:")
for _, row in export_df.iterrows():
    print("\n", row["model"])
    print("PT exists   :", bool(row["pt_path"]) and Path(row["pt_path"]).exists())
    print("ONNX exists :", bool(row["onnx_path"]) and Path(row["onnx_path"]).exists())
    print("FP16 exists :", bool(row["fp16_engine_path"]) and Path(row["fp16_engine_path"]).exists())
    print("INT8 exists :", bool(row["int8_engine_path"]) and Path(row["int8_engine_path"]).exists())

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

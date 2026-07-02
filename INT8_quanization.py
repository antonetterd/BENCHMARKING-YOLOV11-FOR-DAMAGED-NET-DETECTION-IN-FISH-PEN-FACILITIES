
# EXPORT TENSORRT INT8 QUANTIZED MODELS 
# Uses TRAIN images for calibration

from pathlib import Path
import shutil
import pandas as pd
import yaml
from ultralytics import YOLO


DATASET_ROOT = Path("/content/dataset_unzipped/dataset")  

CALIB_YAML = OUTPUT_DIR / "data_train_calibration.yaml"

calib_data = {
    "path": str(DATASET_ROOT),
    "train": "train/images",
    "val": "train/images",      
    "test": "test/images",
    "nc": 1,
    "names": ["defect"]
}

with open(CALIB_YAML, "w") as f:
    yaml.safe_dump(calib_data, f, sort_keys=False)

print("Created train-calibration YAML:")
print(CALIB_YAML)
print(open(CALIB_YAML).read())


for i, record in enumerate(export_records):
    model_name = record["model"]
    pt_path = Path(record["pt_path"])

    print("\n" + "=" * 80)
    print(f"RE-EXPORTING TensorRT INT8 WITH TRAIN CALIBRATION: {model_name}")
    print("=" * 80)

    model_export_dir = EXPORT_DIR / model_name
    model_export_dir.mkdir(parents=True, exist_ok=True)

    int8_dst = model_export_dir / f"{model_name}_int8.engine"

    if int8_dst.exists():
        print(f"[OVERWRITE] Removing old INT8 engine: {int8_dst}")
        int8_dst.unlink()

  
    default_engine = pt_path.with_suffix(".engine")
    if default_engine.exists():
        print(f"[CLEAN] Removing old default engine: {default_engine}")
        default_engine.unlink()

    model = YOLO(str(pt_path))

    try:
        int8_path = model.export(
            format="engine",
            imgsz=IMG_SIZE,
            int8=True,
            data=str(CALIB_YAML),
            fraction=1.0,
            device=DEVICE
        )

        int8_path = Path(int8_path)

        if not int8_path.exists():
            raise FileNotFoundError(f"Exported INT8 engine not found: {int8_path}")

        shutil.copy2(int8_path, int8_dst)

        export_records[i]["int8_engine_path"] = str(int8_dst)

        print(f"[INT8 RE-EXPORT DONE] {int8_dst}")

    except Exception as e:
        print(f"[INT8 ERROR] {model_name}: {e}")

    del model
    clear_memory()



export_df = pd.DataFrame(export_records)

EXPORT_CSV = OUTPUT_DIR / "export_records_final_train_calib_INT8.csv"
export_df.to_csv(EXPORT_CSV, index=False)

display(export_df)
print("Saved updated export records:", EXPORT_CSV)

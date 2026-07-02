
# EXPORT TENSORRT FP16 OPTIMIZED MODELS after onnx model export (Google Colab)


for i, record in enumerate(export_records):
    model_name = record["model"]
    pt_path = Path(record["pt_path"])

    print("\n" + "=" * 80)
    print(f"EXPORTING TensorRT FP16: {model_name}")
    print("=" * 80)

    model_export_dir = EXPORT_DIR / model_name
    model_export_dir.mkdir(parents=True, exist_ok=True)

    fp16_dst = model_export_dir / f"{model_name}_fp16.engine"

    if fp16_dst.exists():
        print(f"[SKIP] FP16 already exists: {fp16_dst}")
        export_records[i]["fp16_engine_path"] = str(fp16_dst)
        continue

    model = YOLO(str(pt_path))

    try:
        fp16_path = model.export(
            format="engine",
            imgsz=IMG_SIZE,
            half=True,
            device=DEVICE
        )

        fp16_path = Path(fp16_path)
        shutil.copy2(fp16_path, fp16_dst)

        export_records[i]["fp16_engine_path"] = str(fp16_dst)
        print(f"[FP16 DONE] {fp16_dst}")

    except Exception as e:
        print(f"[FP16 ERROR] {model_name}: {e}")

    del model
    clear_memory()

export_df = pd.DataFrame(export_records)

EXPORT_CSV = OUTPUT_DIR / "export_records_with_fp16.csv"
export_df.to_csv(EXPORT_CSV, index=False)

display(export_df)
print("Saved:", EXPORT_CSV)

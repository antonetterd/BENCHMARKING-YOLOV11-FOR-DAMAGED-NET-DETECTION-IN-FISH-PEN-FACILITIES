# training script to train all YOLOv11 models (n,s,m,l,x) sequentially.
# This training script was originally designed to run in Google Colab.


%%writefile /content/train.py

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from ultralytics import YOLO
import pandas as pd
import time
import numpy as np
import shutil
import json
import gc
import glob
import random
import hashlib

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_DIR = "/content/drive/MyDrive/thesis_training2"

RUN_NAME = "final_benchmark_clean_v5"
RUN_DIR = f"{BASE_DIR}/{RUN_NAME}"

DATASET_ROOT = "/content/dataset/dataset"
DATASET_YAML = f"{DATASET_ROOT}/data.yaml"

EPOCHS = 100
PATIENCE = 100
IMGSZ = 640

WORKERS = 4
DEVICE = 0

BATCH_SIZES = {
    "n": 16,
    "s": 16,
    "m": 16,
    "l": 16,
    "x": 16
}

ALL_MODELS = ["n", "s", "m", "l", "x"]

TRAINING_DIR = f"{RUN_DIR}/training_results"
RESULTS_DIR = f"{RUN_DIR}/benchmark_results"

os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

PROGRESS_FILE = os.path.join(RESULTS_DIR, "training_progress.json")
TIME_FILE = os.path.join(RESULTS_DIR, "training_time_tracker.json")


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            print("WARNING: Progress file is corrupted. Starting fresh.")

    return {
        "completed_models": [],
        "results_summary": []
    }


def save_progress(completed_models, results_summary):
    temp_file = PROGRESS_FILE + ".tmp"

    with open(temp_file, "w") as f:
        json.dump(
            {
                "completed_models": completed_models,
                "results_summary": results_summary
            },
            f,
            indent=4
        )

    os.replace(temp_file, PROGRESS_FILE)


def load_time_tracker():
    if os.path.exists(TIME_FILE):
        try:
            with open(TIME_FILE, "r") as f:
                return json.load(f)
        except Exception:
            print("WARNING: Time tracker file is corrupted. Starting fresh.")

    return {}


def save_time_tracker(time_tracker):
    temp_file = TIME_FILE + ".tmp"

    with open(temp_file, "w") as f:
        json.dump(time_tracker, f, indent=4)

    os.replace(temp_file, TIME_FILE)


def get_accumulated_time_seconds(model_size):
    time_tracker = load_time_tracker()
    model_key = f"YOLOv11{model_size}"

    return time_tracker.get(model_key, {}).get("accumulated_seconds", 0)


def update_accumulated_time_seconds(model_size, accumulated_seconds):
    time_tracker = load_time_tracker()
    model_key = f"YOLOv11{model_size}"

    time_tracker[model_key] = {
        "accumulated_seconds": accumulated_seconds,
        "accumulated_minutes": round(accumulated_seconds / 60, 2),
        "accumulated_hours": round(accumulated_seconds / 3600, 4),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    save_time_tracker(time_tracker)


def reset_model_time_if_starting_fresh(model_size, resume_training):
    if resume_training:
        return

    time_tracker = load_time_tracker()
    model_key = f"YOLOv11{model_size}"

    if model_key in time_tracker:
        print(f"Fresh training detected. Resetting old accumulated time for {model_key}.")
        del time_tracker[model_key]
        save_time_tracker(time_tracker)


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()
    print("GPU cache cleared.")


def get_file_hash(filepath):
    hasher = hashlib.md5()

    with open(filepath, "rb") as f:
        hasher.update(f.read())

    return hasher.hexdigest()


def count_defect_no_defect(label_dir, image_files):
    defect_count = 0
    no_defect_count = 0
    missing_count = 0

    for image_path in image_files:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(label_dir, filename + ".txt")

        if not os.path.exists(label_path):
            missing_count += 1
        elif os.path.getsize(label_path) > 0:
            defect_count += 1
        else:
            no_defect_count += 1

    return defect_count, no_defect_count, missing_count


def verify_dataset():
    print("\n" + "=" * 70)
    print("VERIFYING DATASET")
    print("=" * 70)

    if not os.path.exists(DATASET_YAML):
        raise FileNotFoundError(f"DATASET_YAML not found: {DATASET_YAML}")

    splits = ["train", "val", "test"]
    all_hashes = {}

    expected_counts = {
        "train": 14562,
        "val": 3120,
        "test": 3120
    }

    for split in splits:
        image_dir = f"{DATASET_ROOT}/{split}/images"
        label_dir = f"{DATASET_ROOT}/{split}/labels"

        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Missing image folder: {image_dir}")

        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Missing label folder: {label_dir}")

        images = glob.glob(f"{image_dir}/*.*")

        print(f"\n{split.upper()} SET")
        print(f"Images: {len(images)}")

        if split in expected_counts and len(images) != expected_counts[split]:
            print(
                f"WARNING: Expected {expected_counts[split]} images for {split}, "
                f"but found {len(images)}."
            )

        missing_labels = 0
        empty_labels = 0
        corrupt_images = 0
        exact_duplicates = 0

        for image_path in images:
            filename = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(label_dir, filename + ".txt")

            if not os.path.exists(label_path):
                missing_labels += 1
            else:
                if os.path.getsize(label_path) == 0:
                    empty_labels += 1

            try:
                from PIL import Image
                img = Image.open(image_path)
                img.verify()
            except Exception:
                corrupt_images += 1

            try:
                file_hash = get_file_hash(image_path)

                if file_hash in all_hashes:
                    exact_duplicates += 1
                else:
                    all_hashes[file_hash] = image_path
            except Exception:
                pass

        defect_count, no_defect_count, missing_count = count_defect_no_defect(label_dir, images)

        print(f"Defect images: {defect_count}")
        print(f"No-defect/empty-label images: {no_defect_count}")
        print(f"Missing labels: {missing_labels}")
        print(f"Empty labels/no-defect images: {empty_labels}")
        print(f"Corrupt images: {corrupt_images}")
        print(f"Exact duplicate images across checked splits: {exact_duplicates}")

        if missing_labels > 0:
            raise RuntimeError(f"{split} has missing labels. Fix before training.")

        if corrupt_images > 0:
            raise RuntimeError(f"{split} has corrupt images. Fix before training.")


def get_actual_epochs(save_dir):
    results_csv = os.path.join(save_dir, "results.csv")

    if not os.path.exists(results_csv):
        return None

    try:
        df = pd.read_csv(results_csv)
        return len(df)
    except Exception:
        return None


def train_model(model_size, results_summary):
    print("\n" + "=" * 70)
    print(f"Training YOLOv11{model_size}")
    print("=" * 70)

    model_name = f"yolo11{model_size}.pt"
    train_name = f"yolo11{model_size}_training"

    save_dir = f"{TRAINING_DIR}/{train_name}"

    checkpoint_path = f"{save_dir}/weights/last.pt"
    best_checkpoint = f"{save_dir}/weights/best.pt"

    resume_training = os.path.exists(checkpoint_path)

    clear_gpu_cache()

    try:
        if resume_training:
            print("Resuming previous training from last.pt...")

            try:
                model = YOLO(checkpoint_path)

            except Exception:
                print("WARNING: last.pt appears corrupted.")
                print("Falling back to best.pt.")

                if not os.path.exists(best_checkpoint):
                    raise FileNotFoundError("Both last.pt and best.pt are unavailable.")

                model = YOLO(best_checkpoint)
                resume_training = False

        else:
            model = YOLO(model_name)

        reset_model_time_if_starting_fresh(model_size, resume_training)

        previous_time_seconds = get_accumulated_time_seconds(model_size)

        print(
            f"Previous accumulated time for YOLOv11{model_size}: "
            f"{previous_time_seconds / 60:.2f} minutes"
        )

        session_start_time = time.time()

        def save_epoch_time(trainer):
            current_total_seconds = previous_time_seconds + (
                time.time() - session_start_time
            )

            update_accumulated_time_seconds(
                model_size,
                current_total_seconds
            )

        model.add_callback("on_train_epoch_end", save_epoch_time)
        model.add_callback("on_train_end", save_epoch_time)

        model.train(
            data=DATASET_YAML,
            epochs=EPOCHS,
            patience=PATIENCE,
            imgsz=IMGSZ,
            batch=BATCH_SIZES[model_size],
            device=DEVICE,
            workers=WORKERS,
            name=train_name,
            project=TRAINING_DIR,
            exist_ok=True,
            verbose=True,

            resume=resume_training,

            save=True,
            save_period=1,

            seed=SEED,
            deterministic=True,
            optimizer="SGD",
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=0.0,
            warmup_bias_lr=0.001,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,

            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,

            fliplr=0.0,
            flipud=0.0,

            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            erasing=0.0,

            close_mosaic=0
        )

        train_time = previous_time_seconds + (
            time.time() - session_start_time
        )

        update_accumulated_time_seconds(
            model_size,
            train_time
        )

        actual_epochs = get_actual_epochs(save_dir)

        best_model_path = f"{save_dir}/weights/best.pt"

        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"best.pt not found: {best_model_path}")

        best_model = YOLO(best_model_path)

        print("\nEvaluating on VALIDATION set...")

        val_results = best_model.val(
            data=DATASET_YAML,
            device=DEVICE,
            split="val",
            imgsz=IMGSZ,
            plots=True,
            project=RESULTS_DIR,
            name=f"yolo11{model_size}_val_eval",
            exist_ok=True
        )

        val_p = val_results.box.mp
        val_r = val_results.box.mr
        val_m50 = val_results.box.map50
        val_m50_95 = val_results.box.map
        val_f1 = 2 * (val_p * val_r) / (val_p + val_r + 1e-6)

        print(
            f"Validation - Precision: {val_p:.4f}, "
            f"Recall: {val_r:.4f}, "
            f"mAP50: {val_m50:.4f}, "
            f"mAP50-95: {val_m50_95:.4f}, "
            f"F1: {val_f1:.4f}"
        )


        print("\nEvaluating on TEST set...")

        test_results = best_model.val(
            data=DATASET_YAML,
            device=DEVICE,
            split="test",
            imgsz=IMGSZ,
            plots=True,
            project=RESULTS_DIR,
            name=f"yolo11{model_size}_test_eval_standard",
            exist_ok=True
        )

        test_p = test_results.box.mp
        test_r = test_results.box.mr
        test_m50 = test_results.box.map50
        test_m50_95 = test_results.box.map
        test_f1 = 2 * (test_p * test_r) / (test_p + test_r + 1e-6)

        print(
            f"TEST - Precision: {test_p:.4f}, "
            f"Recall: {test_r:.4f}, "
            f"mAP50: {test_m50:.4f}, "
            f"mAP50-95: {test_m50_95:.4f}, "
            f"F1: {test_f1:.4f}"
        )


        speed_ms = test_results.speed.get("inference", 0)
        fps = 1000 / speed_ms if speed_ms > 0 else 0


        metrics = {
            "model": f"YOLOv11{model_size}",
            "max_epochs": EPOCHS,
            "actual_epochs": actual_epochs,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZES[model_size],
            "train_time_min": round(train_time / 60, 2),
            "train_time_hours": round(train_time / 3600, 4),

            "val_precision": round(val_p, 4),
            "val_recall": round(val_r, 4),
            "val_mAP50": round(val_m50, 4),
            "val_mAP50_95": round(val_m50_95, 4),
            "val_f1": round(val_f1, 4),

            "test_precision": round(test_p, 4),
            "test_recall": round(test_r, 4),
            "test_mAP50": round(test_m50, 4),
            "test_mAP50_95": round(test_m50_95, 4),
            "test_f1": round(test_f1, 4),

            "inference_speed_ms": round(speed_ms, 2),
            "fps": round(fps, 2),

            "model_size_mb": round(
                os.path.getsize(best_model_path) / (1024 * 1024),
                2
            )
        }

        results_summary = [
            r for r in results_summary
            if r.get("model") != f"YOLOv11{model_size}"
        ]

        results_summary.append(metrics)

        important_files = [
            "results.png",
            "results.csv",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "PR_curve.png",
            "P_curve.png",
            "R_curve.png",
            "F1_curve.png"
        ]

        for file in important_files:
            source_path = os.path.join(save_dir, file)

            if os.path.exists(source_path):
                shutil.copy(
                    source_path,
                    os.path.join(
                        RESULTS_DIR,
                        f"yolo11{model_size}_{file}"
                    )
                )

        shutil.copy(
            best_model_path,
            os.path.join(
                RESULTS_DIR,
                f"yolo11{model_size}_best.pt"
            )
        )

        del model
        del best_model

        clear_gpu_cache()

        return results_summary, True

    except Exception as e:
        print(f"\n!!! ERROR training YOLOv11{model_size}: {e}")

        clear_gpu_cache()

        return results_summary, False

def save_final_report(results_summary):
    df = pd.DataFrame(results_summary)

    csv_path = os.path.join(
        RESULTS_DIR,
        "benchmark_results.csv"
    )

    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(df.to_string(index=False))

    print(f"\nSaved CSV to: {csv_path}")


def main():
    print("=" * 70)
    print("YOLOv11 BENCHMARK TRAINING")
    print("=" * 70)

    print(f"Run directory      : {RUN_DIR}")
    print(f"Dataset root       : {DATASET_ROOT}")
    print(f"Dataset YAML       : {DATASET_YAML}")
    print(f"Training directory : {TRAINING_DIR}")
    print(f"Benchmark directory: {RESULTS_DIR}")
    print(f"Time tracker file  : {TIME_FILE}")
    print(f"Device             : {DEVICE}")

    verify_dataset()

    train_images = glob.glob(f"{DATASET_ROOT}/train/images/*.*")
    val_images = glob.glob(f"{DATASET_ROOT}/val/images/*.*")
    test_images = glob.glob(f"{DATASET_ROOT}/test/images/*.*")

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"Train Images     : {len(train_images)}")
    print(f"Validation Images: {len(val_images)}")
    print(f"Test Images      : {len(test_images)}")

    print("\nTraining settings:")
    print(f"Max epochs : {EPOCHS}")
    print(f"Patience   : {PATIENCE}")
    print(f"Image size : {IMGSZ}")
    print(f"Workers    : {WORKERS}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print("Optimizer: SGD")
    print("Initial learning rate lr0: 0.001")
    print("Online augmentation: DISABLED")
    print("Offline augmentation: ALREADY APPLIED BEFORE SPLIT")

    progress = load_progress()

    completed_models = progress.get("completed_models", [])
    results_summary = progress.get("results_summary", [])

    for model_size in ALL_MODELS:
        if model_size in completed_models:
            print(
                f"\nSkipping YOLOv11{model_size} "
                f"(already completed)."
            )
            continue

        results_summary, success = train_model(
            model_size,
            results_summary
        )

        if success:
            if model_size not in completed_models:
                completed_models.append(model_size)

            save_progress(
                completed_models,
                results_summary
            )

            print(f"\nYOLOv11{model_size} completed.")

        else:
            print(f"\nYOLOv11{model_size} failed.")
            print("Re-run script to continue from last checkpoint.")
            break

    if len(completed_models) == len(ALL_MODELS):
        save_final_report(results_summary)

        print("\n" + "=" * 70)
        print("ALL MODELS COMPLETED")
        print("=" * 70)

    else:
        print("\n" + "=" * 70)
        print("BENCHMARK INCOMPLETE")
        print("=" * 70)
        print(f"Completed models: {completed_models}")
        print(f"Remaining models: {[m for m in ALL_MODELS if m not in completed_models]}")


if __name__ == "__main__":
    main()

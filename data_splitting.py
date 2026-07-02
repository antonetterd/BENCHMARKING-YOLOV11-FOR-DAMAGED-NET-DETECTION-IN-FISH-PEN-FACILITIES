# Apply data splitting technique using 70/15/15.

import os
import shutil
import random
import json
from collections import defaultdict



INPUT_SOURCE_DIR = r"C:\Users\ANTONETTE\thesis_yolo\orig_data_cleaned"

OUTPUT_DATASET_DIR = r"C:\Users\ANTONETTE\thesis_yolo\net_dataset_split_70_15_15"

SOURCE_FOLDERS = [
    "artificial_with_defect",
    "pen1_no_defect",
    "pen2_no_defect",
    "pen2_with_defect",
    "pen3_no_defect",
    "pen4_no_defect",
    "pen4_with_defect",
]

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASS_NAME = "defect"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

SPLITS = ["train", "val", "test"]

random.seed(SEED)

if os.path.exists(OUTPUT_DATASET_DIR):
    print(f"[INFO] Removing old output dataset folder: {OUTPUT_DATASET_DIR}")
    shutil.rmtree(OUTPUT_DATASET_DIR)

for split in SPLITS:
    os.makedirs(os.path.join(OUTPUT_DATASET_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DATASET_DIR, split, "labels"), exist_ok=True)


def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def is_with_defect_folder(folder_name):
    return folder_name.endswith("_with_defect")


def is_no_defect_folder(folder_name):
    return folder_name.endswith("_no_defect")


def label_is_defect(label_path):
    return os.path.exists(label_path) and os.path.getsize(label_path) > 0


def calculate_split_counts(total_count):
    """
    Calculates 70/15/15 split count.
    Test count is calculated last so the total remains exact.
    """
    train_count = round(total_count * TRAIN_RATIO)
    val_count = round(total_count * VAL_RATIO)
    test_count = total_count - train_count - val_count

    return train_count, val_count, test_count


def copy_item(item, split, used_filenames):
    """
    Copies one image and its matching YOLO label file to the selected split.
    A source prefix is added to avoid filename conflicts after merging folders.
    """

    output_image_dir = os.path.join(OUTPUT_DATASET_DIR, split, "images")
    output_label_dir = os.path.join(OUTPUT_DATASET_DIR, split, "labels")

    source = item["source"]
    image_file = item["image_file"]
    label_file = item["label_file"]

    source_image_path = item["image_path"]
    source_label_path = item["label_path"]

    image_base, image_ext = os.path.splitext(image_file)
    label_ext = ".txt"

    output_base_name = f"{source}_{image_base}"
    output_image_name = output_base_name + image_ext
    output_label_name = output_base_name + label_ext

    counter = 1

    while output_image_name in used_filenames:
        output_base_name = f"{source}_{image_base}_{counter}"
        output_image_name = output_base_name + image_ext
        output_label_name = output_base_name + label_ext
        counter += 1

    used_filenames.add(output_image_name)

    output_image_path = os.path.join(output_image_dir, output_image_name)
    output_label_path = os.path.join(output_label_dir, output_label_name)

    shutil.copy2(source_image_path, output_image_path)
    shutil.copy2(source_label_path, output_label_path)


def count_split(split):
    image_dir = os.path.join(OUTPUT_DATASET_DIR, split, "images")
    label_dir = os.path.join(OUTPUT_DATASET_DIR, split, "labels")

    image_files = [
        f for f in os.listdir(image_dir)
        if is_image_file(f)
    ]

    total = len(image_files)
    defect = 0
    no_defect = 0
    missing = 0

    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        label_path = os.path.join(label_dir, base_name + ".txt")

        if not os.path.exists(label_path):
            missing += 1
        elif os.path.getsize(label_path) > 0:
            defect += 1
        else:
            no_defect += 1

    return total, defect, no_defect, missing


print("=" * 90)
print("SOURCE-WISE 70/15/15 DATASET SPLITTING")
print("=" * 90)
print(f"Input cleaned folder : {INPUT_SOURCE_DIR}")
print(f"Output dataset folder: {OUTPUT_DATASET_DIR}")
print(f"Split ratio          : {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}")
print(f"Seed                 : {SEED}")
print("=" * 90)

final_train_items = []
final_val_items = []
final_test_items = []

missing_labels = []
bad_labels = []

source_reports = {}


for source in SOURCE_FOLDERS:

    print("\n" + "=" * 90)
    print(f"PROCESSING SOURCE FOLDER: {source}")
    print("=" * 90)

    image_dir = os.path.join(INPUT_SOURCE_DIR, source, "images")
    label_dir = os.path.join(INPUT_SOURCE_DIR, source, "labels")

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Missing image folder: {image_dir}")

    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"Missing label folder: {label_dir}")

    image_files = [
        f for f in os.listdir(image_dir)
        if is_image_file(f)
    ]

    image_files.sort()

    source_items = []

    source_defect_images = 0
    source_no_defect_images = 0


    for image_file in image_files:

        image_base = os.path.splitext(image_file)[0]
        label_file = image_base + ".txt"

        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            missing_labels.append(f"{source}/{image_file}")
            continue

        has_defect = label_is_defect(label_path)

    
        if is_with_defect_folder(source) and not has_defect:
            bad_labels.append(f"{source}/{label_file} expected defect label but label is empty")
            continue
        if is_no_defect_folder(source) and has_defect:
            bad_labels.append(f"{source}/{label_file} expected empty label but label has annotation")
            continue

        item = {
            "source": source,
            "image_file": image_file,
            "label_file": label_file,
            "image_path": image_path,
            "label_path": label_path,
            "has_defect": has_defect,
        }

        source_items.append(item)

        if has_defect:
            source_defect_images += 1
        else:
            source_no_defect_images += 1


    random.shuffle(source_items)

    train_count, val_count, test_count = calculate_split_counts(len(source_items))

    source_train_items = source_items[:train_count]
    source_val_items = source_items[train_count:train_count + val_count]
    source_test_items = source_items[train_count + val_count:]

    final_train_items.extend(source_train_items)
    final_val_items.extend(source_val_items)
    final_test_items.extend(source_test_items)

    source_reports[source] = {
        "total_images": len(source_items),
        "defect_images": source_defect_images,
        "no_defect_images": source_no_defect_images,
        "train_images": len(source_train_items),
        "val_images": len(source_val_items),
        "test_images": len(source_test_items),
    }

    print(f"Total valid images : {len(source_items)}")
    print(f"Defect images      : {source_defect_images}")
    print(f"No-defect images   : {source_no_defect_images}")
    print("-" * 90)
    print(f"Train images       : {len(source_train_items)}")
    print(f"Val images         : {len(source_val_items)}")
    print(f"Test images        : {len(source_test_items)}")


if missing_labels:
    print("\nERROR: Missing labels found.")
    for item in missing_labels[:50]:
        print(item)
    raise RuntimeError("Fix missing labels before splitting.")

if bad_labels:
    print("\nERROR: Label/folder mismatch found.")
    for item in bad_labels[:50]:
        print(item)
    raise RuntimeError("Fix label/folder mismatch before splitting.")


random.shuffle(final_train_items)
random.shuffle(final_val_items)
random.shuffle(final_test_items)

split_items = {
    "train": final_train_items,
    "val": final_val_items,
    "test": final_test_items,
}


used_filenames_by_split = {
    "train": set(),
    "val": set(),
    "test": set(),
}

print("\n" + "=" * 90)
print("COPYING IMAGES AND LABELS TO FINAL YOLO DATASET")
print("=" * 90)

for split, items in split_items.items():
    print(f"Copying {split}: {len(items)} images")
    for item in items:
        copy_item(item, split, used_filenames_by_split[split])

data_yaml_path = os.path.join(OUTPUT_DATASET_DIR, "data.yaml")

with open(data_yaml_path, "w", encoding="utf-8") as f:
    f.write("train: train/images\n")
    f.write("val: val/images\n")
    f.write("test: test/images\n\n")
    f.write("nc: 1\n")
    f.write(f"names: ['{CLASS_NAME}']\n")


print("\n" + "=" * 90)
print("FINAL SPLIT COUNT")
print("=" * 90)

final_summary = {}

grand_total_images = 0

for split in SPLITS:
    total, defect, no_defect, missing = count_split(split)
    grand_total_images += total

    final_summary[split] = {
        "total": total,
        "defect": defect,
        "no_defect": no_defect,
        "missing": missing,
    }

for split in SPLITS:

    total = final_summary[split]["total"]
    defect = final_summary[split]["defect"]
    no_defect = final_summary[split]["no_defect"]
    missing = final_summary[split]["missing"]

    percent = (total / grand_total_images) * 100 if grand_total_images > 0 else 0

    final_summary[split]["percentage"] = percent

    print(f"\n{split.upper()} SET")
    print("-" * 90)
    print(f"Total images     : {total}")
    print(f"Defect images    : {defect}")
    print(f"No-defect images : {no_defect}")
    print(f"Missing labels   : {missing}")
    print(f"Split percentage : {percent:.2f}%")

print("\n" + "=" * 90)
print("SOURCE-WISE SPLIT DISTRIBUTION CHECK")
print("=" * 90)

for source, info in source_reports.items():

    total_images = info["total_images"]

    train_percent = (info["train_images"] / total_images) * 100 if total_images > 0 else 0
    val_percent = (info["val_images"] / total_images) * 100 if total_images > 0 else 0
    test_percent = (info["test_images"] / total_images) * 100 if total_images > 0 else 0

    print(f"\n{source}")
    print("-" * 90)
    print(f"Total images : {total_images}")
    print(f"Train images : {info['train_images']} ({train_percent:.2f}%)")
    print(f"Val images   : {info['val_images']} ({val_percent:.2f}%)")
    print(f"Test images  : {info['test_images']} ({test_percent:.2f}%)")


report_path = os.path.join(OUTPUT_DATASET_DIR, "split_report.txt")
json_report_path = os.path.join(OUTPUT_DATASET_DIR, "split_report.json")

report_data = {
    "input_source_dir": INPUT_SOURCE_DIR,
    "output_dataset_dir": OUTPUT_DATASET_DIR,
    "seed": SEED,
    "split_ratio": {
        "train": TRAIN_RATIO,
        "val": VAL_RATIO,
        "test": TEST_RATIO,
    },
    "method": "source-wise 70/15/15 split after cleaning, before augmentation",
    "source_reports": source_reports,
    "final_summary": final_summary,
}

with open(json_report_path, "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=4)

with open(report_path, "w", encoding="utf-8") as report:

    report.write("SOURCE-WISE 70/15/15 DATASET SPLIT REPORT\n")
    report.write("=" * 90 + "\n\n")

    report.write(f"Input cleaned folder : {INPUT_SOURCE_DIR}\n")
    report.write(f"Output dataset       : {OUTPUT_DATASET_DIR}\n")
    report.write(f"Seed                 : {SEED}\n")
    report.write("Split ratio          : 70/15/15\n")
    report.write("Method               : source-wise split after cleaning, before augmentation\n\n")

    report.write("SOURCE-WISE SPLIT DETAILS\n")
    report.write("=" * 90 + "\n")

    for source, info in source_reports.items():

        total_images = info["total_images"]

        train_percent = (info["train_images"] / total_images) * 100 if total_images > 0 else 0
        val_percent = (info["val_images"] / total_images) * 100 if total_images > 0 else 0
        test_percent = (info["test_images"] / total_images) * 100 if total_images > 0 else 0

        report.write(f"\n{source}\n")
        report.write("-" * 90 + "\n")
        report.write(f"Total images     : {info['total_images']}\n")
        report.write(f"Defect images    : {info['defect_images']}\n")
        report.write(f"No-defect images : {info['no_defect_images']}\n")
        report.write(f"Train images     : {info['train_images']} ({train_percent:.2f}%)\n")
        report.write(f"Val images       : {info['val_images']} ({val_percent:.2f}%)\n")
        report.write(f"Test images      : {info['test_images']} ({test_percent:.2f}%)\n")

    report.write("\n\nFINAL SPLIT COUNT\n")
    report.write("=" * 90 + "\n")

    for split in SPLITS:
        s = final_summary[split]
        report.write(f"\n{split.upper()} SET\n")
        report.write("-" * 90 + "\n")
        report.write(f"Total images     : {s['total']}\n")
        report.write(f"Defect images    : {s['defect']}\n")
        report.write(f"No-defect images : {s['no_defect']}\n")
        report.write(f"Missing labels   : {s['missing']}\n")
        report.write(f"Split percentage : {s['percentage']:.2f}%\n")

print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
print(f"Output dataset saved to : {OUTPUT_DATASET_DIR}")
print(f"data.yaml saved to      : {data_yaml_path}")
print(f"Split report saved to   : {report_path}")
print(f"JSON report saved to    : {json_report_path}")
print("=" * 90)

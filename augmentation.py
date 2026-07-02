# Applying dataset augmentation using horizontal flip combined with brightness adjustment.


import os
import shutil
from PIL import Image, ImageEnhance


INPUT_DIR = r"C:\Users\ANTONETTE\thesis_yolo\net_dataset_split_70_15_15"

OUTPUT_DIR = r"C:\Users\ANTONETTE\thesis_yolo\net_dataset_split_70_15_15_augmented_all"

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

BRIGHTNESS_FACTOR = 1.10

AUG_SUFFIX = "_flip_bright"



if os.path.exists(OUTPUT_DIR):
    print(f"[INFO] Removing old augmented dataset folder: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)

for split in SPLITS:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)



original_count_by_split = {}
output_count_by_split = {}

original_defect_by_split = {}
original_no_defect_by_split = {}

output_defect_by_split = {}
output_no_defect_by_split = {}

error_records = []



def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def read_yolo_label(label_path):
    """
    Reads YOLO label file.
    Empty label file means no-defect/background image.
    """

    if not os.path.exists(label_path):
        return []

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    labels = []

    for line in lines:
        parts = line.split()

        if len(parts) >= 5:
            labels.append(parts)

    return labels


def write_yolo_label(label_path, labels):
    """
    Writes YOLO labels.
    If labels list is empty, creates an empty .txt file.
    """

    with open(label_path, "w", encoding="utf-8") as f:
        for parts in labels:
            f.write(" ".join(parts) + "\n")


def flip_yolo_labels_horizontally(labels):
    """
    Horizontal flip for YOLO labels.

    YOLO format:
    class_id x_center y_center width height

    For horizontal flip:
    new_x_center = 1 - old_x_center
    """

    flipped_labels = []

    for parts in labels:
        new_parts = parts.copy()

        try:
            x_center = float(parts[1])
            new_x_center = 1.0 - x_center
            new_parts[1] = f"{new_x_center:.6f}"

        except Exception:
            print(f"[WARNING] Could not flip label line: {' '.join(parts)}")

        flipped_labels.append(new_parts)

    return flipped_labels


def apply_brightness(image):
    """
    Mild brightness adjustment only.
    This does not change bounding box coordinates.
    """

    return ImageEnhance.Brightness(image).enhance(BRIGHTNESS_FACTOR)


def count_label_type(label_path):
    """
    Returns defect if label is non-empty.
    Returns no_defect if label is empty.
    """

    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
        return "defect"
    return "no_defect"



print("=" * 90)
print("OFFLINE AUGMENTATION AFTER SPLITTING: TRAIN, VAL, AND TEST")
print("=" * 90)
print(f"Input split dataset  : {INPUT_DIR}")
print(f"Output dataset       : {OUTPUT_DIR}")
print(f"Brightness factor    : {BRIGHTNESS_FACTOR}")
print("Augmentations per image:")
print("1. Original copy")
print("2. Horizontal flip + mild brightness adjustment")
print("=" * 90)

for split in SPLITS:

    input_image_dir = os.path.join(INPUT_DIR, split, "images")
    input_label_dir = os.path.join(INPUT_DIR, split, "labels")

    output_image_dir = os.path.join(OUTPUT_DIR, split, "images")
    output_label_dir = os.path.join(OUTPUT_DIR, split, "labels")

    if not os.path.exists(input_image_dir):
        raise FileNotFoundError(f"Missing image folder: {input_image_dir}")

    if not os.path.exists(input_label_dir):
        raise FileNotFoundError(f"Missing label folder: {input_label_dir}")

    image_files = [
        f for f in os.listdir(input_image_dir)
        if is_image_file(f)
    ]

    image_files.sort()

    original_count_by_split[split] = len(image_files)
    output_count_by_split[split] = 0

    original_defect_by_split[split] = 0
    original_no_defect_by_split[split] = 0

    output_defect_by_split[split] = 0
    output_no_defect_by_split[split] = 0

    print("\n" + "-" * 90)
    print(f"Processing split: {split}")
    print(f"Original images : {len(image_files)}")
    print("-" * 90)

    for filename in image_files:

        image_path = os.path.join(input_image_dir, filename)

        base_name, ext = os.path.splitext(filename)
        label_name = base_name + ".txt"
        label_path = os.path.join(input_label_dir, label_name)

        if not os.path.exists(label_path):
            error_records.append((split, filename, "Missing label file"))
            print(f"[ERROR] Missing label file: {split}/{filename}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            labels = read_yolo_label(label_path)

            label_type = count_label_type(label_path)

            if label_type == "defect":
                original_defect_by_split[split] += 1
            else:
                original_no_defect_by_split[split] += 1



            original_output_image_path = os.path.join(output_image_dir, filename)
            original_output_label_path = os.path.join(output_label_dir, label_name)

            shutil.copy2(image_path, original_output_image_path)
            shutil.copy2(label_path, original_output_label_path)

            output_count_by_split[split] += 1

            if label_type == "defect":
                output_defect_by_split[split] += 1
            else:
                output_no_defect_by_split[split] += 1



            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_bright_image = apply_brightness(flipped_image)

            flipped_labels = flip_yolo_labels_horizontally(labels)

            augmented_image_name = f"{base_name}{AUG_SUFFIX}{ext}"
            augmented_label_name = f"{base_name}{AUG_SUFFIX}.txt"

            augmented_image_path = os.path.join(output_image_dir, augmented_image_name)
            augmented_label_path = os.path.join(output_label_dir, augmented_label_name)

            flipped_bright_image.save(augmented_image_path, quality=95)
            write_yolo_label(augmented_label_path, flipped_labels)

            output_count_by_split[split] += 1

            if label_type == "defect":
                output_defect_by_split[split] += 1
            else:
                output_no_defect_by_split[split] += 1

        except Exception as e:
            error_records.append((split, filename, str(e)))
            print(f"[ERROR] {split}/{filename}: {e}")


data_yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")

with open(data_yaml_path, "w", encoding="utf-8") as f:
    f.write("train: train/images\n")
    f.write("val: val/images\n")
    f.write("test: test/images\n\n")
    f.write("nc: 1\n")
    f.write("names: ['defect']\n")


print("\n" + "=" * 90)
print("AUGMENTATION SUMMARY BY SPLIT")
print("=" * 90)

total_original = 0
total_output = 0

for split in SPLITS:

    original = original_count_by_split[split]
    output = output_count_by_split[split]

    total_original += original
    total_output += output

    multiplier = round(output / original, 2) if original > 0 else 0

    print(f"\n{split.upper()} SET")
    print("-" * 90)
    print(f"Original images          : {original}")
    print(f"Output images            : {output}")
    print(f"Multiplier               : {multiplier}x")
    print(f"Original defect images   : {original_defect_by_split[split]}")
    print(f"Original no-defect images: {original_no_defect_by_split[split]}")
    print(f"Output defect images     : {output_defect_by_split[split]}")
    print(f"Output no-defect images  : {output_no_defect_by_split[split]}")

print("\n" + "=" * 90)
print("OVERALL SUMMARY")
print("=" * 90)
print(f"Total original split images : {total_original}")
print(f"Total augmented output      : {total_output}")
print(f"Expected multiplier         : {round(total_output / total_original, 2) if total_original > 0 else 0}x")
print(f"Errors                      : {len(error_records)}")
print(f"data.yaml saved to          : {data_yaml_path}")
print("=" * 90)


report_path = os.path.join(OUTPUT_DIR, "offline_augmentation_all_splits_report.txt")

with open(report_path, "w", encoding="utf-8") as report:

    report.write("OFFLINE AUGMENTATION REPORT: ALL SPLITS\n")
    report.write("=" * 90 + "\n\n")

    report.write(f"Input split dataset : {INPUT_DIR}\n")
    report.write(f"Output dataset      : {OUTPUT_DIR}\n")
    report.write(f"Brightness factor   : {BRIGHTNESS_FACTOR}\n\n")

    report.write("Process:\n")
    report.write("The dataset was first split into train, validation, and test sets.\n")
    report.write("Offline augmentation was then applied separately to each split.\n")
    report.write("This prevents original images and their augmented versions from crossing into another split.\n\n")

    report.write("Augmentations applied per image:\n")
    report.write("1. Original copy\n")
    report.write("2. Horizontal flip + mild brightness adjustment\n\n")

    report.write("Label transformation:\n")
    report.write("For horizontally flipped images, YOLO x_center was updated using new_x_center = 1 - old_x_center.\n")
    report.write("Brightness adjustment does not change label coordinates.\n\n")

    report.write("AUGMENTATION SUMMARY BY SPLIT\n")
    report.write("=" * 90 + "\n")

    for split in SPLITS:

        original = original_count_by_split[split]
        output = output_count_by_split[split]
        multiplier = round(output / original, 2) if original > 0 else 0

        report.write(f"\n{split.upper()} SET\n")
        report.write("-" * 90 + "\n")
        report.write(f"Original images          : {original}\n")
        report.write(f"Output images            : {output}\n")
        report.write(f"Multiplier               : {multiplier}x\n")
        report.write(f"Original defect images   : {original_defect_by_split[split]}\n")
        report.write(f"Original no-defect images: {original_no_defect_by_split[split]}\n")
        report.write(f"Output defect images     : {output_defect_by_split[split]}\n")
        report.write(f"Output no-defect images  : {output_no_defect_by_split[split]}\n")

    report.write("\n" + "=" * 90 + "\n")
    report.write("OVERALL SUMMARY\n")
    report.write("=" * 90 + "\n")
    report.write(f"Total original split images : {total_original}\n")
    report.write(f"Total augmented output      : {total_output}\n")
    report.write(f"Expected multiplier         : {round(total_output / total_original, 2) if total_original > 0 else 0}x\n")
    report.write(f"Errors                      : {len(error_records)}\n\n")

    if error_records:
        report.write("ERROR RECORDS\n")
        report.write("=" * 90 + "\n")

        for split, filename, error in error_records:
            report.write(f"{split}/{filename}: {error}\n")


print("\n" + "=" * 90)
print("AUGMENTATION COMPLETE")
print("=" * 90)
print(f"Output dataset saved to : {OUTPUT_DIR}")
print(f"Report saved to         : {report_path}")
print("=" * 90)

# Performing label validation by checking if all images have corresponding labels and if the labels are created properly.

import os
from PIL import Image



DATASET_DIR = r"C:\Users\ANTONETTE\thesis_yolo\test2"

SPLITS = ["test"]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

NC = 1  # number of classes
CLASS_NAMES = ["defect"]

REPORT_NAME = "label_validation_report.txt"


total_images = 0
total_labels = 0
total_boxes = 0
empty_label_files = 0

missing_label_records = []
extra_label_records = []
invalid_label_records = []
corrupt_image_records = []
size_records = []

split_summary = {}



def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def validate_yolo_line(line, label_path, line_number):
    """
    Validates one YOLO annotation line.

    Expected format:
    class_id x_center y_center width height
    """

    parts = line.strip().split()

    if len(parts) != 5:
        return False, f"Line {line_number}: Expected 5 values, got {len(parts)}"

    try:
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
    except ValueError:
        return False, f"Line {line_number}: Contains non-numeric value"

    if class_id < 0 or class_id >= NC:
        return False, f"Line {line_number}: Invalid class_id {class_id}"

    if not (0.0 <= x_center <= 1.0):
        return False, f"Line {line_number}: x_center out of range: {x_center}"

    if not (0.0 <= y_center <= 1.0):
        return False, f"Line {line_number}: y_center out of range: {y_center}"

    if not (0.0 < width <= 1.0):
        return False, f"Line {line_number}: width out of range: {width}"

    if not (0.0 < height <= 1.0):
        return False, f"Line {line_number}: height out of range: {height}"

    x_min = x_center - width / 2
    x_max = x_center + width / 2
    y_min = y_center - height / 2
    y_max = y_center + height / 2

    if x_min < 0 or x_max > 1 or y_min < 0 or y_max > 1:
        return False, (
            f"Line {line_number}: Bounding box exceeds image boundary "
            f"(x_min={x_min:.4f}, x_max={x_max:.4f}, "
            f"y_min={y_min:.4f}, y_max={y_max:.4f})"
        )

    return True, "OK"


print("=" * 90)
print("YOLO LABEL VALIDATION AFTER AUGMENTATION")
print("=" * 90)
print(f"Dataset folder : {DATASET_DIR}")
print(f"Classes        : {CLASS_NAMES}")
print("=" * 90)

for split in SPLITS:

    image_dir = os.path.join(DATASET_DIR, split, "images")
    label_dir = os.path.join(DATASET_DIR, split, "labels")

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Missing image folder: {image_dir}")

    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"Missing label folder: {label_dir}")

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if is_image_file(f)
    ])

    label_files = sorted([
        f for f in os.listdir(label_dir)
        if f.lower().endswith(".txt")
    ])

    image_base_names = {
        os.path.splitext(f)[0] for f in image_files
    }

    label_base_names = {
        os.path.splitext(f)[0] for f in label_files
    }

    split_images = len(image_files)
    split_labels = len(label_files)
    split_boxes = 0
    split_empty_labels = 0
    split_invalid_labels = 0
    split_missing_labels = 0
    split_corrupt_images = 0

    print("\n" + "-" * 90)
    print(f"Validating split: {split}")
    print("-" * 90)


    for image_file in image_files:
        image_base = os.path.splitext(image_file)[0]
        label_file = image_base + ".txt"

        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)


        try:
            with Image.open(image_path) as img:
                width, height = img.size
                size_records.append((split, image_file, width, height))
        except Exception as e:
            corrupt_image_records.append((split, image_file, str(e)))
            split_corrupt_images += 1
            continue

        if not os.path.exists(label_path):
            missing_label_records.append((split, image_file, label_file))
            split_missing_labels += 1
            continue


        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if len(lines) == 0:
            split_empty_labels += 1
            continue

        for line_number, line in enumerate(lines, start=1):
            is_valid, message = validate_yolo_line(line, label_path, line_number)

            if is_valid:
                split_boxes += 1
            else:
                invalid_label_records.append((split, label_file, line, message))
                split_invalid_labels += 1



    extra_labels = label_base_names - image_base_names

    for base_name in sorted(extra_labels):
        extra_label_records.append((split, base_name + ".txt"))



    split_summary[split] = {
        "images": split_images,
        "labels": split_labels,
        "boxes": split_boxes,
        "empty_labels": split_empty_labels,
        "missing_labels": split_missing_labels,
        "extra_labels": len(extra_labels),
        "invalid_labels": split_invalid_labels,
        "corrupt_images": split_corrupt_images,
    }

    total_images += split_images
    total_labels += split_labels
    total_boxes += split_boxes
    empty_label_files += split_empty_labels

    print(f"Images              : {split_images}")
    print(f"Label files         : {split_labels}")
    print(f"Bounding boxes      : {split_boxes}")
    print(f"Empty label files   : {split_empty_labels}")
    print(f"Missing labels      : {split_missing_labels}")
    print(f"Extra label files   : {len(extra_labels)}")
    print(f"Invalid labels      : {split_invalid_labels}")
    print(f"Corrupt images      : {split_corrupt_images}")


print("\n" + "=" * 90)
print("FINAL VALIDATION SUMMARY")
print("=" * 90)

for split, summary in split_summary.items():
    print(f"\n{split.upper()} SET")
    print("-" * 90)
    print(f"Images            : {summary['images']}")
    print(f"Label files       : {summary['labels']}")
    print(f"Bounding boxes    : {summary['boxes']}")
    print(f"Empty labels      : {summary['empty_labels']}")
    print(f"Missing labels    : {summary['missing_labels']}")
    print(f"Extra labels      : {summary['extra_labels']}")
    print(f"Invalid labels    : {summary['invalid_labels']}")
    print(f"Corrupt images    : {summary['corrupt_images']}")

print("\n" + "-" * 90)
print("OVERALL")
print("-" * 90)
print(f"Total images          : {total_images}")
print(f"Total label files     : {total_labels}")
print(f"Total bounding boxes  : {total_boxes}")
print(f"Empty label files     : {empty_label_files}")
print(f"Missing labels        : {len(missing_label_records)}")
print(f"Extra label files     : {len(extra_label_records)}")
print(f"Invalid labels        : {len(invalid_label_records)}")
print(f"Corrupt images        : {len(corrupt_image_records)}")


report_path = os.path.join(DATASET_DIR, REPORT_NAME)

with open(report_path, "w", encoding="utf-8") as report:

    report.write("YOLO LABEL VALIDATION REPORT AFTER AUGMENTATION\n")
    report.write("=" * 90 + "\n\n")

    report.write(f"Dataset folder : {DATASET_DIR}\n")
    report.write(f"Classes        : {CLASS_NAMES}\n\n")

    report.write("SUMMARY BY SPLIT\n")
    report.write("=" * 90 + "\n")

    for split, summary in split_summary.items():
        report.write(f"\n{split.upper()} SET\n")
        report.write("-" * 90 + "\n")
        report.write(f"Images            : {summary['images']}\n")
        report.write(f"Label files       : {summary['labels']}\n")
        report.write(f"Bounding boxes    : {summary['boxes']}\n")
        report.write(f"Empty labels      : {summary['empty_labels']}\n")
        report.write(f"Missing labels    : {summary['missing_labels']}\n")
        report.write(f"Extra labels      : {summary['extra_labels']}\n")
        report.write(f"Invalid labels    : {summary['invalid_labels']}\n")
        report.write(f"Corrupt images    : {summary['corrupt_images']}\n")

    report.write("\n\nOVERALL SUMMARY\n")
    report.write("=" * 90 + "\n")
    report.write(f"Total images          : {total_images}\n")
    report.write(f"Total label files     : {total_labels}\n")
    report.write(f"Total bounding boxes  : {total_boxes}\n")
    report.write(f"Empty label files     : {empty_label_files}\n")
    report.write(f"Missing labels        : {len(missing_label_records)}\n")
    report.write(f"Extra label files     : {len(extra_label_records)}\n")
    report.write(f"Invalid labels        : {len(invalid_label_records)}\n")
    report.write(f"Corrupt images        : {len(corrupt_image_records)}\n")

    if missing_label_records:
        report.write("\n\nMISSING LABELS\n")
        report.write("=" * 90 + "\n")
        for split, image_file, label_file in missing_label_records:
            report.write(f"{split}: image={image_file}, missing_label={label_file}\n")

    if extra_label_records:
        report.write("\n\nEXTRA LABEL FILES WITHOUT IMAGES\n")
        report.write("=" * 90 + "\n")
        for split, label_file in extra_label_records:
            report.write(f"{split}: {label_file}\n")

    if invalid_label_records:
        report.write("\n\nINVALID LABELS\n")
        report.write("=" * 90 + "\n")
        for split, label_file, line, message in invalid_label_records:
            report.write(f"{split}: {label_file} | {line} | {message}\n")

    if corrupt_image_records:
        report.write("\n\nCORRUPT IMAGES\n")
        report.write("=" * 90 + "\n")
        for split, image_file, error in corrupt_image_records:
            report.write(f"{split}: {image_file} | {error}\n")


print("\n" + "=" * 90)

if (
    len(missing_label_records) == 0 and
    len(extra_label_records) == 0 and
    len(invalid_label_records) == 0 and
    len(corrupt_image_records) == 0
):
    print("VALIDATION PASSED")
    print("All images have matching label files.")
    print("All YOLO label values are valid.")
    print("Empty labels are allowed for no-defect images.")
else:
    print("VALIDATION FAILED")
    print("Please check the report and fix the listed issues.")

print(f"Report saved to: {report_path}")
print("=" * 90)

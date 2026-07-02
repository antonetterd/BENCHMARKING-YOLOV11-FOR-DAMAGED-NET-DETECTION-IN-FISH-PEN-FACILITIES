# CLEANING DATASET BY REMOVING SIMILAR OR NEARLY SIMILAR FRAMES

import os
import shutil
from PIL import Image
import imagehash



DATASET_DIR = r"C:\Users\ANTONETTE\Videos\TEST_VIDEO"
OUTPUT_DIR = r"C:\Users\ANTONETTE\thesis_yolo\test_cleaned"

SOURCE_FOLDERS = [
    "defect",
    "no_defect",
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
HASH_THRESHOLD = 12

=

if os.path.exists(OUTPUT_DIR):
    print(f"[INFO] Removing old cleaned folder: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

kept_hashes = []
kept_records = []
removed_records = []
corrupt_records = []

original_count_by_folder = {}
cleaned_count_by_folder = {}
removed_count_by_folder = {}
corrupt_count_by_folder = {}


print("=" * 80)
print("RAW FRAME PERCEPTUAL HASH CLEANING")
print("=" * 80)
print(f"Dataset folder : {DATASET_DIR}")
print(f"Output folder  : {OUTPUT_DIR}")
print(f"Threshold      : {HASH_THRESHOLD}")
print("=" * 80)

def init_folder_counts(source):
    original_count_by_folder[source] = 0
    cleaned_count_by_folder[source] = 0
    removed_count_by_folder[source] = 0
    corrupt_count_by_folder[source] = 0


for source in SOURCE_FOLDERS:
    init_folder_counts(source)
    
    source_image_dir = os.path.join(DATASET_DIR, source)
    if not os.path.exists(source_image_dir):
        print(f"[WARNING] Folder does not exist, skipping: {source_image_dir}")
        continue

    output_image_dir = os.path.join(OUTPUT_DIR, source)
    os.makedirs(output_image_dir, exist_ok=True)

    image_files = [f for f in os.listdir(source_image_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
    image_files.sort()
    original_count_by_folder[source] = len(image_files)

    print("\n" + "-" * 80)
    print(f"Processing folder: {source}")
    print(f"Original images : {len(image_files)}")
    print("-" * 80)

    for filename in image_files:
        image_path = os.path.join(source_image_dir, filename)


        try:
            img = Image.open(image_path).convert("RGB")
            img_hash = imagehash.phash(img)
        except Exception:
            corrupt_records.append((source, filename))
            corrupt_count_by_folder[source] += 1
            print(f"[CORRUPT - SKIPPED] {source}/{filename}")
            continue

 
        is_duplicate = False
        matched_record = None

        for existing_hash, existing_record in kept_hashes:
            hash_difference = img_hash - existing_hash
            if hash_difference <= HASH_THRESHOLD:
                is_duplicate = True
                matched_record = existing_record
                break

        if is_duplicate:
            removed_records.append((source, filename, matched_record))
            removed_count_by_folder[source] += 1
            print(f"[REMOVED SIMILAR] {source}/{filename} similar to {matched_record}")
            continue


        kept_hashes.append((img_hash, f"{source}/{filename}"))
        kept_records.append((source, filename))
        cleaned_count_by_folder[source] += 1

        shutil.copy2(image_path, os.path.join(output_image_dir, filename))



total_original = sum(original_count_by_folder.values())
total_cleaned = sum(cleaned_count_by_folder.values())
total_removed = sum(removed_count_by_folder.values())
total_corrupt = sum(corrupt_count_by_folder.values())



print("\n" + "=" * 80)
print("ORIGINAL VS CLEANED DATASET COUNT")
print("=" * 80)

for source in SOURCE_FOLDERS:
    print(f"\n{source}")
    print(f"  Original images          : {original_count_by_folder.get(source, 0)}")
    print(f"  Cleaned/kept images      : {cleaned_count_by_folder.get(source, 0)}")
    print(f"  Removed near-duplicates  : {removed_count_by_folder.get(source, 0)}")
    print(f"  Corrupt images skipped   : {corrupt_count_by_folder.get(source, 0)}")

print("\n" + "-" * 80)
print("OVERALL TOTAL")
print("-" * 80)
print(f"Original images          : {total_original}")
print(f"Cleaned/kept images      : {total_cleaned}")
print(f"Removed near-duplicates  : {total_removed}")
print(f"Corrupt images skipped   : {total_corrupt}")
print("=" * 80)


report_path = os.path.join(OUTPUT_DIR, "near_duplicate_removal_report.txt")

with open(report_path, "w", encoding="utf-8") as report:
    report.write("SOURCE DATASET NEAR-DUPLICATE CLEANING REPORT\n")
    report.write("=" * 80 + "\n\n")
    report.write(f"Dataset folder : {DATASET_DIR}\n")
    report.write(f"Output folder  : {OUTPUT_DIR}\n")
    report.write(f"Hash threshold : {HASH_THRESHOLD}\n\n")
    
    for source in SOURCE_FOLDERS:
        report.write(f"\n{source}\n")
        report.write(f"  Original images          : {original_count_by_folder.get(source, 0)}\n")
        report.write(f"  Cleaned/kept images      : {cleaned_count_by_folder.get(source, 0)}\n")
        report.write(f"  Removed near-duplicates  : {removed_count_by_folder.get(source, 0)}\n")
        report.write(f"  Corrupt images skipped   : {corrupt_count_by_folder.get(source, 0)}\n")

    report.write("\nREMOVED NEAR-DUPLICATES\n")
    report.write("=" * 80 + "\n")
    for source, filename, matched_record in removed_records:
        report.write(f"{source}/{filename} similar to {matched_record}\n")

print(f"\n[SUCCESS] Cleaning complete! Report saved to: {report_path}")

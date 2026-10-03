
import os
import shutil

IMAGE_DIR = "Training/raw_frames"
LABEL_DIR = "auto_labels"
OUTPUT_DIR = "Training/dataset"

pairs = []
for img in sorted(os.listdir(IMAGE_DIR)):
    if not img.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    label_name = os.path.splitext(img)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_name)
    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
        pairs.append((img, label_name))

pairs.sort()

n = len(pairs)
n_train = int(n * 0.7)
n_val = int(n * 0.15)

splits = {
    "train": pairs[:n_train],
    "val": pairs[n_train:n_train + n_val],
    "test": pairs[n_train + n_val:],
}

for split, items in splits.items():
    img_out = os.path.join(OUTPUT_DIR, "images", split)
    lbl_out = os.path.join(OUTPUT_DIR, "labels", split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    for img, lbl in items:
        shutil.copy(os.path.join(IMAGE_DIR, img), os.path.join(img_out, img))
        shutil.copy(os.path.join(LABEL_DIR, lbl), os.path.join(lbl_out, lbl))
    print(f"{split}: {len(items)} images")

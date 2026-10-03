from ultralytics import YOLO
import os
import cv2

MODEL_PATH = "models/"     
IMAGE_DIR = "Training/raw_frames"              
OUTPUT_DIR = "auto_labels"           
CONF_THRESHOLD = 0.25              
CLASS_ID = 0                          

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

image_files = [f for f in os.listdir(IMAGE_DIR)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print(f"Found {len(image_files)} images")

total_boxes = 0
labeled = 0

for img_name in image_files:
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    results = model(img_path, conf=CONF_THRESHOLD, verbose=False)

    label_path = os.path.join(OUTPUT_DIR, os.path.splitext(img_name)[0] + ".txt")

    with open(label_path, "w") as f:
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x_c, y_c, bw, bh = box.xywhn[0].tolist()
                conf = float(box.conf[0])

                f.write(f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

                total_boxes += 1
            labeled += 1

print(f"Done.")

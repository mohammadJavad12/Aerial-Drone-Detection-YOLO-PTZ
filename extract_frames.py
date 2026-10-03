import cv2
import os
import numpy as np

VIDEO_PATH = "QuadCopter.mp4"
OUTPUT_DIR = "raw_frames_test"
TARGET_FPS = 3
SIMILARITY_THRESHOLD = 0.92 
os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
interval = max(1, int(fps / TARGET_FPS))

last_kept = None
saved = 0
skipped = 0
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % interval == 0:
        thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36))
        
        if last_kept is not None:
            diff = np.mean(np.abs(thumb.astype(float) - last_kept.astype(float)))
            similarity = 1 - (diff / 255.0)
            if similarity > SIMILARITY_THRESHOLD:
                skipped += 1
                frame_idx += 1
                continue
        
        path = os.path.join(OUTPUT_DIR, f"frame_{saved:04d}.jpg")
        cv2.imwrite(path, frame)
        last_kept = thumb
        saved += 1
    frame_idx += 1

cap.release()
print(f"Saved {saved} frames AND skipped {skipped} near-duplicates")
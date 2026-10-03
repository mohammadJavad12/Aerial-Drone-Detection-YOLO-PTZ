from ultralytics import YOLO
import cv2
import os
import time
if __name__ == '__main__':
    VIDEO_PATH = "QuadCopter.mp4"
    MODEL_PATH = "models/640_Run/best.pt"
    CONF = 0.4

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / src_fps

    prev_time = time.time()

    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, conf=CONF, verbose=True)[0]
        annotated = results.plot()

        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Video Playback", annotated)

        elapsed = time.time() - loop_start
        wait_ms = max(1, int((frame_delay - elapsed) * 1000))

        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
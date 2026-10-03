from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("models/640_Run/best.pt")
    m = model.val(
        data="Training/dataset_simulation/data.yaml",
        split="train",
        imgsz=640,
        workers=0,
        device=0,
        name="EVAL_640",
    )
    print(f"mAP@0.5:      {m.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {m.box.map:.4f}")
    print(f"Precision:    {m.box.mp:.4f}")
    print(f"Recall:       {m.box.mr:.4f}")


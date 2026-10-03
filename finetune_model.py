from ultralytics import YOLO

model = YOLO("yolo11n_drone.pt")   

model.train(
    data="Training/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=32,
    freeze=8,
    lr0=0.001,
    patience=15,
    workers=0,
    device=0,
    project="dt_for_drone_training",
    name="640_Run",
)
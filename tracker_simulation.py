import select
import socket, cv2, numpy as np
import math
from ultralytics import YOLO
import supervision as sv
import time
import csv

current_pan = 0
current_tilt = 0

HORIZON_TILT = 0
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

y = YOLO("yolo11n_drone.pt")
b = sv.BoxAnnotator()
l = sv.LabelAnnotator()

CAMERA_IP = "127.0.0.1"
UDP_PORT_PAN_TILT = 8888
PAN_AND_TILT_GAIN = 0.3


MAX_ROT_DELTA = 8        
TILT_FORCE_PAN_THRESHOLD = 60.0  
PAN_SUPPRESS_MIN_SCALE = 0.1      

last_pan_dir = 0         
pan_momentum_frames = 0   
PAN_MOMENTUM_DURATION = 15  
PAN_MOMENTUM_STEP = 3.0   


target_id = 0

lost_frames = 0
LOST_FRAME_THRESHOLD = 20
status = "Search"

FOV = 99
F_W, F_H = 1280,720

ZOOM_SCALE_ON_DETECTED  = 0.5   

current_zoom = 1.0
TARGET_ZOOM = 2.2
ZOOM_AC = 0.03
zoom_center = None

log_rows = []
start_time = time.time()
scenario_name = "Moving_NearCenter"

fps_times = []
last_print = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer= cv2.VideoWriter(f"logs/{scenario_name}.avi",fourcc,15,(F_W,F_H))
def pan_suppression_scale():
    elevation = abs(current_tilt - HORIZON_TILT)  
    frac = clamp(elevation / TILT_FORCE_PAN_THRESHOLD, 0.0, 1.0)
    return 1.0 - frac * (1.0 - PAN_SUPPRESS_MIN_SCALE)

def update_zoom(locked=False):
    global current_zoom
    if locked:
        current_zoom += (TARGET_ZOOM - current_zoom) * ZOOM_AC
    else:
        current_zoom += (1.0 - current_zoom) * ZOOM_AC
        if current_zoom < 1.02:
            current_zoom = 1.0
    current_zoom = clamp(current_zoom, 1.0, TARGET_ZOOM)
    return current_zoom

def Get_fov(fov_deg, scale):
    half_angle = math.radians(fov_deg) / 2
    new_half_angle = math.atan(scale * math.tan(half_angle))
    return math.degrees(new_half_angle) * 2

DETECT_FOV = Get_fov(FOV, ZOOM_SCALE_ON_DETECTED)

def __zoom(img, scale, center=None):
    height, width = img.shape[:2]
    if center is None:
        center_x = int(width / 2)
        center_y = int(height / 2)
        radius_x, radius_y = int(width / 2), int(height / 2)
    else:
        rate = height / width
        center_x, center_y = center

        if center_x < width * (1-rate):
            center_x = width * (1-rate)
        elif center_x > width * rate:
            center_x = width * rate
        if center_y < height * (1-rate):
            center_y = height * (1-rate)
        elif center_y > height * rate:
            center_y = height * rate

        center_x, center_y = int(center_x), int(center_y)
        left_x, right_x = center_x, int(width - center_x)
        up_y, down_y = int(height - center_y), center_y
        radius_x = min(left_x, right_x)
        radius_y = min(up_y, down_y)

    radius_x, radius_y = int(scale * radius_x), int(scale * radius_y)

    min_x, max_x = center_x - radius_x, center_x + radius_x
    min_y, max_y = center_y - radius_y, center_y + radius_y

    cropped = img[min_y:max_y, min_x:max_x]


    new_cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LANCZOS4)


    # new_cropped = cv2.bilateralFilter(new_cropped, 5, 30, 30)
    return new_cropped


sz = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sz.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
sz.bind(('127.0.0.1', 5006))
sz.settimeout(0.01)


state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
state_socket.bind(('127.0.0.1', 8889))
state_socket.settimeout(0.01)


def receive_ptz_state():
    global current_pan, current_tilt
    latest = None
    while True:
        try:
            latest, _ = state_socket.recvfrom(1024)
        except (socket.timeout, BlockingIOError):
            break
    if latest is None:
        return
    msg = latest.decode().strip().split()
    if len(msg) == 3 and msg[0] == "STATE":
        current_pan = float(msg[1])
        current_tilt = float(msg[2])


def get_udp_frame(sock):
    ready, _, _ = select.select([sock], [], [], 0)
    if not ready:
        return None
    
    latest_data = None
    while True:
        ready, _, _ = select.select([sock], [], [], 0)
        if not ready:
            break
        try:
            data, _ = sock.recvfrom(65535)
            latest_data = data
        except (socket.timeout, BlockingIOError):
            break
    
    if latest_data is None or len(latest_data) < 4:
        return None
    
    jpeg_data = latest_data[4:]
    img = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img                                   

def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def compute_pan_tilt(bcx, bcy, frame_w, frame_h, fov_deg):
    u = ((2 * bcx) / frame_w) - 1  
    v = 1 - ((2 * bcy) / frame_h)   
    aspect = frame_w / frame_h

    fov = math.radians(fov_deg)
    half_tan = math.tan(fov / 2)

    x = u * aspect * half_tan
    y_ = v * half_tan
    z = 1.0

    pan = math.degrees(math.atan2(x, z))
    tilt = math.degrees(math.atan2(y_, math.sqrt(x * x + z * z)))

    return pan, tilt, v

def save_log(target_idx, boxes):
    global log_rows, target_id

    row = {
        "frame": len(log_rows),
        "time": round(time.time() - start_time, 3),
        "detected": 0,
        "target_id": None,
        "pixel_error": None,
        "angular_error": None,
    }

    if target_idx is not None:
        box = boxes[target_idx]

        bcx = box.xywh[0, 0].item()
        bcy = box.xywh[0, 1].item()

        dx = bcx - F_W / 2
        dy = bcy - F_H / 2

        pixel_err = math.sqrt(dx**2 + dy**2)
        angular_err = (pixel_err / F_W) * FOV

        row["detected"] = 1
        row["target_id"] = target_id
        row["pixel_error"] = round(pixel_err, 3)
        row["angular_error"] = round(angular_err, 3)

    elif len(boxes) > 0:
        row["detected"] = 1

    log_rows.append(row)

def Track():
    global status, lost_frames,target_id

    img = get_udp_frame(sz)
    if img is None:
        return None

    zoom_scale = current_zoom
    if zoom_scale > 1.0:
        img = __zoom(img, 1.0 / zoom_scale, center=None)

    detect_fov = Get_fov(FOV, 1.0 / zoom_scale) if zoom_scale > 1.0 else FOV
    
    for r in y.track(img, device='cuda', verbose=False,
                      tracker="botsort.yaml", conf=0.4):
        d = sv.Detections.from_ultralytics(r)
        annotated = b.annotate(img.copy(), d)
        annotated = l.annotate(
            annotated, d,
            [f"{y.names[c]} {conf:.2f}" for c, conf in zip(d.class_id, d.confidence)]
        )
        



        boxes = r.cpu().boxes
        ids = r.cpu().boxes.id

        if len(boxes) > 0 and ids is not None:
            target_idx = None

            if target_id is None:
                best_idx = int(boxes.conf.argmax())
                target_id = int(ids[best_idx])

            for i, tid in enumerate(ids):
                if int(tid) == target_id:
                    target_idx = i
                    break

            if target_idx is not None:
                status = "Locked"
                lost_frames = 0

                box = boxes[target_idx]
                bcx = box.xywh[0, 0].item()
                bcy = box.xywh[0, 1].item()

                annotated = sv.draw_line(annotated,sv.Point(F_W/2 , F_H/2) ,sv.Point(bcx,bcy),sv.Color.GREEN )

                pan, tilt, v = compute_pan_tilt(
                    bcx, bcy, F_W, F_H, detect_fov
                )

                pan_scale = pan_suppression_scale()

                delta_pan = clamp(
                    pan * PAN_AND_TILT_GAIN * pan_scale,
                    -MAX_ROT_DELTA, MAX_ROT_DELTA
                )

                delta_tilt = clamp(
                    tilt * PAN_AND_TILT_GAIN,
                    -MAX_ROT_DELTA, MAX_ROT_DELTA
                )
                
                command = f"PAN_TILT {delta_pan:.2f} {delta_tilt:.2f}\n"
                udp_socket.sendto(
                    command.encode(),
                    (CAMERA_IP, UDP_PORT_PAN_TILT)
                )
                if(pan < abs(12) and tilt < abs(12)):
                    zoom_scale = update_zoom(locked=True)

            else:
                lost_frames += 1
                zoom_scale = update_zoom(locked=False)

            save_log(target_idx, boxes)
        else:
            lost_frames += 1
            zoom_scale = update_zoom(locked=False)
            
        
        if lost_frames >= LOST_FRAME_THRESHOLD:
            target_id = None
            status = "Return"
        video_writer.write(annotated)
        return annotated
    
    
    video_writer.write(img)
    zoom_scale = update_zoom(locked=False)
    return img

def ReturnHome():
    global status,lost_frames
    if abs(current_pan) >= 3 or abs(current_tilt) >= 3:
        delta_pan = clamp(-current_pan * PAN_AND_TILT_GAIN / 3, -MAX_ROT_DELTA, MAX_ROT_DELTA)
        delta_tilt = clamp(current_tilt * PAN_AND_TILT_GAIN / 3, -MAX_ROT_DELTA, MAX_ROT_DELTA)
        command = f"PAN_TILT {delta_pan:.2f} {delta_tilt:.2f}\n"
        udp_socket.sendto(command.encode(), (CAMERA_IP, UDP_PORT_PAN_TILT))
    else:
        status = "Search"
        lost_frames = 0


try:
    while True:
        receive_ptz_state()
        fz = Track()
        
        if status == "Return":
            ReturnHome()
        
        if fz is not None:
            cv2.imshow('YOLO-Track', fz)
        if cv2.waitKey(1) & 255 == ord('q'):
            break
finally:
    # Save log regardless of how we exited
    if log_rows:
        with open(f"logs/{scenario_name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=["frame", "time", "detected", "target_id", "pixel_error","angular_error"]
            )
            writer.writeheader()
            writer.writerows(log_rows)


cv2.destroyAllWindows()

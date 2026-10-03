import socket, cv2, numpy as np
import math
from ultralytics import YOLO
import supervision as sv
import time
import serial

current_pan = 66
current_tilt = 145
HOME_PAN  = 66
HOME_TILT = 145


SERIAL_PORT = "COM6"         
BAUD        = 115200

ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.01)
time.sleep(2)             
ser.reset_input_buffer()


# udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

y = YOLO("runs/detect/dt_for_drone_training/SimulationRun_2/weights/best.pt")
b = sv.BoxAnnotator()
l = sv.LabelAnnotator()

CAMERA_IP = "127.0.0.1"
UDP_PORT_PAN_TILT = 8888
PAN_AND_TILT_GAIN = 0.05


MAX_ROT_DELTA = 8.0          
TILT_FORCE_PAN_THRESHOLD = 60.0  
PAN_SUPPRESS_MIN_SCALE = 0.1      

target_id = 0

lost_frames = 0
LOST_FRAME_THRESHOLD = 70
status = "Search"

FOV = 60.0 
F_W, F_H = 1280, 720

ZOOM_SCALE_ON_DETECTED  = 0.7  

current_zoom = 1.0
TARGET_ZOOM = 1.3
ZOOM_AC = 0.03
zoom_center = None


C_I = 1               
C_W  = 1280
C_H = 720
C_FPS    = 60

last_send_time = 0
SEND_INTERVAL = 0.03

cap = cv2.VideoCapture(C_I, cv2.CAP_DSHOW)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  C_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C_H)
cap.set(cv2.CAP_PROP_FPS,C_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

def pan_suppression_scale():
    elevation = abs(current_tilt - HOME_TILT) 
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

def send_pan_tilt(dp, dt):
    global last_send_time
    try:
        now = time.time()
        if(now - last_send_time > SEND_INTERVAL):
            last_send_time = now
            ser.write(f"PAN_TILT {dp:.2f} {dt:.2f}\n".encode())
            ser.flush()
    except Exception as e:
        print("Serial write error:", e)

def clamp(value, lo, hi):
    return max(lo, min(hi, value))

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


# sz = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sz.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
# sz.bind(('127.0.0.1', 5006))
# sz.settimeout(0.01)


# state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# state_socket.bind(('127.0.0.1', 8889))
# state_socket.settimeout(0.01)



def receive_ptz_state():
    global current_pan, current_tilt
    latest = None
    try:
        while ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if line.startswith("STATE"):
                latest = line
    except Exception as e:
        print("Serial read error:", e)
        return

    if latest is None:
        return

    parts = latest.split()
    if len(parts) == 3:
        try:
            current_pan  = float(parts[1])
            current_tilt = float(parts[2])
        except ValueError:
            pass


def get_udp_frame(sock):
    latest_data = None
    while True:
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


def Track():
    global status, lost_frames,target_id

    ok, img = cap.read()
    if not ok or img is None:
        return None


    zoom_scale = current_zoom
    if zoom_scale > 1.0:
        img = __zoom(img, 1.0 / zoom_scale, center=None)

    detect_fov = Get_fov(FOV, 1.0 / zoom_scale) if zoom_scale > 1.0 else FOV

    for r in y.track(img, device='cuda', verbose=False,
                      tracker="custom_bytetrack.yaml", conf=0.4):
        d = sv.Detections.from_ultralytics(r)
        annotated = b.annotate(img.copy(), d)
        annotated = l.annotate(
            annotated, d,
            [f"{y.names[c]} {conf:.2f}" for c, conf in zip(d.class_id, d.confidence)]
        )

        boxes = r.cpu().boxes
        ids = r.cpu().boxes.id

        if len(boxes) > 0 and ids is not None:
            if target_id is None:
                best_idx = int(boxes.conf.argmax())
                target_id = int(ids[best_idx])
                lost_frames = 0
                status = "Locked"

            target_id_f = None
            for i, tid in enumerate(ids):
                if int(tid) == target_id:
                    target_id_f= i
                    break
            if(target_id_f is not None):    
                status = "Locked"
                lost_frames = 0

                box = boxes[target_id_f]
                bcx = box.xywh[0, 0].item()
                bcy = box.xywh[0, 1].item()
                
                pan, tilt, v = compute_pan_tilt(bcx, bcy, F_W, F_H, detect_fov)
            
                pan_scale = pan_suppression_scale()
                delta_pan  = clamp(pan  * PAN_AND_TILT_GAIN * pan_scale, -MAX_ROT_DELTA, MAX_ROT_DELTA)
                delta_tilt = clamp(tilt * PAN_AND_TILT_GAIN,             -MAX_ROT_DELTA, MAX_ROT_DELTA)

                # command = f"PAN_TILT {delta_pan:.2f} {delta_tilt:.2f}\n"
                # udp_socket.sendto(command.encode(), (CAMERA_IP, UDP_PORT_PAN_TILT))
                send_pan_tilt(delta_pan,delta_tilt)

                zoom_scale = update_zoom(locked=True)
            else:
                lost_frames += 1
                zoom_scale = update_zoom(locked=False)
        else:
            lost_frames += 1
            zoom_scale = update_zoom(locked=False)
            
        if lost_frames >= LOST_FRAME_THRESHOLD:
            target_id = None
            status = "Return"
        return annotated

    zoom_scale = update_zoom(locked=False)
    return img

def ReturnHome():
    global status
    err_pan  = HOME_PAN  - current_pan
    err_tilt = HOME_TILT - current_tilt

    if abs(err_pan) >= 2 or abs(err_tilt) >= 2:
        delta_pan  = clamp(err_pan  * PAN_AND_TILT_GAIN / 3, -MAX_ROT_DELTA, MAX_ROT_DELTA)
        delta_tilt = clamp(err_tilt * PAN_AND_TILT_GAIN / 3, -MAX_ROT_DELTA, MAX_ROT_DELTA)
        send_pan_tilt(delta_pan, delta_tilt)
    else:
        status = "Search"

while True:
    receive_ptz_state()
    fz = Track()

    if status == "Return":
        ReturnHome()

    if fz is not None:
        cv2.imshow('YOLO-Track', fz)

    if cv2.waitKey(1) & 255 == ord('q'):
        break

cv2.destroyAllWindows()
ser.close()
cap.release()
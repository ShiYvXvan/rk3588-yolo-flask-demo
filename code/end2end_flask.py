#!/usr/bin/env python3
import cv2
import numpy as np
import threading
import time
import yaml
from rknnlite.api import RKNNLite
from flask import Flask, Response

# ── 配置（集中调整） ─────────────────
MODEL_PATH   = 'yolo26n-rk3588.rknn'
LABEL_PATH   = 'metadata.yaml'          # 支持 .yaml 或 .txt
CAMERA_DEV   = '/dev/video-usbcamera0'
INPUT_SIZE   = (640, 640)
CONF_THRESH  = 0.25
JPEG_QUALITY = 80

# ── 预定义鲜艳颜色池（BGR 格式）─────
COLOR_POOL = [
    # (0, 255, 0),     
    # (255, 0, 0),     
    (0, 0, 255),     
    # (0, 255, 255),   
    # (255, 0, 255),    
    # (255, 255, 0),    
    # (128, 0, 128),    
    # (0, 128, 128),    
    # (128, 128, 0),    
    # (0, 0, 128),      
    # (128, 0, 0),      
    # (0, 128, 0),      
]

# ── 加载类别名 ──────────────────────
def load_labels(path):
    if path.endswith('.yaml') or path.endswith('.yml'):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and 'names' in data:
            return data['names']
        if isinstance(data, list):
            return data
        raise ValueError(f"无法从 {path} 解析标签列表")
    else:
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

labels = load_labels(LABEL_PATH)
num_classes = len(labels)
print(f"标签加载完成，共 {num_classes} 类")

# ── 初始化 RKNN ────────────────────
rknn = RKNNLite()
assert rknn.load_rknn(MODEL_PATH) == 0, "模型加载失败"
assert rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) == 0, "Runtime 初始化失败"
print("RKNN 模型就绪")

# ── 摄像头 ─────────────────────────
cap = cv2.VideoCapture(CAMERA_DEV, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
assert cap.isOpened(), "无法打开摄像头"

# ── 全局变量 ──────────────────────
output_frame = None
lock = threading.Lock()
fps_time = time.time()
frame_cnt = 0
show_fps = 0.0

def process_frames():
    global output_frame, fps_time, frame_cnt, show_fps
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # 预处理
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, INPUT_SIZE)
        input_tensor = np.expand_dims(img, 0)

        # 推理
        t0 = time.time()
        outputs = rknn.inference(inputs=[input_tensor])
        infer_ms = (time.time() - t0) * 1000

        # 绘制结果
        draw = img.copy()
        for det in outputs[0][0]:
            x1, y1, x2, y2, conf, cls_id = det
            if conf < CONF_THRESH or not all(np.isfinite(det)):
                continue
            x1 = max(0, min(int(x1), INPUT_SIZE[0]-1))
            y1 = max(0, min(int(y1), INPUT_SIZE[1]-1))
            x2 = max(0, min(int(x2), INPUT_SIZE[0]-1))
            y2 = max(0, min(int(y2), INPUT_SIZE[1]-1))

            cls_id = int(cls_id)
            # 循环使用颜色池
            color = COLOR_POOL[cls_id % len(COLOR_POOL)]
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
            cls_name = labels[cls_id] if cls_id < len(labels) else f"id{cls_id}"
            # 标签背景与框同色
            (tw, th), _ = cv2.getTextSize(f"{cls_name} {conf:.2f}",
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(draw, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(draw, f"{cls_name} {conf:.2f}", (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # FPS 计算
        frame_cnt += 1
        if (elapsed := time.time() - fps_time) >= 1.0:
            show_fps = frame_cnt / elapsed
            frame_cnt = 0
            fps_time = time.time()

        cv2.putText(draw, f"FPS:{show_fps:.1f} Infer:{infer_ms:.0f}ms",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        with lock:
            output_frame = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)

# ── Flask 视频流 ──────────────────
app = Flask(__name__)

def generate_frames():
    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.01)
                continue
            ret, jpeg = cv2.imencode('.jpg', output_frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''<html><head><title>RK3588 YOLO</title></head>
<body><h1>RK3588 YOLO 实时检测</h1>
<img src="/video_feed" width="640" height="640"></body></html>'''

if __name__ == '__main__':
    threading.Thread(target=process_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
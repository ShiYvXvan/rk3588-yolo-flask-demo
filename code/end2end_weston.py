#!/usr/bin/env python3
import cv2
import numpy as np
import time
import yaml
from rknnlite.api import RKNNLite

# ── 配置 ─────────────────
MODEL_PATH   = 'best-rk3588.rknn'
LABEL_PATH   = 'metadata.yaml'
CAMERA_DEV   = '/dev/video-usbcamera0'      # 根据你的实际设备调整
INPUT_SIZE   = (640, 640)
CONF_THRESH  = 0.25
SCREEN_WIDTH = 800          # 你的 MIPI 屏物理分辨率（宽）
SCREEN_HEIGHT = 480         # 你的 MIPI 屏物理分辨率（高）
ROTATION = cv2.ROTATE_90_CLOCKWISE   # 设为 None 则不旋转

# ── 预定义颜色池 ──────
COLOR_POOL = [
    (0, 255, 0),      # 绿
    (255, 0, 0),      # 蓝
    (0, 0, 255),      # 红
    (0, 255, 255),    # 黄
]

# ── 加载标签 ─────────
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

# ── 初始化 RKNN ───────
rknn = RKNNLite()
assert rknn.load_rknn(MODEL_PATH) == 0, "模型加载失败"
assert rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) == 0, "Runtime 初始化失败"
print("RKNN 模型就绪")

# ── 摄像头 ────────────
cap = cv2.VideoCapture(CAMERA_DEV, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
assert cap.isOpened(), "无法打开摄像头"

# 创建全屏窗口
win_name = "RK3588 YOLO"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

fps_time = time.time()
frame_cnt = 0
show_fps = 0.0

print("开始实时检测，按 'q' 或 ESC 退出...")

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.01)
        continue

    # 预处理
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, INPUT_SIZE)
    input_tensor = np.expand_dims(img_resized, 0)

    # 推理
    t0 = time.time()
    outputs = rknn.inference(inputs=[input_tensor])
    infer_ms = (time.time() - t0) * 1000

    # 绘制结果到原图（便于缩放后显示到屏幕）
    draw = img_resized.copy()
    for det in outputs[0][0]:
        x1, y1, x2, y2, conf, cls_id = det
        if conf < CONF_THRESH or not all(np.isfinite(det)):
            continue
        x1 = max(0, min(int(x1), INPUT_SIZE[0]-1))
        y1 = max(0, min(int(y1), INPUT_SIZE[1]-1))
        x2 = max(0, min(int(x2), INPUT_SIZE[0]-1))
        y2 = max(0, min(int(y2), INPUT_SIZE[1]-1))

        cls_id = int(cls_id)
        color = COLOR_POOL[cls_id % len(COLOR_POOL)]
        cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
        cls_name = labels[cls_id] if cls_id < len(labels) else f"id{cls_id}"
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

    # 缩放至屏幕分辨率（可选，如果要 1:1 映射可去掉）
    display_img = cv2.resize(draw, (SCREEN_WIDTH, SCREEN_HEIGHT))
    display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)  # imshow 用 BGR
    display_img = cv2.rotate(display_img, ROTATION)
    cv2.imshow(win_name, display_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:   # q 或 ESC
        break

cap.release()
cv2.destroyAllWindows()
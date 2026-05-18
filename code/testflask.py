import cv2
from flask import Flask, Response

app = Flask(__name__)

# 打开摄像头设备
# camera = cv2.VideoCapture(41)          # 0 对应 /dev/video0
# 如果需要指定节点，可以用：
camera = cv2.VideoCapture('/dev/video-usbcamera0', cv2.CAP_V4L2)

# 设置分辨率和帧率（可选）
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

def generate_frames():
    """生成 MJPEG 流的帧"""
    while True:
        success, frame = camera.read()
        if not success:
            break
        # 把帧编码为 JPEG 格式
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        # 构造 multipart 响应格式
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """简单展示页面"""
    return '''
    <html>
      <head><title>Camera Stream</title></head>
      <body>
        <h1>UVC Camera Live Stream</h1>
        <img src="/video_feed" width="640" height="480">
      </body>
    </html>
    '''

if __name__ == '__main__':
    # host='0.0.0.0' 允许其他设备访问
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

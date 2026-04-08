import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pathlib import Path

# Cấu hình file bạn muốn "Khám nghiệm"
VIDEO_PATH = r"BSQAv2\data\flagged\smash\iuuLXZ4g8bc_038.mp4" 

# Định nghĩa các đường nối của MediaPipe (33 điểm gốc)
POSE_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10), 
    (11,12), (11,13), (13,15), (15,17), (15,19), (15,21), (17,19), 
    (12,14), (14,16), (16,18), (16,20), (16,22), (18,20), (11,23), 
    (12,24), (23,24), (23,25), (24,26), (25,27), (26,28), (27,29), 
    (28,30), (29,31), (30,32), (27,31), (28,32)
]

MODEL_PATH = Path("BSQAv2/pose_landmarker_full.task")
if not MODEL_PATH.exists():
    import urllib.request
    print("Đang tải AI Model...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task", MODEL_PATH)

base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.3,
    min_pose_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)

print("Đang phát Video... Bấm phím 'q' để thoát, phím 'Space' để tạm dừng/phát tiếp")

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        timestamp_ms = int(frame_idx * 1000 / fps)
        frame_idx += 1
        
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Vẽ các điểm và đường nối
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            landmarks = results.pose_landmarks[0]
            
            # Vẽ dây đỏ nối các điểm
            for start_idx, end_idx in POSE_CONNECTIONS:
                p1 = landmarks[start_idx]
                p2 = landmarks[end_idx]
                if p1.visibility > 0.3 and p2.visibility > 0.3:
                    x1, y1 = int(p1.x * w), int(p1.y * h)
                    x2, y2 = int(p2.x * w), int(p2.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Vẽ điểm xanh lá
            for lm in landmarks:
                if lm.visibility > 0.3:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # Hiển thị frame counter
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_sec = frame_idx / fps
        cv2.putText(
            frame,
            f"Frame: {frame_idx}/{total_frames}  |  {current_sec:.2f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        cv2.imshow('BSQAv2 - X-Ray Mode', frame)
        
        key = cv2.waitKey(100) & 0xFF 
        if key == ord('q'): 
            break
        elif key == ord(' '): 
            cv2.waitKey(-1)

cap.release()
cv2.destroyAllWindows()

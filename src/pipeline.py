import cv2
import time
from ultralytics import YOLO

class QCPipeline:
    def __init__(self, video_source="data/test_video.mp4", model_weights="yolov8n.pt"):
        self.model = YOLO(model_weights)
        self.video_source = video_source
        self.cap = cv2.VideoCapture(self.video_source)

    def process_frame(self):
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_source)

        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        start_time = time.time()
        results = self.model(frame, verbose=False)[0]
        
        latency_ms = round((time.time() - start_time) * 1000, 2)
        fps = round(1000 / latency_ms, 1) if latency_ms > 0 else 0.0

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            conf = round(float(box.conf[0]), 2)
            xyxy = [int(x) for x in box.xyxy[0].tolist()]

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": xyxy
            })

        telemetry = {
            "fps": fps,
            "latency_ms": latency_ms,
            "detected_count": len(detections),
            "detections": detections
        }

        annotated_frame = results.plot()
        return annotated_frame, telemetry

    def release(self):
        self.cap.release()
import cv2
import torch
import numpy as np
import datetime
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Detection:
    x1: int
    y1: int
    x2: int 
    y2: int
    confidence: float
    
    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def get_roi(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y1:self.y2, self.x1:self.x2]

class EnhancedAnomalyDetector:
    def __init__(self, frame_shape: Tuple[int, int]):
        # Detection parameters
        self.confidence_threshold = 0.7
        self.min_anomaly_duration = 2  # seconds
        self.adaptive_threshold = 0.65
        
        # Multi-modal detection weights
        self.weights = {
            'density': 0.35,
            'movement': 0.3,
            'social': 0.25,
            'persistence': 0.1
        }
        
        # Historical data buffers
        self.density_history = deque(maxlen=30)
        self.flow_history = deque(maxlen=30)
        self.social_history = deque(maxlen=30)
        self.anomaly_buffer = deque(maxlen=15)
        
        # Optical flow setup
        self.prev_gray = None
        self.flow_scale = frame_shape[1] / 640  # Normalization factor
        
        # Social distancing parameters
        self.safe_distance = 50 * self.flow_scale  # Adaptive to resolution
        self.min_group_size = 3
        
        # Tracking
        self.last_anomaly_time = 0
        self.last_save_time = 0
        self.save_cooldown = 5

    def calculate_movement_anomaly(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
            
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        magnitude = cv2.GaussianBlur(magnitude, (5,5), 0)
        
        avg_magnitude = np.mean(magnitude)
        self.flow_history.append(avg_magnitude)
        
        if len(self.flow_history) < 2:
            return 0.0
            
        mean_flow = np.mean(self.flow_history)
        std_flow = np.std(self.flow_history)
        if std_flow < 1e-6:
            return 0.0
            
        z_score = (avg_magnitude - mean_flow) / std_flow
        return min(abs(z_score) / 3.0, 1.0)

    def calculate_social_anomaly(self, detections: List[Detection]) -> float:
        if len(detections) < self.min_group_size:
            return 0.0
            
        centers = [((det.x1 + det.x2)//2, (det.y1 + det.y2)//2) for det in detections]
        violation_count = 0
        
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                distance = np.sqrt(dx**2 + dy**2)
                if distance < self.safe_distance:
                    violation_count += 1
                    
        self.social_history.append(violation_count)
        
        if len(self.social_history) < 2:
            return 0.0
            
        mean_social = np.mean(self.social_history)
        std_social = np.std(self.social_history)
        if std_social < 1e-6:
            return 0.0
            
        z_score = (violation_count - mean_social) / std_social
        return min(abs(z_score) / 3.0, 1.0)

    def calculate_density_anomaly(self, frame: np.ndarray, detections: List[Detection]) -> float:
        frame_area = frame.shape[0] * frame.shape[1]
        if frame_area == 0:
            return 0.0
            
        total_area = sum(det.area for det in detections)
        density = total_area / frame_area
        self.density_history.append(density)
        
        if len(self.density_history) < 2:
            return 0.0
            
        mean_density = np.mean(self.density_history)
        std_density = np.std(self.density_history)
        if std_density < 1e-6:
            return 0.0
            
        z_score = (density - mean_density) / std_density
        return min(abs(z_score) / 3.0, 1.0)

    def detect_anomaly(self, frame: np.ndarray, detections: List[Detection]) -> Tuple[float, bool]:
        # Calculate individual anomaly scores
        density_score = self.calculate_density_anomaly(frame, detections)
        movement_score = self.calculate_movement_anomaly(frame)
        social_score = self.calculate_social_anomaly(detections)
        
        # Calculate persistence factor
        self.anomaly_buffer.append(1 if any([density_score > 0.7, movement_score > 0.7, social_score > 0.7]) else 0)
        persistence_score = sum(self.anomaly_buffer) / len(self.anomaly_buffer)
        
        # Weighted confidence score
        confidence = (
            self.weights['density'] * density_score +
            self.weights['movement'] * movement_score +
            self.weights['social'] * social_score +
            self.weights['persistence'] * persistence_score
        )
        
        # Adaptive thresholding
        current_threshold = self.adaptive_threshold + (0.1 * persistence_score)
        is_anomaly = confidence > current_threshold
        
        # Temporal consistency check
        if is_anomaly:
            self.last_anomaly_time = time.time()
        else:
            if time.time() - self.last_anomaly_time < self.min_anomaly_duration:
                is_anomaly = True  # Maintain anomaly state for minimum duration
                
        return min(confidence, 1.0), is_anomaly

class SplitScreenDisplay:
    def __init__(self, output_size: Tuple[int, int] = (1280, 720)):
        self.output_width, self.output_height = output_size
        self.background_color = (40, 40, 40)
        self.text_color = (255, 255, 255)
        self.accent_color = (0, 140, 255)
        
    def create_layout(self, main_frame: np.ndarray, anomaly_detections: List[Detection], confidence: float) -> np.ndarray:
        canvas = np.full((self.output_height, self.output_width, 3), 
                        self.background_color, dtype=np.uint8)
        
        view_width = self.output_width // 2 - 20
        view_height = self.output_height - 40
        main_frame_resized = self.resize_frame(main_frame, (view_width, view_height))
        
        y_offset = 20
        x_offset = 10
        h, w = main_frame_resized.shape[:2]
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = main_frame_resized
        
        cv2.putText(canvas, "Live Feed", 
                   (x_offset, y_offset - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.text_color, 2)
        
        x_offset = self.output_width // 2 + 10
        cv2.putText(canvas, "Anomaly Detections", 
                   (x_offset, y_offset - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.text_color, 2)
        
        cv2.putText(canvas, f"Confidence: {confidence*100:.1f}%",
                   (x_offset, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.accent_color, 1)

        if anomaly_detections:
            grid_size = min(len(anomaly_detections), 4)
            grid_width = view_width // 2
            grid_height = view_height // 2
            
            for idx, det in enumerate(anomaly_detections[:4]):
                grid_x = idx % 2
                grid_y = idx // 2
                
                roi = det.get_roi(main_frame)
                if roi.size == 0:
                    continue
                    
                roi_resized = self.resize_frame(roi, (grid_width - 20, grid_height - 40))
                
                x_pos = x_offset + grid_x * grid_width + 10
                y_pos = y_offset + grid_y * grid_height + 10
                
                h, w = roi_resized.shape[:2]
                canvas[y_pos:y_pos+h, x_pos:x_pos+w] = roi_resized
                
                info_text = f"Detection {idx+1} ({det.confidence*100:.1f}%)"
                cv2.putText(canvas, info_text,
                           (x_pos, y_pos + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.accent_color, 1)
        else:
            msg = "No anomalies detected"
            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = x_offset + (view_width - text_size[0]) // 2
            text_y = y_offset + view_height // 2
            cv2.putText(canvas, msg,
                       (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, self.text_color, 2)
        
        return canvas
    
    @staticmethod
    def resize_frame(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        target_w, target_h = size
        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        
        if new_w < target_w or new_h < target_h:
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            return padded
        
        return resized

class CrowdAnalyzer:
    def __init__(self):
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.detector = None
        self.display = SplitScreenDisplay()
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.detector is None:
            self.detector = EnhancedAnomalyDetector(frame.shape[:2])
            
        # Detect people
        detections = self.detect_people(frame)
        
        # Analyze frame
        confidence, is_anomaly = self.detector.detect_anomaly(frame, detections)
        
        # Visualization and saving logic
        frame_with_boxes = self.draw_detections(frame.copy(), detections, confidence)
        output_frame = self.display.create_layout(frame_with_boxes, detections if is_anomaly else [], confidence)
        
        if is_anomaly and confidence >= self.detector.confidence_threshold:
            self.save_anomaly(output_frame)
            
        return output_frame

    def detect_people(self, frame: np.ndarray) -> List[Detection]:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.yolo_model(img_rgb)
        
        detections = []
        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result.cpu().numpy()
            if conf > 0.5 and int(cls) == 0:
                detections.append(Detection(int(x1), int(y1), int(x2), int(y2), conf))
        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[Detection], confidence: float) -> np.ndarray:
        # Always use red boxes when confidence indicates an anomaly
        for det in detections:
            # Use red color if confidence is above threshold (indicating anomaly)
            color = (0, 0, 255) if confidence >= self.detector.confidence_threshold else (0, 255, 0)
            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        return frame

    def save_anomaly(self, frame: np.ndarray):
        current_time = time.time()
        if current_time - self.detector.last_save_time >= self.detector.save_cooldown:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"anomaly_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            self.detector.last_save_time = current_time

def main():
    # Use 0 for webcam or provide video path
    video_source = 0
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Cannot access video source.")
        return

    analyzer = CrowdAnalyzer()
    
    print("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        output_frame = analyzer.process_frame(frame)
        cv2.imshow("Crowd Analysis System", output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# Run this before processing video

def show_analysis_overlay(frame: np.ndarray, scores: dict):
    y = 30
    for name, value in scores.items():
        cv2.putText(frame, f"{name}: {value:.2f}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        y += 30

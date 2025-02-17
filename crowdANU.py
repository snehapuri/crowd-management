import cv2
import torch
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
from mediapipe import solutions as mp_solutions

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

class SplitScreenDisplay:
    def __init__(self, output_size: Tuple[int, int] = (1280, 720)):
        self.output_width, self.output_height = output_size
        self.background_color = (40, 40, 40)
        self.text_color = (255, 255, 255)
        self.accent_color = (0, 140, 255)
        
    def create_layout(self, main_frame: np.ndarray, anomaly_detections: List[Detection]) -> np.ndarray:
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
    def __init__(self, confidence_threshold: float = 0.4,
                 density_threshold: float = 0.2,
                 anomaly_window: int = 30,
                 z_score_threshold: float = 2.0):
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.pose_model = mp_solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.confidence_threshold = confidence_threshold
        self.density_threshold = density_threshold
        self.z_score_threshold = z_score_threshold
        self.density_history = deque(maxlen=anomaly_window)
        self.display = SplitScreenDisplay()
        
    def detect_people(self, frame: np.ndarray) -> List[Detection]:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.yolo_model(img_rgb)
        
        detections = []
        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result.cpu().numpy()
            if conf > self.confidence_threshold and int(cls) == 0:
                detections.append(Detection(
                    int(x1), int(y1), int(x2), int(y2), conf
                ))
        return detections
    
    def analyze_frame(self, frame: np.ndarray, detections: List[Detection]) -> Tuple[float, bool, Detection]:
        frame_area = frame.shape[0] * frame.shape[1]
        total_person_area = sum(det.area for det in detections)
        density = total_person_area / frame_area
        
        self.density_history.append(density)
        is_anomaly = False
        most_anomalous_detection = None
        
        if len(self.density_history) > 1:
            mean_density = np.mean(self.density_history)
            std_density = np.std(self.density_history)
            if std_density > 0:
                z_score = abs(density - mean_density) / std_density
                is_anomaly = z_score > self.z_score_threshold
        
        # Identify the person with the highest anomaly confidence
        if detections:
            most_anomalous_detection = max(detections, key=lambda det: det.confidence)
        
        return density, is_anomaly, most_anomalous_detection
    
    def analyze_posture(self, frame: np.ndarray, detections: List[Detection]) -> List[str]:
        posture_issues = []
        for det in detections:
            roi = det.get_roi(frame)
            if roi.size == 0:
                continue
            
            results = self.pose_model.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                pose_landmarks = results.pose_landmarks.landmark
                
                if pose_landmarks[23].y < pose_landmarks[24].y:
                    posture_issues.append("Possible squatting posture")
                
                if pose_landmarks[9].y < pose_landmarks[7].y:
                    posture_issues.append("Possible raised hand")
                
                if pose_landmarks[12].y > pose_landmarks[24].y:
                    posture_issues.append("Possible bending posture")
        
        return posture_issues

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        detections = self.detect_people(frame)
        density, is_anomaly, most_anomalous_detection = self.analyze_frame(frame, detections)
        posture_issues = self.analyze_posture(frame, detections)
        
        frame_with_boxes = frame.copy()

        # Draw red box for the person with the highest anomaly confidence
        if most_anomalous_detection:
            # Always mark the most anomalous person with a red box
            color = (0, 0, 255)  # Red box
            cv2.rectangle(frame_with_boxes, 
                        (most_anomalous_detection.x1, most_anomalous_detection.y1), 
                        (most_anomalous_detection.x2, most_anomalous_detection.y2),
                        color, 2)
            
            # Display information about the person with the highest anomaly
            info_text = f"Most Anomalous: {most_anomalous_detection.confidence*100:.1f}%"
            cv2.putText(frame_with_boxes, info_text, 
                        (most_anomalous_detection.x1, most_anomalous_detection.y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display density info and posture issues
        cv2.putText(frame_with_boxes, 
                f"Density: {density*100:.1f}%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if density > self.density_threshold else (0, 255, 0),
                2)
        
        if posture_issues:
            cv2.putText(frame_with_boxes, 
                    "Posture Issue: " + ", ".join(posture_issues),
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Only include anomaly detections in the layout
        anomaly_detections = detections if is_anomaly else []
        return self.display.create_layout(frame_with_boxes, anomaly_detections)

def main():
    video_path = 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot access the video.")
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

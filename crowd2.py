import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
from collections import deque
import time

# Configuration
CONFIDENCE_THRESHOLD = 0.3
CROWD_DENSITY_THRESHOLD_PERCENTAGE = 0.2
NMS_THRESHOLD = 0.3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ANOMALY_DETECTION_THRESHOLD = 0.7
TRACKING_BUFFER_SIZE = 32

class AnomalyDetector:
    def __init__(self):
        # Initialize YOLOv8 pose estimation model
        self.pose_model = YOLO('yolov8n-pose.pt')
        
    def detect_poses(self, frame, bbox):
        """Detect human poses using YOLOv8"""
        x1, y1, x2, y2 = map(int, bbox)
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None

        # Run pose detection on the cropped image
        results = self.pose_model(person_crop)
        if len(results) > 0 and hasattr(results[0], 'keypoints') and len(results[0].keypoints) > 0:
            return results[0].keypoints.data[0]
        return None

    def analyze_motion(self, motion_history):
        """Analyze motion patterns for anomaly detection"""
        if len(motion_history) < 2:
            return "normal", 0.0

        # Calculate motion metrics
        velocities = []
        accelerations = []
        jerk = []  # Rate of change of acceleration
        
        for i in range(1, len(motion_history)):
            prev_pos = motion_history[i-1]
            curr_pos = motion_history[i]
            
            velocity = np.array(curr_pos) - np.array(prev_pos)
            velocity_magnitude = np.linalg.norm(velocity)
            velocities.append(velocity_magnitude)
            
            if i > 1:
                acceleration = velocities[-1] - velocities[-2]
                accelerations.append(abs(acceleration))
                
                if i > 2:
                    jerk.append(abs(acceleration - accelerations[-2]))

        # Calculate statistical features
        avg_velocity = np.mean(velocities) if velocities else 0
        max_velocity = np.max(velocities) if velocities else 0
        avg_acceleration = np.mean(accelerations) if accelerations else 0
        max_acceleration = np.max(accelerations) if accelerations else 0
        avg_jerk = np.mean(jerk) if jerk else 0

        # Analyze patterns for different anomalies
        if max_acceleration > 3.0 and avg_jerk > 1.0:  # Sudden, erratic movements
            if max_velocity > 30:
                return "fight", min(0.95, 0.7 + max_acceleration/10)
            elif avg_velocity < 5 and max_acceleration > 4.0:
                return "fallen", min(0.95, 0.7 + max_acceleration/8)
        elif max_velocity > 20 and avg_velocity > 15:
            if len(velocities) > 5 and all(v > 10 for v in velocities[-5:]):
                return "theft", min(0.9, 0.6 + max_velocity/40)
        
        return "normal", 0.0

    def is_person_fallen(self, keypoints):
        """Analyze pose keypoints to detect if person has fallen"""
        if keypoints is None or len(keypoints) < 17:
            return False

        # Extract key body landmarks
        shoulders = keypoints[[5, 6]]  # Left and right shoulder
        hips = keypoints[[11, 12]]  # Left and right hip
        head = keypoints[0]  # Nose keypoint
        feet = keypoints[[15, 16]]  # Left and right ankle

        # Calculate body orientation
        vertical_alignment = np.abs(np.mean(shoulders[:, 1]) - np.mean(hips[:, 1]))
        horizontal_alignment = np.abs(np.mean(shoulders[:, 0]) - np.mean(hips[:, 0]))
        
        # Calculate head-to-feet vertical distance
        head_feet_vertical = np.abs(head[1] - np.mean(feet[:, 1]))
        
        # Multiple conditions for fallen person detection
        is_horizontal = horizontal_alignment > vertical_alignment
        is_low_height = head_feet_vertical < 0.5  # Normalized height threshold
        
        return is_horizontal and is_low_height

class CompleteCrowdManagementSystem:
    def __init__(self, video_path):
        # Initialize YOLOv8 model for person detection
        self.model = YOLO('yolov8x.pt')
        self.model.conf = CONFIDENCE_THRESHOLD
        
        # Initialize DeepSORT
        self.deepsort = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=100,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True if DEVICE == 'cuda' else False
        )
        
        # Initialize anomaly detector
        self.anomaly_detector = AnomalyDetector()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Error: Cannot access the video.")
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize tracking variables
        self.person_count = 0
        self.tracked_ids = set()
        self.track_history = {}
        self.anomaly_history = {}
        self.anomaly_counts = {"fight": 0, "theft": 0, "fallen": 0}
        
        # Initialize interaction tracking
        self.interaction_history = {}
        
    def preprocess_frame(self, frame):
        """Apply preprocessing to improve detection quality"""
        # Convert to LAB color space for better contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge((l,a,b))
        enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply slight Gaussian blur to reduce noise
        enhanced_frame = cv2.GaussianBlur(enhanced_frame, (3,3), 0)
        
        return enhanced_frame

    def detect_and_track(self, frame):
        """Detect people and track them using DeepSORT"""
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Run YOLOv8 detection
        results = self.model(processed_frame)
        detections = []
        
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                if box.cls[0] == 0:  # person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))
        
        # Update DeepSORT tracker
        tracks = self.deepsort.update_tracks(detections, frame=frame)
        return [track for track in tracks if track.is_confirmed()]

    def track_motion(self, track_id, bbox):
        """Track motion history for each person"""
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=TRACKING_BUFFER_SIZE)
        
        self.track_history[track_id].append(center)
        
        # Clean up old tracks
        current_time = time.time()
        self.track_history = {k: v for k, v in self.track_history.items() 
                            if current_time - v[-1][2] < 5.0 if len(v) > 0}

    def calculate_crowd_density(self, tracks, frame):
        """Calculate and visualize crowd density"""
        if not tracks:
            return frame, 0
        
        total_person_area = sum((track.to_tlbr()[2] - track.to_tlbr()[0]) * 
                               (track.to_tlbr()[3] - track.to_tlbr()[1]) 
                               for track in tracks)
        frame_area = self.frame_width * self.frame_height
        density = total_person_area / frame_area
        
        # Draw density information
        color = (0, 0, 255) if density > CROWD_DENSITY_THRESHOLD_PERCENTAGE else (0, 255, 0)
        cv2.putText(frame, f"Density: {density*100:.2f}%", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw alert if overcrowded
        if density > CROWD_DENSITY_THRESHOLD_PERCENTAGE:
            cv2.putText(frame, "ALERT: Overcrowding!", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame, density

    def detect_anomalies(self, frame, tracks):
        """Detect anomalies based on motion and pose analysis"""
        anomalies = []
        
        for track in tracks:
            track_id = track.track_id
            bbox = track.to_tlbr()
            
            # Track motion
            self.track_motion(track_id, bbox)
            
            # Analyze motion patterns
            if track_id in self.track_history and len(self.track_history[track_id]) >= 2:
                anomaly_type, confidence = self.anomaly_detector.analyze_motion(
                    self.track_history[track_id]
                )
                
                # Enhanced fallen person detection
                if anomaly_type in ["fallen", "fight"]:
                    pose_keypoints = self.anomaly_detector.detect_poses(frame, bbox)
                    if pose_keypoints is not None:
                        if self.anomaly_detector.is_person_fallen(pose_keypoints):
                            anomaly_type = "fallen"
                            confidence = max(confidence, 0.9)
                
                if confidence > ANOMALY_DETECTION_THRESHOLD:
                    anomalies.append((track_id, bbox, anomaly_type, confidence))
                    
                    # Update anomaly history and counts
                    if track_id not in self.anomaly_history:
                        self.anomaly_history[track_id] = []
                    self.anomaly_history[track_id].append((anomaly_type, time.time()))
                    
                    # Increment counter only for new anomalies
                    if not self.anomaly_history[track_id] or \
                       self.anomaly_history[track_id][-1][0] != anomaly_type:
                        self.anomaly_counts[anomaly_type] += 1
        
        return anomalies

    def draw_tracks(self, frame, tracks):
        """Draw tracking information on frame"""
        current_ids = set()
        
        for track in tracks:
            bbox = track.to_tlbr()
            track_id = track.track_id
            
            # Update tracked IDs
            current_ids.add(track_id)
            if track_id not in self.tracked_ids:
                self.tracked_ids.add(track_id)
                self.person_count += 1
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw ID and trajectory
            cv2.putText(frame, f"ID: {track_id}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw trajectory
            if track_id in self.track_history:
                for i in range(1, len(self.track_history[track_id])):
                    prev_pt = tuple(map(int, self.track_history[track_id][i-1]))
                    curr_pt = tuple(map(int, self.track_history[track_id][i]))
                    cv2.line(frame, prev_pt, curr_pt, (0, 255, 0), 1)
        
        # Draw counts
        cv2.putText(frame, f"Total Count: {self.person_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Current Count: {len(current_ids)}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame

    def draw_anomalies(self, frame, anomalies):
        """Visualize detected anomalies"""
        for track_id, bbox, anomaly_type, confidence in anomalies:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Color coding for different anomalies
            color = {
                "fight": (0, 0, 255),    # Red
                "theft": (255, 0, 0),    # Blue
                "fallen": (0, 165, 255)  # Orange
            }.get(anomaly_type, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw anomaly label with confidence
            text = f"{anomaly_type.upper()} ({confidence:.2f})"
            cv2.putText(frame, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw alert for high-confidence anomalies
            if confidence > 0.85:
                alert_text = f"ALERT: {anomaly_type.upper()} detected!"
                cv2.putText(frame, alert_text, (10, 190)),
                cv2.putText(frame, alert_text, (10, 190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw anomaly statistics
        y_pos = 230
        for anomaly_type, count in self.anomaly_counts.items():
            if count > 0:
                text = f"{anomaly_type.capitalize()} incidents: {count}"
                cv2.putText(frame, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 30

    def detect_interactions(self, tracks):
        """Detect interactions between people"""
        interactions = []
        for i, track1 in enumerate(tracks):
            for track2 in tracks[i+1:]:
                # Calculate distance between people
                center1 = np.array([(track1.to_tlbr()[0] + track1.to_tlbr()[2])/2,
                                  (track1.to_tlbr()[1] + track1.to_tlbr()[3])/2])
                center2 = np.array([(track2.to_tlbr()[0] + track2.to_tlbr()[2])/2,
                                  (track2.to_tlbr()[1] + track2.to_tlbr()[3])/2])
                
                distance = np.linalg.norm(center1 - center2)
                
                # Check for close interactions
                if distance < 100:  # Threshold in pixels
                    interactions.append((track1.track_id, track2.track_id, distance))
                    
                    # Record interaction in history
                    interaction_key = tuple(sorted([track1.track_id, track2.track_id]))
                    if interaction_key not in self.interaction_history:
                        self.interaction_history[interaction_key] = []
                    self.interaction_history[interaction_key].append(time.time())
        
        return interactions

    def process_video(self):
        """Main processing loop"""
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect and track people
            tracks = self.detect_and_track(frame)
            
            # Detect anomalies
            anomalies = self.detect_anomalies(frame, tracks)
            
            # Detect interactions
            interactions = self.detect_interactions(tracks)
            
            # Draw tracking information
            frame = self.draw_tracks(frame, tracks)
            
            # Draw anomalies
            self.draw_anomalies(frame, anomalies)
            
            # Calculate and visualize crowd density
            frame, density = self.calculate_crowd_density(tracks, frame)
            
            # Calculate and display FPS
            if frame_count % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                start_time = time.time()
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, self.frame_height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow("Advanced Crowd Management System", frame)
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\nFinal Statistics:")
        print(f"Total people tracked: {self.person_count}")
        print("\nAnomalies detected:")
        for anomaly_type, count in self.anomaly_counts.items():
            print(f"{anomaly_type.capitalize()}: {count}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        if hasattr(self, 'cap'):
            self.cleanup()

def main():
    try:
        # Initialize the system
        video_path = input(r"C:\Users\Sneha\Documents\crowd yolo\Crowd-UIT\Video\1.mp4")
        system = CompleteCrowdManagementSystem(video_path)
        
        print("\nInitialization complete. Starting video processing...")
        print("Press 'q' to quit the application.")
        
        # Process the video
        system.process_video()
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please make sure you have:")
        print("1. Installed all required dependencies (ultralytics, deep_sort_realtime)")
        print("2. Provided a valid video file path")
        print("3. Have sufficient GPU/CPU resources available")
        
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
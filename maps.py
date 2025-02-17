import cv2
import torch
import numpy as np
from collections import defaultdict
import datetime
import os
import math
import requests
from PIL import Image, ImageDraw
import io

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Configuration
CONFIDENCE_THRESHOLD = 0.4
CROWD_DENSITY_THRESHOLD_PERCENTAGE = 0.2
MAX_DISAPPEARED = 30  # Maximum frames a person can disappear before getting a new ID

class PersonTracker:
    def __init__(self):
        self.next_id = 1
        self.persons = {}  # {id: {"box": (x1,y1,x2,y2), "disappeared": count}}
        self.disappeared = defaultdict(int)
    
    def update(self, detections):
        # If no detections, increment disappeared count for all existing persons
        if not detections:
            for person_id in list(self.persons.keys()):
                self.disappeared[person_id] += 1
                if self.disappeared[person_id] > MAX_DISAPPEARED:
                    del self.persons[person_id]
                    del self.disappeared[person_id]
            return self.persons

        # Initialize new persons if none exist
        if not self.persons:
            for det in detections:
                self.register(det)
            return self.persons

        # Calculate IoU between existing persons and new detections
        person_boxes = [person["box"] for person in self.persons.values()]
        iou_matrix = self._calculate_iou(person_boxes, detections)

        # Match detections to existing persons using IoU
        matched_indices = self._match_detections(iou_matrix)
        used_persons = set()
        used_detections = set()

        # Update matched persons
        for person_idx, det_idx in matched_indices:
            person_id = list(self.persons.keys())[person_idx]
            self.persons[person_id]["box"] = detections[det_idx]
            self.disappeared[person_id] = 0
            used_persons.add(person_id)
            used_detections.add(det_idx)

        # Register new detections
        for i in range(len(detections)):
            if i not in used_detections:
                self.register(detections[i])

        # Remove persons that have disappeared
        for person_id in list(self.persons.keys()):
            if person_id not in used_persons:
                self.disappeared[person_id] += 1
                if self.disappeared[person_id] > MAX_DISAPPEARED:
                    del self.persons[person_id]
                    del self.disappeared[person_id]

        return self.persons

    def register(self, detection):
        self.persons[self.next_id] = {"box": detection, "disappeared": 0}
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _calculate_iou(self, boxes1, boxes2):
        iou_matrix = np.zeros((len(boxes1), len(boxes2)))
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                iou_matrix[i, j] = self._box_iou(box1, box2)
        return iou_matrix

    def _box_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / float(box1_area + box2_area - intersection)

    def _match_detections(self, iou_matrix, iou_threshold=0.3):
        matched_indices = []
        if iou_matrix.size > 0:
            for i in range(iou_matrix.shape[0]):
                j = iou_matrix[i].argmax()
                if iou_matrix[i][j] >= iou_threshold:
                    matched_indices.append((i, j))
        return matched_indices

class CrowdMapper:
    def __init__(self, video_path):
        # ... (keep existing initialization code) ...
        
        # Add person tracker
        self.person_tracker = PersonTracker()
        
        # Initialize detection data with IDs
        self.detection_points = {}  # {id: [(lat, lon, timestamp), ...]}

    def draw_bounding_boxes(self, frame, detections):
        """Draw bounding boxes around detected people and return their positions with IDs."""
        positions = {}
        tracked_persons = self.person_tracker.update(detections)

        for person_id, person_data in tracked_persons.items():
            x1, y1, x2, y2, confidence = person_data["box"]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw ID
            label = f"ID: {person_id} ({confidence * 100:.1f}%)"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Calculate center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Convert to GPS coordinates
            lat, lon = self.pixel_to_gps(center_x, center_y)
            
            # Store position with timestamp
            timestamp = datetime.datetime.now()
            if person_id not in self.detection_points:
                self.detection_points[person_id] = []
            self.detection_points[person_id].append((lat, lon, timestamp))
            
            positions[person_id] = (lat, lon)
            
        return positions

    def update_live_map(self, map_image):
        """Update the live map with current detection points showing IDs"""
        current_map = map_image.copy()
        
        # Draw detection points with IDs
        for person_id, points in self.detection_points.items():
            if not points:
                continue
                
            # Get most recent position
            lat, lon, _ = points[-1]
            
            try:
                px, py = self.lat_lon_to_pixel(lat, lon, CAMERA_LAT, CAMERA_LON)
                
                if 0 <= px < self.map_size[0] and 0 <= py < self.map_size[1]:
                    # Draw circle with ID
                    cv2.circle(current_map, (px, py), 4, (255, 255, 255), -1)
                    cv2.circle(current_map, (px, py), 3, (0, 0, 255), -1)
                    cv2.putText(current_map, str(person_id), 
                              (px + 5, py - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(current_map, str(person_id), 
                              (px + 5, py - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except (ValueError, OverflowError):
                continue
        
        return current_map
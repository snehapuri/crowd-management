import cv2
import torch
import numpy as np
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

# Camera GPS location (example coordinates - replace with actual camera location)
CAMERA_LAT = 12.8250105  # Example coordinates
CAMERA_LON = 80.0449266
CAMERA_FOV_HORIZONTAL = math.radians(60)  # Set your camera's actual horizontal FOV
CAMERA_HEIGHT = 10  # Set your camera's actual height in meters

# Field of view parameters
CAMERA_FOV_HORIZONTAL = math.radians(60)  # 60 degrees horizontal FOV
CAMERA_HEIGHT = 10  # Height of camera in meters
PIXELS_TO_METERS_RATIO = None  # Will be calculated based on frame width

class CrowdMapper:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Error: Cannot access the video.")

        # Get video frame dimensions
        _, first_frame = self.cap.read()
        self.frame_width = first_frame.shape[1]
        self.frame_height = first_frame.shape[0]

        # Calculate ground plane coverage at camera height
        ground_width = 2 * CAMERA_HEIGHT * math.tan(CAMERA_FOV_HORIZONTAL / 2)
        global PIXELS_TO_METERS_RATIO
        PIXELS_TO_METERS_RATIO = ground_width / self.frame_width

        # Initialize detection data
        self.detection_points = []
        self.heatmap_data = []

        # Map settings
        self.zoom = 19
        self.map_size = (800, 800)
        self.tile_size = 256

        # Initialize base map
        self.base_map = self.download_base_map()

    def pixel_to_gps(self, x, y):
        """Convert pixel coordinates to GPS coordinates using improved projection."""
        # Calculate the real-world distance based on camera parameters
        angle_x = ((x - self.frame_width/2) / self.frame_width) * CAMERA_FOV_HORIZONTAL

        # Calculate ground distance using camera height and angle
        ground_distance = CAMERA_HEIGHT * math.tan(math.pi/2 - math.atan(self.frame_height/2 - y))

        # Calculate real world coordinates
        meters_x = ground_distance * math.sin(angle_x)
        meters_y = ground_distance * math.cos(angle_x)

        # Convert to GPS coordinates using Haversine formula
        R = 6378137  # Earth's radius in meters

        lat_offset = (meters_y / R) * (180 / math.pi)
        lon_offset = (meters_x / R) * (180 / math.pi) / math.cos(math.radians(CAMERA_LAT))

        return CAMERA_LAT + lat_offset, CAMERA_LON + lon_offset

    def lon_to_tile_x(self, lon, zoom):
        """Convert longitude to tile x coordinate"""
        return int((lon + 180.0) / 360.0 * (1 << zoom))

    def lat_to_tile_y(self, lat, zoom):
        """Convert latitude to tile y coordinate"""
        lat_rad = math.radians(lat)
        return int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * (1 << zoom))

    def get_tile_image(self, x, y, zoom):
        """Download map tile from OpenStreetMap."""
        url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
        headers = {
            'User-Agent': 'CrowdMapperApp/1.0'
        }
        response = requests.get(url, headers=headers)
        return Image.open(io.BytesIO(response.content))

    def download_base_map(self):
        """Download and stitch map tiles for the base map"""
        # Calculate center tile coordinates
        center_tile_x = self.lon_to_tile_x(CAMERA_LON, self.zoom)
        center_tile_y = self.lat_to_tile_y(CAMERA_LAT, self.zoom)
        
        # Calculate how many tiles we need
        tiles_x = math.ceil(self.map_size[0] / self.tile_size) + 1
        tiles_y = math.ceil(self.map_size[1] / self.tile_size) + 1
        
        # Create a large enough image to hold all tiles
        base_map = Image.new('RGB', (tiles_x * self.tile_size, tiles_y * self.tile_size))
        
        # Download and stitch tiles
        for dx in range(-tiles_x//2, tiles_x//2 + 1):
            for dy in range(-tiles_y//2, tiles_y//2 + 1):
                tile_x = center_tile_x + dx
                tile_y = center_tile_y + dy
                
                try:
                    tile_img = self.get_tile_image(tile_x, tile_y, self.zoom)
                    base_map.paste(tile_img, 
                                 ((dx + tiles_x//2) * self.tile_size, 
                                  (dy + tiles_y//2) * self.tile_size))
                except Exception as e:
                    print(f"Error downloading tile {tile_x},{tile_y}: {e}")
                    continue
        
        # Crop to desired size
        center_x = base_map.width // 2
        center_y = base_map.height // 2
        crop_box = (
            center_x - self.map_size[0]//2,
            center_y - self.map_size[1]//2,
            center_x + self.map_size[0]//2,
            center_y + self.map_size[1]//2
        )
        base_map = base_map.crop(crop_box)
        
        # Convert to numpy array for OpenCV
        return cv2.cvtColor(np.array(base_map), cv2.COLOR_RGB2BGR)

    def lat_lon_to_pixel(self, lat, lon, center_lat, center_lon):
        """Convert latitude/longitude to pixel coordinates relative to center."""
        # Constants for Web Mercator projection
        TILE_SIZE = 256
        EARTH_RADIUS = 6378137  # Earth's radius in meters

        def mercator_x(lon):
            return EARTH_RADIUS * math.radians(lon)

        def mercator_y(lat):
            y = EARTH_RADIUS * math.log(math.tan(math.pi/4 + math.radians(lat)/2))
            return y

        # Convert center and point to mercator coordinates
        center_x = mercator_x(center_lon)
        center_y = mercator_y(center_lat)
        point_x = mercator_x(lon)
        point_y = mercator_y(lat)

        # Calculate pixel offsets
        scale = 1 << self.zoom
        pixels_per_meter = TILE_SIZE * scale / (2 * math.pi * EARTH_RADIUS)

        dx = (point_x - center_x) * pixels_per_meter
        dy = (point_y - center_y) * pixels_per_meter

        # Convert to pixel coordinates relative to center of map
        px = int(self.map_size[0]/2 + dx)
        py = int(self.map_size[1]/2 - dy)  # Subtract because y increases downward in image

        return px, py

    def draw_bounding_boxes(self, frame, detections):
        """Draw bounding boxes around detected people and return their positions."""
        positions = []
        for det in detections:
            x1, y1, x2, y2, confidence = det
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person ({confidence * 100:.1f}%)"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Calculate center point of detection
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Convert to GPS coordinates
            lat, lon = self.pixel_to_gps(center_x, center_y)
            positions.append([lat, lon, 1.0])  # Keep intensity for future use
            self.detection_points.append([lat, lon, 1.0])
            
        return positions

    def process_frame(self, frame):
        """Process frame for people detection and update map."""
        # Convert frame to RGB for YOLO
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = yolo_model(img_rgb)
        detections = []

        # Filter detections
        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result.cpu().numpy()
            if conf > CONFIDENCE_THRESHOLD and int(cls) == 0:  # Class 0 is person
                detections.append((int(x1), int(y1), int(x2), int(y2), conf))

        # Get positions and draw boxes
        positions = self.draw_bounding_boxes(frame, detections)
        
        # Update heatmap data
        self.heatmap_data.extend(positions)

        # Calculate and display crowd density
        total_person_area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2, _ in detections)
        frame_area = frame.shape[0] * frame.shape[1]
        density_percentage = total_person_area / frame_area

        # Display crowd density information
        text = f"Crowd Density: {density_percentage * 100:.2f}%"
        color = (0, 0, 255) if density_percentage > CROWD_DENSITY_THRESHOLD_PERCENTAGE else (0, 255, 0)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if density_percentage > CROWD_DENSITY_THRESHOLD_PERCENTAGE:
            cv2.putText(frame, "ALERT: Overcrowding detected!", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

    def create_live_map(self):
        """Return the base map for live updates"""
        return self.base_map.copy()

    def update_live_map(self, map_image):
        """Update the live map with current detection points"""
        current_map = map_image.copy()
        
        # Get only the most recent positions (last 50 detections)
        recent_points = self.detection_points[-50:] if len(self.detection_points) > 50 else self.detection_points
        
        # Draw detection points
        for lat, lon, _ in recent_points:
            try:
                px, py = self.lat_lon_to_pixel(lat, lon, CAMERA_LAT, CAMERA_LON)
                
                # Check if point is within map bounds
                if 0 <= px < self.map_size[0] and 0 <= py < self.map_size[1]:
                    # Draw a red circle with white border for better visibility
                    cv2.circle(current_map, (px, py), 4, (255, 255, 255), -1)  # White background
                    cv2.circle(current_map, (px, py), 3, (0, 0, 255), -1)      # Red dot
            except (ValueError, OverflowError):
                continue
        
        return current_map

    def run(self):
        """Main processing loop with live map visualization."""
        print("Press 'q' to exit")

        cv2.namedWindow("Crowd Detection", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Live Map", cv2.WINDOW_NORMAL)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Process frame and update detections
            processed_frame = self.process_frame(frame)

            # Update and show the live map
            live_map = self.update_live_map(self.base_map)

            # Show both windows
            cv2.imshow("Crowd Detection", processed_frame)
            cv2.imshow("Live Map", live_map)

           
            cv2.resizeWindow("Crowd Detection", 800, 600)
            cv2.resizeWindow("Live Map", 600, 600)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 0
    #r"C:\Users\Sneha\Documents\crowd yolo\Crowd-UIT\Video\1.mp4"
    mapper = CrowdMapper(video_path)
    mapper.run()
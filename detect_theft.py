import cv2
import torch
from mtcnn import MTCNN
import numpy as np
from pathlib import Path
import os

# Initialize MTCNN for face detection
detector = MTCNN()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def get_faces(image):
    """Detect faces in the image using MTCNN"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_image)
    return faces

def get_bike_and_human_detections(image):
    """Detect bikes and humans using YOLOv5"""
    results = model(image)
    detections = results.xyxy[0].numpy()
    # Filter for bikes (class 1) and persons (class 0)
    relevant_detections = [d for d in detections if int(d[5]) in [0, 1]]
    return relevant_detections

def is_theft_event(faces, detections):
    """Determine if current scene represents a potential theft"""
    if len(faces) > 0 and len(detections) > 0:
        # Simple heuristic: If person and bike are close to each other
        for detection in detections:
            if int(detection[5]) == 1:  # bike
                person_nearby = any(
                    int(d[5]) == 0 and calculate_distance(detection[:4], d[:4]) < 100 
                    for d in detections
                )
                if person_nearby:
                    return True
    return False

def calculate_distance(box1, box2):
    """Calculate distance between centers of two bounding boxes"""
    center1 = [(box1[0] + box1[2])/2, (box1[1] + box1[3])/2]
    center2 = [(box2[0] + box2[2])/2, (box2[1] + box2[3])/2]
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def draw_bounding_boxes(image, detections):
    """Draw bounding boxes for detected objects"""
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if int(cls) == 0:  # person
            color = (0, 255, 0)  # green
        else:  # bike
            color = (0, 0, 255)  # red
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

def main():
    # Replace with your RTSP stream URL
    rtsp_url = "rtsp://your_camera_ip:port/stream"
    
    # Create output directories
    output_dir = Path("detected_events")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize video capture
    cap = cv2.VideoCapture(rtsp_url)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Process frame
        faces = get_faces(frame)
        detections = get_bike_and_human_detections(frame)
        
        # Draw detections
        draw_bounding_boxes(frame, detections)
        
        # Check for theft event
        if is_theft_event(faces, detections):
            # Save frame if theft detected
            timestamp = cv2.getTickCount()
            cv2.imwrite(str(output_dir / f"theft_event_{timestamp}.jpg"), frame)
            print(f"Potential theft event detected at {timestamp}")
        
        # Display frame
        cv2.imshow('Bike Theft Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
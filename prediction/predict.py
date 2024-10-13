import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker

# Load the YOLOv8 model
model = YOLO('./prediction/content/runs/detect/train/weights/last.pt')

# Load the video file
input_video_path = './prediction/video1.mp4'
output_video_path = './out1.mp4'

# Open the video using OpenCV
video_capture = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object to save output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize a set to track unique car IDs
unique_car_ids = set()

# Initialize the Norfair tracker for tracking cars
tracker = Tracker(distance_function=lambda det1, det2: np.linalg.norm(det1.points[0] - det2.estimate), distance_threshold=30)

# Function to convert YOLO detections to Norfair detections
def yolo_detections_to_norfair_detections(yolo_results):
    detections = []
    for result in yolo_results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result[:6]
        if conf > 0.5 and model.names[int(cls)] == "car":  # Only track cars
            # Calculate the centroid of the bounding box
            centroid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])  # Use centroid for tracking
            # Create a Detection object with the centroid as points
            detections.append(Detection(points=centroid.reshape(1, 2)))  # Reshape to match expected input
    return detections

# Iterate over each frame
frame_count = 0
while video_capture.isOpened():
    ret, frame = video_capture.read()  # Read a frame
    if not ret:
        break
    
    # Ensure frame size consistency
    if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
        frame = cv2.resize(frame, (frame_width, frame_height))
    
    # Apply YOLOv8 object detection
    results = model(frame)[0]

    # Convert YOLO detections to Norfair detections
    detections = yolo_detections_to_norfair_detections(results)
    
    # Update tracker with the current detections
    tracked_objects = tracker.update(detections)

    # Draw tracked bounding boxes and update unique car IDs
    for obj in tracked_objects:
        # Access the estimated position directly
        centroid = obj.estimate.flatten()  # Ensure centroid is flat
        cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 5, (0, 255, 0), -1)  # Draw circle on car's center
        cv2.putText(frame, f"ID: {obj.id}", (int(centroid[0]), int(centroid[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Add the unique ID to the set
        unique_car_ids.add(obj.id)

        # Draw bounding box around detected cars using the centroid and add some offset for width/height
        x_min, y_min = int(centroid[0]) - 25, int(centroid[1]) - 25  # Offset for centering the box
        x_max, y_max = int(centroid[0]) + 25, int(centroid[1]) + 25  # 50x50 box around the centroid
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Draw bounding box

    # Update the unique car count based on the size of the set
    unique_car_count = len(unique_car_ids)

    # Display the total car count on the video
    cv2.putText(frame, f'Total cars: {unique_car_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the processed frame to the output video
    out_video.write(frame)

    # Print progress
    frame_count += 1
    print(f'Processed frame {frame_count}/{total_frames}')

# Release resources properly
video_capture.release()
out_video.release()
cv2.destroyAllWindows()

# Print the total unique car count
print(f'Total unique cars counted: {unique_car_count}')
print(f'Output video saved to {output_video_path}')

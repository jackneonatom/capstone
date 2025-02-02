import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
from sort import Sort  # Import the SORT tracker
import time  # For FPS calculation

# Load the YOLO model
model = YOLO('best_full_integer_quant_edgetpu.tflite')

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Load class names
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Initialize the SORT tracker
tracker = Sort()

# Dictionary to map track_id to class_id
track_id_to_class_id = {}

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()

# Create a named window
cv2.namedWindow("FRAME", cv2.WINDOW_NORMAL)

# Variable to track fullscreen state
fullscreen = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame (optional, depending on your display resolution)
    frame = cv2.resize(frame, (1020, 600))

    # Increment frame count for FPS calculation
    frame_count += 1

    # Perform object detection every 3 frames (to reduce computation)
    if frame_count % 3 != 0:
        continue

    # Perform object detection
    results = model.predict(frame, imgsz=240)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # Extract bounding boxes and class IDs
    detections = []
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = row
        detections.append([x1, y1, x2, y2, d])  # Include class_id in detections

    # Update the tracker with the detected objects
    if len(detections) > 0:
        detections = np.array(detections)
        tracks = tracker.update(detections[:, :4])  # Pass only [x1, y1, x2, y2] to the tracker

        # Update the track_id_to_class_id mapping
        for detection, track in zip(detections, tracks):
            class_id = int(detection[4])  # Class ID from the detection
            track_id = int(track[4])      # Track ID from the tracker
            track_id_to_class_id[track_id] = class_id  # Map track_id to class_id
    else:
        tracks = np.empty((0, 5))

    # Draw bounding boxes and IDs on the frame
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)

        # Get class_id from the mapping
        if track_id in track_id_to_class_id:
            class_id = track_id_to_class_id[track_id]
            class_name = class_list[class_id]
        else:
            class_name = "Unknown"  # Handle cases where the mapping is missing

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Display class name and track ID
        cvzone.putTextRect(frame, f'{class_name} ID:{track_id}', (x1, y1), 1, 1)

    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    # Display FPS on the top right corner of the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("FRAME", frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF

    # Toggle fullscreen on 'f' key press
    if key == ord('f'):
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty("FRAME", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("FRAME", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    # Exit on ESC key press
    if key == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

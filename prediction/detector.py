

import argparse
import cv2
import os
import numpy as np
import requests
import time
from collections import deque
from smbus2 import SMBus
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
from sort import Sort

# Cloud API configuration
CLOUD_API_URL = "https://api.v-tally.com/store"

# â”€â”€â”€ IÂ²C & Register Definitions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
INA226_ADDR        = 0x50    # Found via `i2cdetect -y 1`
REG_CONFIG         = 0x00    # Configuration (R/W)
REG_SHUNT_VOLTAGE  = 0x01    # Differential input (R)
REG_BUS_VOLTAGE    = 0x02    # Bus voltage (R)

# â”€â”€â”€ Measurement Parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
R_SHUNT_OHMS       = 0.1          # Î©
LSB_SHUNT_VOLTAGE  = 2.5e-6       # 2.5 ÂµV per bit
LSB_BUS_VOLTAGE    = 1.25e-3      # 1.25 mV per bit
CONFIG_VALUE       = 0x4127       # Continuous shunt+bus conversions

# â”€â”€â”€ Averaging Settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
AVG_COUNT = 10                    # Number of current samples to average

# Initialize counters and trackers
counted_tracks = set()
category_counts = {
    "car": 0,
    "truck": 0,
    "bike": 0,
    "person": 0
}

# Track ID to (class ID, confidence) mapping
track_id_class_map = {}

# Color palette for different classes
COLORS = {
    "car": (0, 255, 0),       # Green
    "truck": (0, 165, 255),   # Orange
    "bike": (255, 0, 0),      # Blue
    "person": (0, 0, 255)     # Red
}

# Power monitoring variables
current_buffer = deque(maxlen=AVG_COUNT)
total_energy_wh = 0.0  # Total energy consumed in watt-hours
last_power_time = None
bus = None

# â”€â”€â”€ IÂ²C Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def read_reg(bus, reg):
    try:
        raw = bus.read_word_data(INA226_ADDR, reg)
        # Swap bytes (littleâ€‘endian â†’ bigâ€‘endian)
        return ((raw & 0xFF) << 8) | (raw >> 8)
    except Exception as e:
        print(f"I2C read error: {e}")
        return 0

def write_reg(bus, reg, value):
    try:
        v = ((value & 0xFF) << 8) | (value >> 8)
        bus.write_word_data(INA226_ADDR, reg, v)
    except Exception as e:
        print(f"I2C write error: {e}")

def read_shunt_voltage(bus):
    raw = read_reg(bus, REG_SHUNT_VOLTAGE)
    # Two'sâ€‘complement conversion
    if raw & 0x8000:
        raw -= 1 << 16
    return raw * LSB_SHUNT_VOLTAGE

def read_bus_voltage(bus):
    raw = read_reg(bus, REG_BUS_VOLTAGE)
    return raw * LSB_BUS_VOLTAGE

def get_power_consumption():
    """Get current power consumption and update total energy"""
    global current_buffer, total_energy_wh, last_power_time, bus
    
    if bus is None:
        return 0.0, 0.0  # Return 0 watts and 0 watt-hours if no sensor
    
    try:
        current_time = time.time()
        
        # Read power measurements
        Vshunt = read_shunt_voltage(bus)
        Iinst = Vshunt / R_SHUNT_OHMS
        Vbus = read_bus_voltage(bus)
        
        # Update moving average for current
        current_buffer.append(Iinst)
        Iavg = sum(current_buffer) / len(current_buffer)
        
        # Calculate power
        current_power_w = Vbus * Iavg
        
        # Update energy consumption (watt-hours)
        if last_power_time is not None:
            time_diff_hours = (current_time - last_power_time) / 3600.0
            energy_increment = current_power_w * time_diff_hours
            total_energy_wh += energy_increment
        
        last_power_time = current_time
        
        return current_power_w, total_energy_wh
        
    except Exception as e:
        print(f"Power measurement error: {e}")
        return 0.0, total_energy_wh

def classify_object(class_id, labels):
    """Classify object based on COCO class ID"""
    try:
        class_name = labels.get(class_id, '').lower()
        if class_id == 2:  # car
            return 'car'
        elif class_id == 5:  # bus
            return 'car'  # Group buses with cars
        elif class_id == 7:  # truck
            return 'truck'
        elif class_id == 1:  # bicycle
            return 'bike'
        elif class_id == 3:  # motorcycle
            return 'bike'
        elif class_id == 0:  # person
            return 'person'
        return None
    except Exception as e:
        print(f"Classification error for class_id {class_id}: {e}")
        return None

def send_to_cloud():
    """Send count data to cloud server in API-compatible format"""
    current_power, energy_consumed = get_power_consumption()
    
    # Send data in the exact format your API expects
    data = {
        "car_count": category_counts["car"],
        "truck_count": category_counts["truck"],
        "bike_count": category_counts["bike"],
        "ped_count": category_counts["person"],
        "battery_percentage": round(energy_consumed, 3)  # Store energy consumption as "battery_percentage"
    }
    try:
        response = requests.post(CLOUD_API_URL, json=data, timeout=5)
        response.raise_for_status()
        print(f"Data sent: {data} (Power: {current_power:.2f}W, Energy: {energy_consumed:.3f}Wh)")
        return True
    except Exception as e:
        print(f"Cloud error: {e}")
        return False

def main():
    global bus
    
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=10,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use.', default=0)
    parser.add_argument('--threshold', type=float, default=0.65,
                        help='classifier score threshold')
    parser.add_argument('--no-power', action='store_true',
                        help='Disable power monitoring (useful if INA226 not available)')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    # Initialize power monitoring
    if not args.no_power:
        try:
            bus = SMBus(1)
            write_reg(bus, REG_CONFIG, CONFIG_VALUE)
            time.sleep(0.1)
            print("INA226 power sensor initialized")
        except Exception as e:
            print(f"Warning: Could not initialize INA226 sensor: {e}")
            print("Running without power monitoring")
            bus = None
    else:
        print("Power monitoring disabled")

    # Initialize video capture
    cap = cv2.VideoCapture(args.camera_idx)
    cap.set(3, 1280)  # Width
    cap.set(4, 720)   # Height

    # Initialize SORT tracker with optimized parameters
    tracker = Sort(max_age=10, min_hits=1)

    # FPS variables
    start_time = time.time()
    frame_count = 0
    last_sent_time = time.time()
    
    cv2.namedWindow("Object Counter", cv2.WINDOW_NORMAL)
    fullscreen = False

    print("Starting detection and tracking...")
    print("Tracking categories: car, truck, bike, person")
    if bus:
        print("Power monitoring: ENABLED")
    print("Press 'f' for fullscreen, 'ESC' to quit")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = time.time()
            new_detections = False

            # Process detections every 3 frames for performance
            if frame_count % 3 == 0:
                # Prepare frame for inference
                cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
                
                # Run inference
                run_inference(interpreter, cv2_im_rgb.tobytes())
                objs = get_objects(interpreter, args.threshold)[:args.top_k]
                
                # Convert detections to format for SORT tracker
                detections = []
                height, width = frame.shape[:2]
                scale_x, scale_y = width / inference_size[0], height / inference_size[1]
                
                for obj in objs:
                    # Only process objects we care about
                    category = classify_object(obj.id, labels)
                    if category:
                        bbox = obj.bbox.scale(scale_x, scale_y)
                        x1, y1 = int(bbox.xmin), int(bbox.ymin)
                        x2, y2 = int(bbox.xmax), int(bbox.ymax)
                        conf = float(obj.score)
                        detections.append([x1, y1, x2, y2, obj.id, conf])

                # Update tracker with detections
                if detections:
                    detections_array = np.array([d[:4] for d in detections])
                    trackers = tracker.update(detections_array)
                    
                    # Update class ID and confidence mapping
                    for detection, track in zip(detections, trackers):
                        track_id = int(track[4])
                        class_id = detection[4]
                        conf = detection[5]
                        track_id_class_map[track_id] = (class_id, conf)
                else:
                    trackers = tracker.update(np.empty((0, 5)))
            else:
                # Continue tracking with existing information
                trackers = tracker.update(np.empty((0, 5)))

            # Process tracks and draw boxes every frame
            for track in trackers:
                x1, y1, x2, y2, track_id = map(int, track[:5])
                
                # Get class and confidence information
                track_info = track_id_class_map.get(track_id, (None, None))
                class_id, confidence = track_info
                category = classify_object(class_id, labels) if class_id is not None else None
                
                # Draw bounding box and label if confidence is sufficient
                if category and confidence is not None:
                    color = COLORS.get(category, (255, 255, 255))
                    label = f"{category} ID:{track_id} {confidence:.1%}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Count new tracks only if confidence > threshold
                if (track_id not in counted_tracks and 
                    category and 
                    confidence is not None and 
                    confidence >= args.threshold):
                    
                    counted_tracks.add(track_id)
                    category_counts[category] += 1
                    new_detections = True
                    print(f"New {category} detected! Total: {category_counts[category]}")

            # Get current power consumption for display
            current_power, energy_consumed = get_power_consumption()

            # Cloud sync logic
            if new_detections or (current_time - last_sent_time) > 5:
                if send_to_cloud():
                    last_sent_time = current_time
                    cv2.putText(frame, "Data Sent!", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display info
            fps = frame_count / (current_time - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display power information
            if bus:
                cv2.putText(frame, f"Power: {current_power:.2f}W", (50, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Energy: {energy_consumed:.3f}Wh", (50, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            y_offset = 190
            for category, count in category_counts.items():
                cv2.putText(frame, f"{category.capitalize()}: {count}", 
                           (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, COLORS.get(category, (255, 255, 255)), 2)
                y_offset += 30

            cv2.imshow("Object Counter", frame)

            key = cv2.waitKey(1)
            if key == ord('f'):
                fullscreen = not fullscreen
                cv2.setWindowProperty("Object Counter", cv2.WND_PROP_FULLSCREEN,
                                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key == 27:  # ESC key
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if bus:
            bus.close()
        print("Final counts:", category_counts)
        print(f"Total energy consumed: {total_energy_wh:.3f} Wh")

if __name__ == '__main__':
    main()

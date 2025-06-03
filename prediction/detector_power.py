import argparse
import cv2
import os
import numpy as np
import requests
import time
import glob
import threading
import queue
from collections import deque
from datetime import datetime
from smbus2 import SMBus
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
from sort import Sort

# Cloud API configuration
CLOUD_API_URL = "https://api.v-tally.com/store"

# I2C & Register Definitions
INA226_ADDR        = 0x50    # Found via `i2cdetect -y 1`
REG_CONFIG         = 0x00    # Configuration (R/W)
REG_SHUNT_VOLTAGE  = 0x01    # Differential input (R)
REG_BUS_VOLTAGE    = 0x02    # Bus voltage (R)

# Measurement Parameters
R_SHUNT_OHMS       = 0.1          # Ohms
LSB_SHUNT_VOLTAGE  = 2.5e-6       # 2.5 uV per bit
LSB_BUS_VOLTAGE    = 1.25e-3      # 1.25 mV per bit
CONFIG_VALUE       = 0x4127       # Continuous shunt+bus conversions

# Averaging Settings
AVG_COUNT = 10                    # Number of current samples to average

# Video Recording Settings
RECORDING_DIR = "recordings"       # Directory to save recordings
RECORDING_RESOLUTION = (640, 360)  # Lower resolution for recording (16:9 aspect ratio)
RECORDING_FPS = 15                 # Lower FPS for recording to save space
RECORDING_DURATION = 300           # 5 minutes in seconds
MAX_RECORDINGS = 100               # Maximum number of recordings to keep

# Cloud sync settings
CLOUD_SYNC_INTERVAL = 5           # Try to sync every 5 seconds
CLOUD_RETRY_INTERVAL = 30         # Retry failed syncs every 30 seconds
CLOUD_TIMEOUT = 2                 # Reduced timeout for faster failure detection
MAX_QUEUE_SIZE = 100              # Maximum number of data points to queue

# Region of Interest (ROI) settings
ROI_ENABLED = True                # Enable ROI filtering
roi_box = None                    # Will be set based on frame dimensions

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

# Track persistence for reducing duplicate counting
track_persistence = {}  # track_id -> frame_count_when_first_seen

# NEW: Track lifecycle management to prevent duplicate counting
track_lifecycle = {}  # track_id -> {'first_seen': frame, 'last_seen': frame, 'counted': bool, 'category': str, 'in_roi': bool}
TRACK_CLEANUP_INTERVAL = 150  # Clean up old tracks every N frames
TRACK_MEMORY_FRAMES = 900     # Remember tracks for this many frames (~30 seconds at 30fps)

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

# Video recording variables
video_writer = None
recording_start_time = None
current_recording_path = None

# Cloud sync variables
cloud_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
cloud_thread = None
cloud_running = False
last_sync_attempt = 0
connection_status = "Unknown"
last_successful_sync = 0

def setup_roi(frame_width, frame_height):
    """Set up the Region of Interest box (half the screen width, centered)"""
    global roi_box
    
    # ROI covers half the screen width, centered horizontally
    roi_width = frame_width // 2
    roi_height = frame_height
    roi_x = (frame_width - roi_width) // 2
    roi_y = 0
    
    roi_box = {
        'x1': roi_x,
        'y1': roi_y,
        'x2': roi_x + roi_width,
        'y2': roi_y + roi_height
    }
    
    print(f"ROI set up: x1={roi_x}, y1={roi_y}, x2={roi_x + roi_width}, y2={roi_y + roi_height}")
    print(f"ROI dimensions: {roi_width}x{roi_height} (Frame: {frame_width}x{frame_height})")

def is_in_roi(x1, y1, x2, y2):
    """Check if a bounding box intersects with the ROI"""
    if not ROI_ENABLED or roi_box is None:
        return True
    
    # Check if bounding box intersects with ROI
    # No intersection if one rectangle is completely to the left, right, above, or below the other
    if (x2 < roi_box['x1'] or x1 > roi_box['x2'] or 
        y2 < roi_box['y1'] or y1 > roi_box['y2']):
        return False
    
    return True

def get_bbox_center(x1, y1, x2, y2):
    """Get the center point of a bounding box"""
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def is_center_in_roi(x1, y1, x2, y2):
    """Check if the center of a bounding box is within the ROI"""
    if not ROI_ENABLED or roi_box is None:
        return True
    
    center_x, center_y = get_bbox_center(x1, y1, x2, y2)
    
    return (roi_box['x1'] <= center_x <= roi_box['x2'] and 
            roi_box['y1'] <= center_y <= roi_box['y2'])

def draw_roi(frame):
    """Draw the ROI box on the frame"""
    if not ROI_ENABLED or roi_box is None:
        return
    
    # Draw ROI rectangle with semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                  (roi_box['x1'], roi_box['y1']), 
                  (roi_box['x2'], roi_box['y2']), 
                  (0, 255, 255),  # Yellow
                  -1)  # Fill the rectangle
    
    # Blend the overlay with the original frame (make it semi-transparent)
    cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
    
    # Draw ROI border
    cv2.rectangle(frame, 
                  (roi_box['x1'], roi_box['y1']), 
                  (roi_box['x2'], roi_box['y2']), 
                  (0, 255, 255),  # Yellow
                  3)
    
    # Add ROI label
    cv2.putText(frame, "COUNTING ZONE", 
                (roi_box['x1'] + 10, roi_box['y1'] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# I2C Helper Functions
def read_reg(bus, reg):
    try:
        raw = bus.read_word_data(INA226_ADDR, reg)
        # Swap bytes (little-endian to big-endian)
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
    # Two's-complement conversion
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

def setup_recording_directory():
    """Create recordings directory if it doesn't exist"""
    if not os.path.exists(RECORDING_DIR):
        os.makedirs(RECORDING_DIR)
        print(f"Created recording directory: {RECORDING_DIR}")

def cleanup_old_recordings():
    """Remove oldest recordings if we exceed MAX_RECORDINGS"""
    recording_files = glob.glob(os.path.join(RECORDING_DIR, "recording_*.mp4"))
    recording_files.sort(key=os.path.getctime)  # Sort by creation time
    
    while len(recording_files) >= MAX_RECORDINGS:
        oldest_file = recording_files[0]
        try:
            os.remove(oldest_file)
            print(f"Deleted old recording: {os.path.basename(oldest_file)}")
            recording_files.pop(0)
        except Exception as e:
            print(f"Error deleting {oldest_file}: {e}")
            break

def start_new_recording():
    """Start a new video recording"""
    global video_writer, recording_start_time, current_recording_path
    
    # Stop current recording if active
    if video_writer is not None:
        video_writer.release()
        if current_recording_path:
            print(f"Saved recording: {os.path.basename(current_recording_path)}")
    
    # Clean up old recordings
    cleanup_old_recordings()
    
    # Create new recording filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_recording_path = os.path.join(RECORDING_DIR, f"recording_{timestamp}.mp4")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        current_recording_path, 
        fourcc, 
        RECORDING_FPS, 
        RECORDING_RESOLUTION
    )
    
    recording_start_time = time.time()
    print(f"Started new recording: {os.path.basename(current_recording_path)}")

def stop_recording():
    """Stop the current recording and save the file"""
    global video_writer, current_recording_path
    
    if video_writer is not None:
        video_writer.release()
        video_writer = None
        if current_recording_path:
            print(f"Recording stopped and saved: {os.path.basename(current_recording_path)}")
        current_recording_path = None

def record_frame(frame, frame_count):
    """Record a frame to the current video file (only every few frames to reduce overhead)"""
    global video_writer, recording_start_time
    
    if video_writer is None:
        return
    
    # Only record every 2nd frame to reduce CPU overhead (7.5 FPS effective recording rate)
    if frame_count % 2 != 0:
        return
    
    try:
        # Resize frame to recording resolution
        recording_frame = cv2.resize(frame, RECORDING_RESOLUTION)
        video_writer.write(recording_frame)
        
        # Check if we need to start a new recording (5 minutes elapsed)
        if time.time() - recording_start_time >= RECORDING_DURATION:
            start_new_recording()
            
    except Exception as e:
        print(f"Recording error: {e}")

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

def cleanup_old_tracks(current_frame):
    """Clean up old track lifecycle data to prevent memory issues"""
    global track_lifecycle, counted_tracks, track_persistence, track_id_class_map
    
    tracks_to_remove = []
    for track_id, info in track_lifecycle.items():
        if current_frame - info['last_seen'] > TRACK_MEMORY_FRAMES:
            tracks_to_remove.append(track_id)
    
    for track_id in tracks_to_remove:
        track_lifecycle.pop(track_id, None)
        track_persistence.pop(track_id, None)
        track_id_class_map.pop(track_id, None)
        # Note: We don't remove from counted_tracks to maintain count accuracy

def should_count_track(track_id, category, confidence, current_frame, threshold, bbox):
    """Determine if a track should be counted based on improved logic with ROI"""
    global track_lifecycle, counted_tracks, track_persistence
    
    # Skip if already counted
    if track_id in counted_tracks:
        return False
    
    # Skip if confidence too low
    if confidence < threshold:
        return False
    
    # Check if object is in ROI (using center point method for more accurate counting)
    x1, y1, x2, y2 = bbox
    in_roi = is_center_in_roi(x1, y1, x2, y2)
    
    # Initialize or update track lifecycle
    if track_id not in track_lifecycle:
        track_lifecycle[track_id] = {
            'first_seen': current_frame,
            'last_seen': current_frame,
            'counted': False,
            'category': category,
            'confidence_history': [confidence],
            'in_roi': in_roi,
            'was_in_roi': in_roi  # Track if it was ever in ROI
        }
        
        # Only count if in ROI
        if in_roi:
            return True
        else:
            return False
    else:
        # Update existing track
        track_info = track_lifecycle[track_id]
        track_info['last_seen'] = current_frame
        track_info['confidence_history'].append(confidence)
        
        # Keep only recent confidence values
        if len(track_info['confidence_history']) > 5:
            track_info['confidence_history'] = track_info['confidence_history'][-5:]
        
        # Update category if needed
        track_info['category'] = category
        
        # Update ROI status
        prev_in_roi = track_info['in_roi']
        track_info['in_roi'] = in_roi
        
        # Track if it was ever in ROI
        if in_roi:
            track_info['was_in_roi'] = True
        
        # Don't count again if already counted
        return False

def cloud_worker():
    """Background thread worker for handling cloud sync without blocking main thread"""
    global cloud_running, connection_status, last_successful_sync
    
    while cloud_running:
        try:
            # Get data from queue with timeout
            data = cloud_queue.get(timeout=1.0)
            
            try:
                # Attempt to send data with short timeout
                response = requests.post(CLOUD_API_URL, json=data, timeout=CLOUD_TIMEOUT)
                response.raise_for_status()
                
                connection_status = "Connected"
                last_successful_sync = time.time()
                print(f"? Cloud sync: {data} (Queue: {cloud_queue.qsize()})")
                
                # Mark task as done
                cloud_queue.task_done()
                
            except requests.exceptions.RequestException as e:
                connection_status = f"Error: {type(e).__name__}"
                print(f"? Cloud sync failed: {e}")
                
                # Put data back in queue if there's space (retry logic)
                try:
                    cloud_queue.put_nowait(data)
                except queue.Full:
                    print("Cloud queue full, dropping oldest data")
                    try:
                        cloud_queue.get_nowait()  # Remove oldest
                        cloud_queue.put_nowait(data)  # Add current
                    except queue.Empty:
                        pass
                
                # Mark task as done even if failed
                cloud_queue.task_done()
                
                # Sleep longer on connection errors to avoid spam
                time.sleep(min(30, max(5, time.time() - last_successful_sync)))
                
        except queue.Empty:
            # No data to process, continue loop
            continue
        except Exception as e:
            print(f"Cloud worker error: {e}")
            time.sleep(1)

def queue_cloud_data():
    """Queue data for cloud sync without blocking main thread"""
    current_power, energy_consumed = get_power_consumption()
    
    data = {
        "car_count": category_counts["car"],
        "truck_count": category_counts["truck"],
        "bike_count": category_counts["bike"],
        "ped_count": category_counts["person"],
        "battery_percentage": round(energy_consumed, 3)
    }
    
    try:
        # Non-blocking queue add
        cloud_queue.put_nowait(data)
        return True
    except queue.Full:
        print("Cloud queue full, dropping data")
        # Remove oldest item and add new one
        try:
            cloud_queue.get_nowait()
            cloud_queue.put_nowait(data)
            return True
        except queue.Empty:
            return False

def start_cloud_sync():
    """Start the background cloud sync thread"""
    global cloud_thread, cloud_running
    
    cloud_running = True
    cloud_thread = threading.Thread(target=cloud_worker, daemon=True)
    cloud_thread.start()
    print("Cloud sync thread started")

def stop_cloud_sync():
    """Stop the background cloud sync thread"""
    global cloud_running, cloud_thread
    
    cloud_running = False
    if cloud_thread and cloud_thread.is_alive():
        print("Stopping cloud sync thread...")
        cloud_thread.join(timeout=5)
        if cloud_thread.is_alive():
            print("Warning: Cloud thread did not stop gracefully")

def main():
    global bus, last_sync_attempt, ROI_ENABLED
    
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
    parser.add_argument('--no-recording', action='store_true',
                        help='Disable video recording')
    parser.add_argument('--no-cloud', action='store_true',
                        help='Disable cloud sync')
    parser.add_argument('--no-roi', action='store_true',
                        help='Disable Region of Interest filtering')
    args = parser.parse_args()

    # Set ROI status based on argument
    ROI_ENABLED = not args.no_roi

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

    # Initialize cloud sync
    if not args.no_cloud:
        start_cloud_sync()
    else:
        print("Cloud sync disabled")

    # Initialize video recording
    if not args.no_recording:
        setup_recording_directory()
        start_new_recording()
        print(f"Video recording enabled - Resolution: {RECORDING_RESOLUTION}, FPS: {RECORDING_FPS}")
        print(f"Recording duration: {RECORDING_DURATION//60} minutes, Max recordings: {MAX_RECORDINGS}")
    else:
        print("Video recording disabled")

    # Initialize video capture
    cap = cv2.VideoCapture(args.camera_idx)
    cap.set(3, 1280)  # Width
    cap.set(4, 720)   # Height

    # Initialize SORT tracker with permissive parameters to ensure tracks are created
    tracker = Sort(max_age=15, min_hits=1, iou_threshold=0.3)

    # FPS variables
    start_time = time.time()
    frame_count = 0
    last_sent_time = time.time()
    roi_initialized = False
    
    cv2.namedWindow("Object Counter", cv2.WINDOW_NORMAL)
    fullscreen = False

    print("Starting detection and tracking...")
    print("Tracking categories: car, truck, bike, person")
    if ROI_ENABLED:
        print("Region of Interest: ENABLED (counting zone = half screen width, centered)")
    else:
        print("Region of Interest: DISABLED")
    if bus:
        print("Power monitoring: ENABLED")
    if not args.no_recording:
        print("Video recording: ENABLED")
    if not args.no_cloud:
        print("Cloud sync: ENABLED (non-blocking)")
    print("Press 'f' for fullscreen, 'r' to toggle ROI, 'ESC' to quit")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = time.time()
            new_detections = False

            # Initialize ROI on first frame
            if not roi_initialized and ROI_ENABLED:
                height, width = frame.shape[:2]
                setup_roi(width, height)
                roi_initialized = True

            # Clean up old tracks periodically
            if frame_count % TRACK_CLEANUP_INTERVAL == 0:
                cleanup_old_tracks(frame_count)

            # Record frame if recording is enabled (reduced frequency to preserve performance)
            if not args.no_recording:
                record_frame(frame, frame_count)

            # Process detections every 2 frames for better performance while maintaining tracking quality
            if frame_count % 2 == 0:
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

            # Draw ROI first (so it appears behind objects)
            if ROI_ENABLED:
                draw_roi(frame)

            # Process tracks and draw boxes every frame
            for track in trackers:
                x1, y1, x2, y2, track_id = map(int, track[:5])
                
                # Get class and confidence information
                track_info = track_id_class_map.get(track_id, (None, None))
                class_id, confidence = track_info
                category = classify_object(class_id, labels) if class_id is not None else None
                
                # Draw bounding box and label if confidence is sufficient
                if category and confidence is not None:
                    # Check if object is in ROI
                    in_roi = is_center_in_roi(x1, y1, x2, y2) if ROI_ENABLED else True
                    
                    # Choose color based on ROI status
                    base_color = COLORS.get(category, (255, 255, 255))
                    if ROI_ENABLED:
                        # Dim the color if outside ROI
                        color = base_color if in_roi else tuple(c // 3 for c in base_color)
                        roi_status = " [IN]" if in_roi else " [OUT]"
                    else:
                        color = base_color
                        roi_status = ""
                    
                    # Add status indicator for counted tracks
                    count_status = " ?" if track_id in counted_tracks else ""
                    label = f"{category} ID:{track_id} {confidence:.1%}{roi_status}{count_status}"
                    
                    # Draw thicker border for objects in ROI
                    thickness = 3 if (in_roi and ROI_ENABLED) else 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Improved counting logic with ROI
                if (category and 
                    confidence is not None and 
                    should_count_track(track_id, category, confidence, frame_count, args.threshold, (x1, y1, x2, y2))):
                    
                    counted_tracks.add(track_id)
                    track_lifecycle[track_id]['counted'] = True
                    category_counts[category] += 1
                    new_detections = True
                    roi_msg = " (in ROI)" if ROI_ENABLED else ""
                    print(f"New {category} detected{roi_msg}! ID:{track_id} (conf: {confidence:.1%}) Total: {category_counts[category]}")

            # Get current power consumption for display
            current_power, energy_consumed = get_power_consumption()

            # Non-blocking cloud sync logic
            if not args.no_cloud and (new_detections or (current_time - last_sent_time) > CLOUD_SYNC_INTERVAL):
                if queue_cloud_data():
                    last_sent_time = current_time

            # Display info
            fps = frame_count / (current_time - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display ROI status
            if ROI_ENABLED:
                cv2.putText(frame, "ROI: ENABLED", (50, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "ROI: DISABLED", (50, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            
            # Display power information
            if bus:
                cv2.putText(frame, f"Power: {current_power:.2f}W", (50, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Energy: {energy_consumed:.3f}Wh", (50, 190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display connection status
            if not args.no_cloud:
                queue_size = cloud_queue.qsize()
                status_color = (0, 255, 0) if connection_status == "Connected" else (0, 0, 255)
                y_pos = 220 if bus else 160
                cv2.putText(frame, f"Cloud: {connection_status}", (50, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                if queue_size > 0:
                    cv2.putText(frame, f"Queue: {queue_size}", (50, y_pos + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Display recording status
            if not args.no_recording and recording_start_time:
                elapsed = int(time.time() - recording_start_time)
                remaining = RECORDING_DURATION - elapsed
                y_pos = 280 if not args.no_cloud and bus else (240 if bus or not args.no_cloud else 190)
                cv2.putText(frame, f"Recording: {elapsed//60:02d}:{elapsed%60:02d}", 
                           (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display tracking statistics
            active_tracks = len([t for t in track_lifecycle.values() if frame_count - t['last_seen'] < 30])
            roi_tracks = len([t for t in track_lifecycle.values() 
                            if frame_count - t['last_seen'] < 30 and t.get('in_roi', False)]) if ROI_ENABLED else active_tracks
            
            y_pos = 310 if not args.no_cloud and bus else (270 if bus or not args.no_cloud else 220)
            if ROI_ENABLED:
                cv2.putText(frame, f"Active Tracks: {active_tracks} (ROI: {roi_tracks})", 
                           (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            else:
                cv2.putText(frame, f"Active Tracks: {active_tracks}", 
                           (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Display category counts
            y_offset = y_pos + 30
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
            elif key == ord('r'):
                # Toggle ROI
                ROI_ENABLED = not ROI_ENABLED
                if ROI_ENABLED:
                    height, width = frame.shape[:2]
                    setup_roi(width, height)
                    print("ROI enabled")
                else:
                    print("ROI disabled")
            elif key == 27:  # ESC key
                break

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if not args.no_recording:
            stop_recording()
        if not args.no_cloud:
            stop_cloud_sync()
        if bus:
            bus.close()
        print("Final counts:", category_counts)
        print(f"Total energy consumed: {total_energy_wh:.3f} Wh")
        if not args.no_recording:
            recording_count = len(glob.glob(os.path.join(RECORDING_DIR, "recording_*.mp4")))
            print(f"Total recordings saved: {recording_count}")

if __name__ == '__main__':
    main()

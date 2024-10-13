from ultralytics import YOLO

# Load the YOLOv8 model that you trained
model = YOLO('./prediction/content/runs/detect/train/weights/last.pt')

# Export the model to TFLite with integer quantization for Edge TPU
model.export(format="tflite", int8=True)  # This will create a 'last_full_integer_quant.tflite' file

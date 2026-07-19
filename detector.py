#!/usr/bin/env python3
"""
AI-Powered Self-Checkout Detector
Uses TensorFlow Lite FOMO model for real-time product detection

FIXES:
- Correct FOMO INT8 output parsing
- Proper spatial grid handling
- Dequantization formula applied correctly
- 5-second cooldown between detections
- Persistent transaction ID
"""

import os
import sys
import time
import cv2
import numpy as np
import requests
import threading
from collections import defaultdict
from datetime import datetime

try:
    import tensorflow as tf
except ImportError:
    print("Error: TensorFlow not installed. Install with: pip install tensorflow")
    sys.exit(1)

# Configuration
DJANGO_API_BASE = "http://127.0.0.1:8000/api"
MODEL_PATH = "model.tflite"
CONFIDENCE_THRESHOLD = 0.70
COOLDOWN_SECONDS = 5
CAMERA_INDEX = 0
INPUT_SIZE = 96

# Class labels in order matching the model output
CLASS_LABELS = [
    "chocolate_chips",
    "coca_cola_500ml",
    "kalypp_150ml",
    "small_bread",
    "voltic_500ml"
]

# Global state
last_detection_time = defaultdict(float)
current_transaction_id = None
inference_interpreter = None
input_details = None
output_details = None


def load_model():
    """
    Load the TFLite model and return interpreter
    """
    global inference_interpreter, input_details, output_details
    
    print(f"[INIT] Loading model from {MODEL_PATH}...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print(f"[ERROR] Make sure model.tflite is in: {os.path.abspath(MODEL_PATH)}")
        sys.exit(1)
    
    try:
        inference_interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        inference_interpreter.allocate_tensors()
        
        input_details = inference_interpreter.get_input_details()
        output_details = inference_interpreter.get_output_details()
        
        print(f"[INIT] ✓ Model loaded successfully!")
        print(f"[INIT] Input shape: {input_details[0]['shape']}")
        print(f"[INIT] Input dtype: {input_details[0]['dtype']}")
        print(f"[INIT] Output shape: {output_details[0]['shape']}")
        print(f"[INIT] Output dtype: {output_details[0]['dtype']}")
        print(f"[INIT] Output quantization: {output_details[0]['quantization']}")
        print(f"[INIT] Number of classes: {len(CLASS_LABELS)}")
        print(f"[INIT] Class labels: {CLASS_LABELS}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def preprocess_frame(frame):
    """
    Preprocess frame for model input
    """
    # Resize to model input size
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to uint8
    normalized = rgb_frame.astype(np.uint8)
    
    # Add batch dimension
    batch_frame = np.expand_dims(normalized, axis=0)
    
    return batch_frame


def dequantize_output(quantized_value, quantization_params):
    """
    Dequantize INT8 output
    Formula: float_val = (int8_val - zero_point) * scale
    """
    scale, zero_point = quantization_params
    
    if scale is None or scale == 0:
        return float(quantized_value)
    
    float_value = (int(quantized_value) - int(zero_point)) * float(scale)
    return float_value


def run_inference(frame):
    """
    Run inference on frame using FOMO model
    Returns: (class_label, confidence) or (None, None)
    """
    global inference_interpreter, input_details, output_details
    
    try:
        # Preprocess
        input_data = preprocess_frame(frame)
        
        # Set input and run inference
        inference_interpreter.set_tensor(input_details[0]['index'], input_data)
        inference_interpreter.invoke()
        
        # Get output
        output_data = inference_interpreter.get_tensor(output_details[0]['index'])
        
        # Parse FOMO output: (1, grid_h, grid_w, num_classes)
        batch_size, grid_h, grid_w, num_classes = output_data.shape
        
        # Reshape to (grid_h * grid_w, num_classes)
        output_flat = output_data[0].reshape(-1, num_classes)
        
        # Find cell with highest activation
        max_index = np.argmax(output_flat)
        max_value = output_flat.flatten()[max_index]
        
        # Get class index
        grid_cell_idx = max_index // num_classes
        class_idx = max_index % num_classes
        
        # Dequantize confidence
        quantization = output_details[0]['quantization']
        confidence = dequantize_output(max_value, quantization)
        
        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        if frame_count % 30 == 0:
            print(f"[INFERENCE] Class: {class_idx} ({CLASS_LABELS[class_idx]}), "
                  f"Confidence: {confidence:.3f}")
        
        # Check threshold
        if confidence >= CONFIDENCE_THRESHOLD:
            class_label = CLASS_LABELS[class_idx]
            return class_label, confidence
        else:
            return None, None
            
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def check_cooldown(class_label):
    """
    Check if enough time passed since last detection
    """
    current_time = time.time()
    last_time = last_detection_time.get(class_label, 0)
    
    if current_time - last_time >= COOLDOWN_SECONDS:
        last_detection_time[class_label] = current_time
        return True
    
    return False


def notify_django(class_label, confidence):
    """
    Send detection to Django API
    """
    global current_transaction_id
    
    try:
        # Step 1: Get product info
        print(f"[API] Fetching product for '{class_label}'...")
        detect_response = requests.post(
            f"{DJANGO_API_BASE}/detect/",
            json={"label": class_label, "confidence": float(confidence)},
            timeout=5
        )
        
        if detect_response.status_code != 200:
            print(f"[ERROR] Detect API failed: {detect_response.status_code}")
            return False
        
        product_data = detect_response.json()
        product_id = product_data.get('id')
        product_name = product_data.get('name')
        product_price = product_data.get('price')
        
        print(f"[API] Product: {product_name} (GHS {product_price})")
        
        # Step 2: Get or create transaction
        if current_transaction_id is None:
            print(f"[API] Getting latest transaction...")
            trans_response = requests.get(
                f"{DJANGO_API_BASE}/latest-transaction/",
                timeout=5
            )
            
            if trans_response.status_code == 200:
                trans_data = trans_response.json()
                current_transaction_id = trans_data.get('id')
                print(f"[API] Transaction ID: {current_transaction_id}")
            else:
                print(f"[ERROR] Failed to get transaction")
                return False
        
        # Step 3: Add to cart
        print(f"[API] Adding to cart...")
        cart_response = requests.post(
            f"{DJANGO_API_BASE}/add-to-cart/",
            json={
                "transaction_id": current_transaction_id,
                "product_id": product_id,
                "quantity": 1,
                "unit_price": product_price
            },
            timeout=5
        )
        
        if cart_response.status_code == 201:
            print(f"[API] ✓ Item added to cart!\n")
            return True
        else:
            print(f"[ERROR] Add-to-cart failed: {cart_response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


frame_count = 0

def main():
    """
    Main detection loop
    """
    global current_transaction_id, frame_count
    
    print("="*60)
    print("AI-POWERED SELF-CHECKOUT DETECTOR")
    print("="*60)
    
    # Load model
    load_model()
    
    # Initialize camera
    print(f"[INIT] Initializing camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print(f"[ERROR] Failed to open camera")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"[INIT] ✓ Camera ready!")
    print(f"[INIT] Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"[INIT] Cooldown: {COOLDOWN_SECONDS}s")
    print("="*60)
    print("Starting detection... Press Ctrl+C to stop.\n")
    
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("[ERROR] Failed to read frame")
                break
            
            frame_count += 1
            
            # Run inference every 5 frames
            if frame_count % 5 == 0:
                class_label, confidence = run_inference(frame)
                
                if class_label is not None:
                    if check_cooldown(class_label):
                        print(f"[DETECTION] ✓ {class_label} @ {confidence*100:.1f}%")
                        success = notify_django(class_label, confidence)
                    else:
                        remaining = COOLDOWN_SECONDS - (time.time() - last_detection_time[class_label])
                        if frame_count % 30 == 0:
                            print(f"[COOLDOWN] {class_label} ({remaining:.1f}s remaining)")
    
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[SHUTDOWN] Detector stopped")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
AI-Powered Self-Checkout Detector
Uses TensorFlow Lite FOMO model for real-time product detection
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
COOLDOWN_SECONDS = 5  # Seconds between detections of the same item
CAMERA_INDEX = 0  # 0 for default webcam
INPUT_SIZE = 96  # FOMO model input size

# Class labels in order matching the model output
CLASS_LABELS = [
    "chocolate_chips",      # 0
    "coca_cola_500ml",      # 1
    "kalypp_150ml",         # 2
    "small_bread",          # 3
    "voltic_500ml"          # 4
]

# Global state
last_detection_time = defaultdict(float)  # Track last detection time per class
current_transaction_id = None
inference_interpreter = None
input_details = None
output_details = None


def load_model():
    """
    Load the TFLite model and return interpreter with input/output details
    """
    global inference_interpreter, input_details, output_details
    
    print(f"[INIT] Loading model from {MODEL_PATH}...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        sys.exit(1)
    
    try:
        inference_interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        inference_interpreter.allocate_tensors()
        
        input_details = inference_interpreter.get_input_details()
        output_details = inference_interpreter.get_output_details()
        
        print(f"[INIT] Model loaded successfully!")
        print(f"[INIT] Input shape: {input_details[0]['shape']}")
        print(f"[INIT] Input dtype: {input_details[0]['dtype']}")
        print(f"[INIT] Output shape: {output_details[0]['shape']}")
        print(f"[INIT] Output dtype: {output_details[0]['dtype']}")
        print(f"[INIT] Output quantization: {output_details[0]['quantization']}")
        print(f"[INIT] Number of classes: {len(CLASS_LABELS)}")
        print(f"[INIT] Class labels: {CLASS_LABELS}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)


def preprocess_frame(frame):
    """
    Preprocess frame for model input:
    - Resize to INPUT_SIZE x INPUT_SIZE
    - Normalize for the model
    """
    # Resize frame to model input size
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    
    # Convert BGR to RGB if needed (most models expect RGB)
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to 0-255 uint8 (model expects this for INT8)
    # If model expects -1 to 1, we'd divide by 127.5 and subtract 1
    normalized = rgb_frame.astype(np.uint8)
    
    # Add batch dimension: (H, W, C) -> (1, H, W, C)
    batch_frame = np.expand_dims(normalized, axis=0)
    
    return batch_frame


def dequantize_output(quantized_value, quantization_params):
    """
    Dequantize INT8 output using scale and zero_point
    Formula: float_val = (int8_val - zero_point) * scale
    """
    scale, zero_point = quantization_params
    
    # Handle case where scale is 0 or None
    if scale is None or scale == 0:
        return float(quantized_value)
    
    float_value = (int(quantized_value) - int(zero_point)) * float(scale)
    return float_value


def run_inference(frame):
    """
    Run inference on frame using FOMO model
    Returns: (class_label, confidence) or (None, None) if no detection
    """
    global inference_interpreter, input_details, output_details
    
    try:
        # Preprocess frame
        input_data = preprocess_frame(frame)
        
        # Set input tensor
        inference_interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        inference_interpreter.invoke()
        
        # Get output tensor
        output_data = inference_interpreter.get_tensor(output_details[0]['index'])
        
        # Debug: Print output shape and sample values
        # print(f"[DEBUG] Output shape: {output_data.shape}")
        # print(f"[DEBUG] Output dtype: {output_data.dtype}")
        # print(f"[DEBUG] Output sample (first 5 values): {output_data.flatten()[:5]}")
        
        # FOMO output format: (1, grid_h, grid_w, num_classes)
        # Reshape to (grid_h * grid_w, num_classes) to find best class activation
        batch_size, grid_h, grid_w, num_classes = output_data.shape
        
        # Flatten spatial dimensions: (grid_h * grid_w, num_classes)
        output_flat = output_data[0].reshape(-1, num_classes)  # Remove batch dim
        
        # Find the cell with the highest activation (max across all grid cells and classes)
        # This represents the strongest detection in the image
        max_index = np.argmax(output_flat)
        max_value = output_flat.flatten()[max_index]
        
        # Convert flat index to (grid_cell_idx, class_idx)
        grid_cell_idx = max_index // num_classes
        class_idx = max_index % num_classes
        
        # Get the quantization parameters for dequantization
        quantization = output_details[0]['quantization']
        
        # Dequantize the confidence score
        confidence = dequantize_output(max_value, quantization)
        
        # Ensure confidence is in valid range [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        print(f"[INFERENCE] Class: {class_idx} ({CLASS_LABELS[class_idx]}), "
              f"Confidence: {confidence:.3f}, Grid cell: {grid_cell_idx}/{grid_h*grid_w}")
        
        # Check if confidence meets threshold
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
    Check if enough time has passed since last detection of this class
    Returns: True if detection should be processed, False if still in cooldown
    """
    current_time = time.time()
    last_time = last_detection_time.get(class_label, 0)
    
    if current_time - last_time >= COOLDOWN_SECONDS:
        last_detection_time[class_label] = current_time
        return True
    
    return False


def notify_django(class_label, confidence):
    """
    Send detection to Django API and add to cart
    """
    global current_transaction_id
    
    try:
        # Step 1: Get product info from Django
        print(f"[API] Fetching product info for '{class_label}'...")
        detect_response = requests.post(
            f"{DJANGO_API_BASE}/detect/",
            json={"label": class_label, "confidence": float(confidence)},
            timeout=5
        )
        
        if detect_response.status_code != 200:
            print(f"[ERROR] Detect API failed: {detect_response.status_code} {detect_response.text}")
            return False
        
        product_data = detect_response.json()
        product_id = product_data.get('id')
        product_name = product_data.get('name')
        product_price = product_data.get('price')
        
        print(f"[API] Product found: {product_name} (${product_price})")
        
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
                print(f"[API] Using transaction ID: {current_transaction_id}")
            else:
                print(f"[ERROR] Failed to get transaction: {trans_response.status_code}")
                return False
        
        # Step 3: Add item to cart
        print(f"[API] Adding item to cart (transaction_id={current_transaction_id})...")
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
            print(f"[API] ✓ Item added to cart successfully!")
            return True
        else:
            print(f"[ERROR] Add-to-cart failed: {cart_response.status_code} {cart_response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API request failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Notification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main detection loop
    """
    global current_transaction_id
    
    print("="*60)
    print("AI-POWERED SELF-CHECKOUT DETECTOR")
    print("="*60)
    
    # Load model
    load_model()
    
    # Initialize camera
    print(f"[INIT] Initializing camera (index={CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print(f"[ERROR] Failed to open camera")
        sys.exit(1)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"[INIT] Camera ready!")
    print(f"[INIT] Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"[INIT] Cooldown: {COOLDOWN_SECONDS} seconds")
    print("="*60)
    print("Starting detection loop... Press Ctrl+C to stop.\n")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("[ERROR] Failed to read frame")
                break
            
            frame_count += 1
            
            # Run inference every 5 frames to reduce CPU load
            if frame_count % 5 == 0:
                class_label, confidence = run_inference(frame)
                
                if class_label is not None:
                    # Check cooldown
                    if check_cooldown(class_label):
                        print(f"\n[DETECTION] ✓✓✓ {class_label} detected at {confidence*100:.1f}% confidence")
                        
                        # Send to Django
                        success = notify_django(class_label, confidence)
                        
                        if success:
                            print(f"[SUCCESS] Item added to cart!\n")
                        else:
                            print(f"[FAILED] Could not add item to cart\n")
                    else:
                        elapsed = time.time() - last_detection_time[class_label]
                        print(f"[COOLDOWN] {class_label} detected but in cooldown "
                              f"({COOLDOWN_SECONDS - elapsed:.1f}s remaining)")
                else:
                    # Only print occasionally to avoid spam
                    if frame_count % 30 == 0:
                        print(f"[SCANNING] No detection above {CONFIDENCE_THRESHOLD} threshold")
            
            # Display frame with FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame locally (optional - comment out if running headless)
            # cv2.imshow('Self-Checkout Detector', frame)
            
            # Press 'q' to quit
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("[SHUTDOWN] Detector stopped")


if __name__ == "__main__":
    main()

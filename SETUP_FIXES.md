# Fix for OpenCV + NumPy compatibility and missing django-environ

You're running into dependency conflicts. Here's the solution:

## Problem 1: OpenCV requires numpy>=2 but TensorFlow needs numpy<2

The issue:
```
opencv-python 4.13.0.92 requires numpy>=2; python_version >= "3.9"
but you have numpy 1.26.4 which is incompatible.
```

## Problem 2: Missing django-environ module

## Solution

### Step 1: Create a fresh virtual environment

```bash
# Remove old venv
rm -rf venv

# Create new venv
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# OR on Windows:
# venv\Scripts\activate
```

### Step 2: Install dependencies in correct order

```bash
# First, upgrade pip
pip install --upgrade pip

# Install numpy 1.26.4 first (for TensorFlow)
pip install numpy==1.26.4

# Install downgraded opencv to match numpy
pip install opencv-python==4.8.1.78

# Now install the rest
pip install tensorflow==2.16.2
pip install django-environ
pip install django
pip install djangorestframework
pip install requests
```

### Step 3: OR use the fixed requirements.txt

We've updated requirements.txt to pin compatible versions:

```bash
pip install -r requirements.txt
```

### Step 4: Ensure model.tflite exists

The detector.py is looking for `model.tflite` in the current directory.

**Make sure you have:**
1. model.tflite (your trained FOMO model)
2. Place it in the root directory: `./self-checkout-system/model.tflite`

If you don't have it:
- Export from Edge Impulse as TFLite INT8
- Copy to project root

### Step 5: Test detector.py

```bash
# Make sure model.tflite exists in current directory
ls -la model.tflite

# Run detector
python3 detector.py
```

You should see:
```
============================================================
AI-POWERED SELF-CHECKOUT DETECTOR
============================================================
[INIT] Loading model from model.tflite...
[INIT] Model loaded successfully!
[INIT] Input shape: (1, 96, 96, 3)
[INIT] Output shape: (1, 12, 12, 5)
```

## Troubleshooting

### If you still get numpy/opencv conflicts:

```bash
# Force reinstall with specific versions
pip install --force-reinstall --no-deps numpy==1.26.4
pip install --force-reinstall --no-deps opencv-python==4.8.1.78
```

### If environ still not found:

```bash
pip install django-environ==0.21.0
```

### If model.tflite not found:

```bash
# Check if file exists
find . -name "model.tflite" -type f

# If not found, you need to:
# 1. Train/export from Edge Impulse
# 2. Download as TFLite INT8 format
# 3. Copy to project root directory
```

## Quick Setup Command

Run this all at once:

```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy==1.26.4
pip install opencv-python==4.8.1.78
pip install -r requirements.txt
echo "✓ Setup complete!"
```

## Then run both services:

**Terminal 1:**
```bash
source venv/bin/activate
python manage.py runserver
```

**Terminal 2:**
```bash
source venv/bin/activate
python detector.py
```

**Terminal 3 (Optional - Celery):**
```bash
source venv/bin/activate
celery -A config worker -l info
```

## Open in browser:
http://127.0.0.1:8000

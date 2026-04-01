#!/bin/bash

# Navigate to the project directory
cd ~/Documents/fuse_project/manf465

# 1. Activate the virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found. Please run this in the project root."
    exit 1
fi

# 2. Fix for 'Illegal Instruction' on Raspberry Pi 4
export OPENBLAS_CORETYPE=ARMV8

# 3. Bridge system hardware drivers (libcamera, pykms) to the venv
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages

# 4. Ensure the OpenCV window opens on the VNC/main desktop
export DISPLAY=:0

echo "------------------------------------------------"
echo "Starting MANF 465 Fuse Detection System..."
echo "Press 'q' in the video window to stop."
echo "------------------------------------------------"

# Run the python script
python3 pi_test.py

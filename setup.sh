#!/bin/bash

echo "=========================================="
echo " VEHICLE DETECTION PROJECT SETUP SCRIPT"
echo "=========================================="
set -e

# Step 1: Update and install system packages
echo "[1/5] Updating package list and installing system dependencies..."
sudo apt update
sudo apt install -y python3 python3-pip python3-venv build-essential ffmpeg libgl1 libglib2.0-0

# Step 2: Set up Python virtual environment
echo "[2/5] Setting up Python virtual environment..."
python3 -m venv venv

# Activate the virtual environment for this script
source venv/bin/activate

# Step 3: Upgrade pip and install Python dependencies
echo "[3/5] Upgrading pip and installing Python packages..."
pip install --upgrade pip
pip install flask opencv-python numpy supervision ultralytics easyocr mysql-connector-python

# Step 4: (Optional) Install MySQL server for local use
echo "[4/5] Installing MySQL server (optional)..."
sudo apt install -y mysql-server
sudo service mysql start
echo "MySQL server installed. You may need to run 'sudo mysql_secure_installation' for security."
echo "Remember to create the 'veh_logs' database before running the app:"
echo "    sudo mysql -u root"
echo "    CREATE DATABASE veh_logs;"
echo "    exit;"
echo "MySQL server is now running."

# Step 6: Final notes and how to run
echo "[5/5] Setup complete!"
echo "To activate your environment and run the app:"
echo ""
echo "    source venv/bin/activate"
echo "    python app.py"
echo ""
echo "Open http://localhost:5000 in your browser."
echo ""
echo "If you need to install extra system libraries for OpenCV/EasyOCR image display, try:"
echo "    sudo apt install libgl1 libglib2.0-0"
echo ""
echo "=========================================="
echo "           SETUP FINISHED!"
echo "=========================================="

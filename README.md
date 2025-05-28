# Vehicle Detection and License Plate Recognition System

## Overview
A Flask-based web application for real-time vehicle detection, classification, speed estimation, and license plate recognition using deep learning models. The system:

- Classifies vehicles into Military and Civilian categories
- Performs OCR on license plates
- Provides real-time monitoring and analytics dashboard
- Supports both video files and RTSP streams

## Features
- 🚗 Real-time vehicle detection and tracking (YOLOv8 + Supervision)
- 🏃 Speed estimation with perspective correction
- 📝 License plate detection and OCR (EasyOCR)
- 💾 Automatic image capture and storage
- 📊 Analytics dashboard with search capabilities
- 🔄 Support for video files and RTSP streams
- ✏️ Manual result editing before database storage

## Requirements
- Python 3.8+
- MySQL Server (optional)
- CUDA-capable GPU (recommended)
- Required model weights (YOLOv8)

## Installation

1. **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2. **Setup Environment**
    ```bash
    # Run the setup script
    bash setup.sh

    # Or manually install dependencies
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. **Prepare Model Weights**
    - Vehicle classification: `/workspace/data/runs/results/veh_cls/weights/best.pt`
    - License plate detection: `runs/detect/license3/weights/best.pt`

4. **Configure MySQL (Optional)**
    ```sql
    CREATE DATABASE veh_logs;
    ```

## Project Structure
    ```
    .
    ├── app.py                  # Flask application
    ├── wkg_with_sv.py         # Core detection logic
    ├── license.py             # License plate processing
    ├── templates/             # UI templates
    │   ├── dashboard.html     # Analytics dashboard
    │   └── index.html        # Main interface
    ├── static/               # Assets and outputs
    └── runs/                 # YOLO models
    ```

## Usage
1. Start the Flask server:
    ```bash
    python app.py
    ```

2. Access the web interface:
    - Main interface: `http://localhost:5000`
    - Dashboard: `http://localhost:5000/dashboard`

## Contributing
Pull requests are welcome. For major changes:
1. Fork the repository
2. Create your feature branch
3. Submit a pull request

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements
- Roboflow-Supervision for tracking utilities
- YOLOv8 for object detection
- EasyOCR for license plate recognition
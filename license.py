import uuid
import os
import cv2
import datetime
import numpy as np
import supervision as sv
import easyocr
from ultralytics import YOLO

current_date = datetime.datetime.now().strftime("%Y%m%d")  # Format: YYYYMMDD

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Initialize YOLO model
# Paths
MODEL_WEIGHTS_PATH = "runs/detect/license3/weights/best.pt"
# Load the YOLO model
model = YOLO(MODEL_WEIGHTS_PATH)
model.overrides['conf'] = 0.6

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0] * region.shape[1]
    plate = []
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        ratio = (length * height) / rectangle_size
        if ratio > region_threshold:
            plate.append(result[1])
    return plate

def ocr_it(image, detections, track_id=None, detection_threshold=0.3, region_threshold=0.3, save_dir="static/license"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]
    recognized_texts = []

    for idx, box in enumerate(boxes):
        y1, x1, y2, x2 = box
        region = image[int(y1):int(y2), int(x1):int(x2)]
        if region.size == 0:
            continue
        
        # Uncomment and adjust the following lines if needed for pre-processing:
        # region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        # region_gray = cv2.equalizeHist(region_gray)
        # region_thresh = cv2.adaptiveThreshold(region_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        ocr_result = reader.readtext(region)
        filtered_text = filter_text(region, ocr_result, region_threshold)
        if filtered_text:
            recognized_text = " ".join(filtered_text)
            recognized_texts.append(recognized_text)

    return recognized_texts

def detect_license_plate(image, track_id=None, veh_save_dir=f"static/{current_date}/veh", license_save_dir=f"static/{current_date}/license"):
    # Ensure the directories exist
    if not os.path.exists(veh_save_dir):
        os.makedirs(veh_save_dir)
    if not os.path.exists(license_save_dir):
        os.makedirs(license_save_dir)
    
    # Define the filename using the track ID with .jpeg extension
    img_name = f"{track_id}.jpeg"
    
    # Save the passed image (vehicle image) in the veh directory
    cv2.imwrite(os.path.join(veh_save_dir, img_name), image)
    
    # Run the YOLO model on the image to detect the license plate
    results = model(image)
    
    detection_boxes = []
    detection_scores = []
    detection_classes = []
    
    best_box = None
    best_score = -1

    for det_result in results:
        for box in det_result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()  # Move GPU->CPU
            x1, y1, x2, y2 = xyxy
            score = float(box.conf[0].cpu().item())
            cls_id = int(box.cls[0].cpu().item())
            
            # Save detection details
            detection_boxes.append([y1, x1, y2, x2])
            detection_scores.append(score)
            detection_classes.append(cls_id)
            
            # Select the best detection based on highest confidence
            if score > best_score:
                best_score = score
                best_box = (int(x1), int(y1), int(x2), int(y2))
    
    detection_boxes = np.array(detection_boxes, dtype=np.float32)
    
    # If a detection was found, crop that region and save in the license directory
    if best_box is not None:
        lx1, ly1, lx2, ly2 = best_box
        license_crop = image[ly1:ly2, lx1:lx2]
        cv2.imwrite(os.path.join(license_save_dir, img_name), license_crop)
    
    results = {
        'detection_boxes': detection_boxes,
        'detection_scores': detection_scores,
        'detection_classes': detection_classes
    }
    
    return results


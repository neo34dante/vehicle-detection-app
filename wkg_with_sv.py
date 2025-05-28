import os
import json
import time
import datetime
import random
import cv2
import re
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque, Counter
from license import detect_license_plate, ocr_it 

################################################################################
# Hard-coded configuration
################################################################################

# Generate a new log file path dynamically
current_date = datetime.datetime.now().strftime("%Y%m%d")  # Format: YYYYMMDD
is_file_processing = False
license_text_cache = {} 

# Paths
MODEL_WEIGHTS_PATH = "/workspace/data/runs/results/veh_cls/weights/best.pt"
LOG_FILE_PATH = f"static/logs/logs_{current_date}.json"
# LOG_FILE_PATH = f"static/logs/logs.json"
# Define class names
CLASS_NAMES = {0: "Mil Veh", 1: "Civil Veh"}

# Confidence and IOU thresholds
CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.7

# Coordinates for polygon zone (adjust based on your use case)
# old SOURCE = np.array([[200, 840], [1733, 890], [2442, 356], [1814, 342]])
# TCP1
# SOURCE = np.array([[128, 337], [420, 272], [1920, 612], [941, 1125]])
# TCP2  
# SOURCE = np.array([[54,740], [1348,1006], [2560,252], [1804,172]])
# DOGRA FORT TCP 2
SOURCE = np.array([[252, 223], [855, 209], [1916, 810], [542, 925]])
TARGET_WIDTH = 7
TARGET_HEIGHT = 70
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

def edit_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings.
    A small distance (< 2) suggests the strings are extremely similar.
    """
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,       # deletion
                dp[i][j-1] + 1,       # insertion
                dp[i-1][j-1] + cost   # substitution
            )
    return dp[m][n]


def normalize_plate_text(txt: str) -> str:
    """
    1) Uppercase
    2) Remove spaces and punctuation for approximate matching
    3) (Optionally unify 'O'->'0', 'I'->'1', etc. if needed)
    """
    txt = txt.upper()
    # Remove all non-alphanumeric except the '↑' if you want to preserve it for mil
    # Or just remove everything non [A-Z0-9]:
    txt = re.sub(r'[^A-Z0-9↑]', '', txt)
    return txt


def finalize_license(candidates: list[str], class_id: int) -> str:
    """
    Pick a single best plate from multiple OCR outputs, factoring in the class_id:
      - class_id=0 -> 'military' plate format:   ↑NNANNNNNNA
      - class_id=1 -> 'civil' plate format:      AANNAANNNN

    1) Cluster by edit_distance of normalized strings.
    2) Choose the cluster with highest total frequency.
    3) Within that cluster, if any raw text matches the class-specific pattern, pick
       the best one by frequency. Otherwise, fallback to the most frequent raw text.
    """
    if not candidates:
        return "Unknown"

    # Define separate regex patterns for each class
    # Example:
    #   class_id=0 (military):
    #       '↑NNANNNNNNA'
    #       i.e. 1 up-arrow + 2 digits + 1 alpha + 5 digits + 1 alpha
    #   class_id=1 (civil):
    #       'AANNAANNNN'
    #       i.e. 2 alpha + 2 digits + 2 alpha + 4 digits

    # For demonstration:
    mil_pattern = r'^↑\d{2}[A-Za-z]\d{5}[A-Za-z]$'   # ex: ↑12A34567B
    civ_pattern = r'^[A-Za-z]{2}\d{2}[A-Za-z]{2}\d{4}$'  # ex: AB12CD3456

    # Pick which pattern to use
    if class_id == 0:
        target_pattern = mil_pattern
    else:
        target_pattern = civ_pattern

    # 1) Count raw occurrences
    raw_counter = Counter(candidates)

    # 2) Build a list of (raw_text, normalized_text, freq)
    items = []
    for txt, freq in raw_counter.items():
        norm = normalize_plate_text(txt)
        items.append((txt, norm, freq))

    # 3) Group them into clusters of "similar" normalized texts using edit_distance
    clusters = []
    for (raw, norm, freq) in items:
        placed = False
        for cluster in clusters:
            # Compare with the first item in the cluster
            _, cluster_norm, _ = cluster[0]
            if edit_distance(norm, cluster_norm) < 2:
                cluster.append((raw, norm, freq))
                placed = True
                break
        if not placed:
            clusters.append([(raw, norm, freq)])

    # 4) Find the cluster with the highest total frequency
    max_freq = 0
    best_cluster = None
    for cluster in clusters:
        cluster_freq = sum(x[2] for x in cluster)
        if cluster_freq > max_freq:
            max_freq = cluster_freq
            best_cluster = cluster

    if not best_cluster:
        return "Unknown"

    # 5) Within that cluster, pick the single raw text with the highest freq
    # But first, see if any raw text matches the pattern for this class_id
    raw_text_subcounter = Counter()
    for (raw, norm, freq) in best_cluster:
        raw_text_subcounter[raw] += freq

    # Sort possible raw texts in this cluster by frequency descending
    sorted_cluster_texts = raw_text_subcounter.most_common()  # list of (raw, freq)

    # 6) Check for matches to the target_pattern
    matching_candidates = []
    for raw_str, f in sorted_cluster_texts:
        # We'll do pattern matching on the raw string. Alternatively, you could match on norm.
        if re.match(target_pattern, raw_str):
            matching_candidates.append((raw_str, f))

    # 7) If any text matches, pick the top by frequency
    if matching_candidates:
        # matching_candidates is already unsorted; let's sort by freq descending
        matching_candidates.sort(key=lambda x: x[1], reverse=True)
        best_raw_text = matching_candidates[0][0]
        return best_raw_text
    else:
        # Fallback: return the highest-frequency raw text from the cluster
        best_raw_text, _ = sorted_cluster_texts[0]
        return best_raw_text

# Function to save logs to the JSON file
def save_logs_dict(logs_dict, file_path=LOG_FILE_PATH):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert all NumPy types to Python native types
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int_)):  # Convert NumPy int to Python int
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_)):  # Convert NumPy float to Python float
            return float(obj)
        elif isinstance(obj, np.ndarray):  # Convert NumPy arrays to lists
            return obj.tolist()
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(file_path, "w") as log_file:
        json.dump(logs_dict, log_file, indent=4, default=convert_to_serializable)

# Initialize an empty logs dictionary
logs_dict = {}

################################################################################
# Perspective Transformation (ViewTransformer)
################################################################################

class ViewTransformer:
    """
    Applies a homography (perspective) transform from a 'source' quadrilateral
    to a 'target' quadrilateral. This allows consistent distance measurement.
    """
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

################################################################################
# Frame-by-Frame Generator
################################################################################

def generate_annotated_frames(source_video_path):
    is_file_processing = True

    """
    Generate annotated frames for rendering and save logs to JSON file.
    Only process license detection if the detected region is fully inside the frame (with a buffer).
    """
    # Clear logs at the start of a new run
    logs_dict = {}
    with open(LOG_FILE_PATH, 'w') as f:
        f.write("[]")

    # Load the YOLO model
    model = YOLO(MODEL_WEIGHTS_PATH)

    # Get video information
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    # Setup ByteTrack tracker
    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps,
        track_activation_threshold=CONFIDENCE_THRESHOLD,
    )

    # Configure annotation details
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    # Frame generator to iterate over video frames
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    # 6. PolygonZone for filtering detections
    polygon_zone = sv.PolygonZone(polygon=SOURCE)

    # 7. Perspective transformation
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    # Initialize storage for speed calculation
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    prev_active_ids = set() 

    buffer_px = 10  # Buffer in pixels

    for frame in frame_generator:
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        # --- Filter Detections ---
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        class_ids_to_keep = [0, 1]
        detections = detections[np.isin(detections.class_id, class_ids_to_keep)]
        detections = detections[polygon_zone.trigger(detections)]
        detections = detections.with_nms(threshold=IOU_THRESHOLD)
        detections = byte_track.update_with_detections(detections=detections)

        # --- Perspective Transform (for speed measurement) ---
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)

        labels = []
        log_type = "uploadLogs" if is_file_processing else "streamLogs"  
        if log_type not in logs_dict:
            logs_dict[log_type] = {}

        current_active_ids = set(detections.tracker_id.tolist())

        # Finalize for tracks that have disappeared
        removed_ids = prev_active_ids - current_active_ids
        for rid in removed_ids:
            if rid in license_text_cache:
                if str(rid) in logs_dict[log_type]:
                    cls_id = logs_dict[log_type][str(rid)]["Class ID"]
                    final_license = finalize_license(license_text_cache[rid], cls_id)
                    print(f"Finalized license for {rid}: {final_license}")
                    logs_dict[log_type][str(rid)]["License"] = final_license
                del license_text_cache[rid]

        # Process each detection
        for tracker_id, bbox in zip(detections.tracker_id, detections.xyxy):
            x1, y1, x2, y2 = bbox.astype(int)
            track_id_str = str(tracker_id)

            # Check if the bounding box is fully inside the frame with buffer
            if (x1 < buffer_px or y1 < buffer_px or 
                x2 > frame_width - buffer_px or y2 > frame_height - buffer_px):
                # Skip license detection for this detection
                labels.append(f"#{tracker_id}")
                continue

            # Extract the region and perform license detection and OCR
            region = frame[y1:y2, x1:x2]
            license_detections = detect_license_plate(region, track_id_str)
            license_texts = ocr_it(region, license_detections)
            if tracker_id not in license_text_cache:
                license_text_cache[tracker_id] = []
            license_text_cache[tracker_id].extend(license_texts)

            # Only compute speed if sufficient frames are accumulated
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
                continue

            coordinate_start = coordinates[tracker_id][-1]
            coordinate_end = coordinates[tracker_id][0]
            distance = abs(coordinate_start - coordinate_end)
            time_elapsed = len(coordinates[tracker_id]) / video_info.fps
            speed_kmh = distance / time_elapsed * 3.6

            cls_id = detections.class_id[0]
            class_name = CLASS_NAMES[cls_id]
            time_of_detection = time.strftime("%Y-%m-%d %H:%M:%S")
            speeds = logs_dict.get(track_id_str, {}).get("speeds", [])
            speeds.append(speed_kmh)
            avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

            log_entry = {
                "Track ID": tracker_id,
                "Class Name": class_name,
                "Avg Speed": f"{speed_kmh:.2f} km/h",
                "License": "",
                "Time": time_of_detection,
                "Class ID": cls_id
            }

            if track_id_str in logs_dict[log_type]:
                logs_dict[log_type][track_id_str].update({
                    "Avg Speed": f"{speed_kmh:.2f}",
                    "Time": time_of_detection
                })
            else:
                logs_dict[log_type][track_id_str] = log_entry

            labels.append(f"#{tracker_id} : {class_name} {int(speed_kmh)} km/h")

        save_logs_dict(logs_dict)
        prev_active_ids = current_active_ids

        # --- Annotation ---
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        _, buffer_img = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer_img.tobytes() + b'\r\n')


def generate_annotated_frames_rtsp(source_video_path):
    """
    Generate annotated frames for rendering from an RTSP stream and save logs to a JSON file.
    Only passes regions fully inside the frame (with a buffer) to license detection and OCR.
    """
    is_file_processing = False
    log_type = "streamLogs"
    logs_dict = {}
    with open(LOG_FILE_PATH, 'w') as f:
        f.write("[]")

    model = YOLO(MODEL_WEIGHTS_PATH)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    if video_info.fps < 1:
        video_info.fps = 24

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps,
        track_activation_threshold=CONFIDENCE_THRESHOLD,
    )

    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream at {source_video_path}")
        return

    print(f"Successfully connected to RTSP stream: {source_video_path}")

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    prev_active_ids = set()

    buffer_px = 10  # Buffer in pixels

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to retrieve frame or stream ended.")
            break

        frame_height, frame_width = frame.shape[:2]
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        class_ids_to_keep = [0, 1]
        detections = detections[np.isin(detections.class_id, class_ids_to_keep)]
        detections = detections[polygon_zone.trigger(detections)]
        detections = detections.with_nms(threshold=IOU_THRESHOLD)
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)

        labels = []
        current_active_ids = set(detections.tracker_id.tolist())

        removed_ids = prev_active_ids - current_active_ids
        for rid in removed_ids:
            if rid in license_text_cache:
                if str(rid) in logs_dict.get(log_type, {}):
                    cls_id = logs_dict[log_type][str(rid)]["Class ID"]
                    final_license = finalize_license(license_text_cache[rid], cls_id)
                    logs_dict[log_type][str(rid)]["License"] = final_license
                    print(f"Finalized license for {rid}: {final_license}")
                del license_text_cache[rid]

        for i in range(len(detections.tracker_id)):
            tracker_id = detections.tracker_id[i]
            bbox = detections.xyxy[i]
            cls_id = detections.class_id[i]
            track_id_str = str(tracker_id)
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Check that the region is fully inside the frame (with buffer)
            if (x1 < buffer_px or y1 < buffer_px or 
                x2 > frame_width - buffer_px or y2 > frame_height - buffer_px):
                labels.append(f"#{tracker_id}")
                continue

            region = frame[y1:y2, x1:x2]
            license_detections = detect_license_plate(region, track_id_str)
            license_texts = ocr_it(region, license_detections)
            if tracker_id not in license_text_cache:
                license_text_cache[tracker_id] = []
            license_text_cache[tracker_id].extend(license_texts)

            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time_elapsed = len(coordinates[tracker_id]) / video_info.fps
                speed_kmh = distance / time_elapsed * 3.6

                class_name = CLASS_NAMES.get(cls_id, "Unknown")
                time_of_detection = time.strftime("%Y-%m-%d %H:%M:%S")
                speeds = logs_dict.get(track_id_str, {}).get("speeds", [])
                speeds.append(speed_kmh)
                avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

                log_entry = {
                    "Track ID": tracker_id,
                    "Class Name": class_name,
                    "Avg Speed": f"{speed_kmh:.2f} km/h",
                    "License": "",
                    "Time": time_of_detection,
                    "Class ID": cls_id
                }
                if log_type not in logs_dict:
                    logs_dict[log_type] = {}
                if track_id_str in logs_dict[log_type]:
                    logs_dict[log_type][track_id_str].update({
                        "Avg Speed": f"{speed_kmh:.2f}",
                        "Time": time_of_detection,
                    })
                else:
                    logs_dict[log_type][track_id_str] = log_entry

                labels.append(f"#{tracker_id} : {class_name} {int(speed_kmh)} km/h")

        save_logs_dict(logs_dict)
        prev_active_ids = current_active_ids

        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        _, buffer_img = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer_img.tobytes() + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()



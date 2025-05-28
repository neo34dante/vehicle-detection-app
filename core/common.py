import os
import json
import datetime
import cv2
import re
import numpy as np
from collections import Counter

# Generate a new log file path dynamically
current_date = datetime.datetime.now().strftime("%Y%m%d")  # Format: YYYYMMDD
is_file_processing = False
license_text_cache = {} 

# Paths
MODEL_WEIGHTS_PATH = "/workspace/data/runs/results/veh_cls/weights/best.pt"
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOG_FILE_PATH = os.path.join(BASE_DIR, "flask_app", "static", "logs", f"logs_{current_date}.json")
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
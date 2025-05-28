import cv2
import numpy as np
import supervision as sv
from collections import defaultdict, deque
from ultralytics import YOLO

from license import detect_license_plate, ocr_it
from wkg_with_sv import (
    MODEL_WEIGHTS_PATH,
    LOG_FILE_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    CLASS_NAMES,
    SOURCE,
    TARGET,
    finalize_license,
    save_logs_dict,
    license_text_cache,
    logs_dict,
    ViewTransformer,
)


def generate_annotated_frames_rtsp(source_video_path):
    """Yield annotated frames from an RTSP stream."""
    is_file_processing = False
    log_type = "streamLogs"
    logs_dict.clear()
    with open(LOG_FILE_PATH, "w") as f:
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
    buffer_px = 10

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
                del license_text_cache[rid]

        for i in range(len(detections.tracker_id)):
            tracker_id = detections.tracker_id[i]
            bbox = detections.xyxy[i]
            cls_id = detections.class_id[i]
            track_id_str = str(tracker_id)
            x1, y1, x2, y2 = map(int, bbox)

            if (
                x1 < buffer_px
                or y1 < buffer_px
                or x2 > frame_width - buffer_px
                or y2 > frame_height - buffer_px
            ):
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
                    "Class ID": cls_id,
                }
                if log_type not in logs_dict:
                    logs_dict[log_type] = {}
                if track_id_str in logs_dict[log_type]:
                    logs_dict[log_type][track_id_str].update(
                        {"Avg Speed": f"{speed_kmh:.2f}", "Time": time_of_detection}
                    )
                else:
                    logs_dict[log_type][track_id_str] = log_entry

                labels.append(f"#{tracker_id} : {class_name} {int(speed_kmh)} km/h")

        save_logs_dict(logs_dict)
        prev_active_ids = current_active_ids

        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        _, buffer_img = cv2.imencode(".jpg", annotated_frame)
        yield (
            b"--frame\r\n" + b"Content-Type: image/jpeg\r\n\r\n" + buffer_img.tobytes() + b"\r\n"
        )

    cap.release()
    cv2.destroyAllWindows()
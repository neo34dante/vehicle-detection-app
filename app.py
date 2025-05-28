from flask import Flask, render_template, request, Response, jsonify
import os
import cv2
import json
import time
import datetime
from wkg_with_sv import generate_annotated_frames, LOG_FILE_PATH, generate_annotated_frames_rtsp

# NEW: Import MySQL connector (ensure you have mysql-connector-python installed)
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)

# Directory to store uploaded videos
UPLOAD_FOLDER = 'static/videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Directory to store processed videos
OUTPUT_FOLDER = 'static/processed_videos'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Directory to store logs
LOGS_FOLDER = 'static/logs'
os.makedirs(LOGS_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """Render the main interface."""
    current_date = datetime.datetime.now().strftime("%Y%m%d")  # Format: YYYYMMDD
    return render_template('index1.html', current_date=current_date) #index1.html

# NEW: Dashboard endpoint (dashboard.html to be created later)
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard1.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """
    Handle video uploads (file or RTSP URL) and return the source path for processing.
    """
    rtsp_url = request.form.get('stream_url')
    if rtsp_url:
        if not rtsp_url.startswith("rtsp://"):
            return jsonify({"error": "Invalid RTSP URL"}), 400
        # Clear previous logs
        with open(LOG_FILE_PATH, 'w') as log_file:
            log_file.write("[]")
        return jsonify({
            "video_path": rtsp_url,
            "logs": [],
        })

    video = request.files.get('video')
    if video:
        video_path = os.path.join(UPLOAD_FOLDER, video.filename)
        video.save(video_path)
        with open(LOG_FILE_PATH, 'w') as log_file:
            log_file.write("[]")
        return jsonify({
            "video_path": os.path.basename(video_path),
            "logs": [],
        })

    return jsonify({"error": "No video or RTSP URL provided"}), 400

@app.route('/process_video/<filename>')
def process_video(filename):
    """
    Process the uploaded video using the generate_annotated_frames function.
    """
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404
    try:
        output_video_path = os.path.join(OUTPUT_FOLDER, f"processed_{filename}")
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 12  # Alternatively, use int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        for frame in generate_annotated_frames(video_path):
            frame_decoded = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
            time.sleep(1)
            out.write(frame_decoded)
        cap.release()
        out.release()
    except Exception as e:
        return jsonify({"error": f"Error processing video: {str(e)}"}), 500
    return jsonify({
        "processed_video_path": f"/{output_video_path}",
        "logs_path": f"/{LOG_FILE_PATH}",
    })

@app.route('/frame_stream/<path:stream_source>')
def frame_stream(stream_source):
    rtsp = False
    if stream_source.startswith("rtsp://"):
        source_path = stream_source
        rtsp = True
    else:
        source_path = os.path.join(UPLOAD_FOLDER, stream_source)
        if not os.path.exists(source_path):
            return jsonify({"error": "Video file not found"}), 404
    def generate():
        if rtsp:
            for frame in generate_annotated_frames_rtsp(source_path):
                yield frame
        else:
            for frame in generate_annotated_frames(source_path):
                yield frame
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )

@app.route('/logs', methods=['GET'])
def fetch_logs():
    logs_dict = {"uploadLogs": {}, "streamLogs": {}}
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, 'r') as log_file:
            try:
                logs_dict = json.load(log_file)
            except json.JSONDecodeError:
                pass
    if isinstance(logs_dict, dict):
        upload_logs = list(logs_dict.get("uploadLogs", {}).values())
        stream_logs = list(logs_dict.get("streamLogs", {}).values())
    return jsonify({
        "uploadLogs": upload_logs,
        "streamLogs": stream_logs
    })

# NEW: Endpoint to save logs to MySQL database
@app.route('/save_logs', methods=['POST'])
def save_logs():
    data = request.get_json()
    if not data or 'logs' not in data:
        return jsonify({"error": "No logs provided"}), 400
    logs = data['logs']
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    table_name = f"veh_log_{current_date}"
    try:
        connection = mysql.connector.connect(
            host="host.docker.internal",
            user="root",
            password="",      # Change if needed
            database="veh_logs"
        )
        cursor = connection.cursor()
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            track_id INT PRIMARY KEY,
            class_name VARCHAR(50),
            avg_speed VARCHAR(20),
            license VARCHAR(100),
            time VARCHAR(50),
            license_img VARCHAR(255),
            veh_img VARCHAR(255),
            class_id INT
        );
        """
        cursor.execute(create_table_query)
        connection.commit()

        insert_query = f"""
        INSERT INTO {table_name} 
        (track_id, class_name, avg_speed, license, time, license_img, veh_img, class_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            class_name=VALUES(class_name),
            avg_speed=VALUES(avg_speed),
            license=VALUES(license),
            time=VALUES(time),
            license_img=VALUES(license_img),
            veh_img=VALUES(veh_img),
            class_id=VALUES(class_id);
        """
        for log in logs:
            track_id = log.get("Track ID")
            class_name = log.get("Class Name")
            avg_speed = log.get("Avg Speed")
            license_val = log.get("License")
            time_val = log.get("Time")
            class_id = log.get("Class ID")
            license_img = f"/static/license/{track_id}.jpeg"
            veh_img = f"/static/veh/{track_id}.jpeg"
            values = (track_id, class_name, avg_speed, license_val, time_val, license_img, veh_img, class_id)
            cursor.execute(insert_query, values)
        connection.commit()
        cursor.close()
        connection.close()
        return jsonify({"status": "success", "message": "Logs saved to database."})
    except Error as e:
        return jsonify({"error": str(e)}), 500
    
# --- New helper function to query one day's table, including overspeeding count ---
def query_table_for_day(day_str):
    table_name = f"veh_log_{day_str}"
    conn = mysql.connector.connect(
         host="host.docker.internal",
         user="root",
         password="",      # Adjust if needed
         database="veh_logs"
    )
    cursor = conn.cursor(dictionary=True)
    query = f"""
        SELECT 
            class_name, 
            COUNT(*) as volume,
            AVG(CAST(REPLACE(avg_speed, ' km/h','') AS DECIMAL(10,2))) as avg_speed,
            SUM(CASE WHEN CAST(REPLACE(avg_speed, ' km/h','') AS DECIMAL(10,2)) > 40 THEN 1 ELSE 0 END) as over_speed_volume
        FROM {table_name}
        GROUP BY class_name
    """
    try:
        cursor.execute(query)
        results = cursor.fetchall()
    except mysql.connector.Error:
        results = []
    cursor.close()
    conn.close()
    return results

# Endpoint to provide dashboard data (last 7 days)
@app.route('/dashboard_data', methods=['GET'])
def dashboard_data():
    today = datetime.date.today()
    # Last 7 days (from 6 days ago until today)
    dates = [(today - datetime.timedelta(days=i)).strftime("%Y%m%d") for i in range(6, -1, -1)]
    
    volumes = {}
    avg_speeds = {}
    over_speed_volumes = {}
    
    # Initialize dictionaries for each day and each class
    for d in dates:
        volumes[d] = {"Mil Veh": 0, "Civil Veh": 0}
        avg_speeds[d] = {"Mil Veh": 0, "Civil Veh": 0}
        over_speed_volumes[d] = {"Mil Veh": 0, "Civil Veh": 0}
        results = query_table_for_day(d)
        if results:
            for row in results:
                class_name = row['class_name']
                volumes[d][class_name] = row['volume']
                avg_speeds[d][class_name] = float(row['avg_speed']) if row['avg_speed'] is not None else 0.0
                over_speed_volumes[d][class_name] = row['over_speed_volume']
    
    data = {
        "dates": dates,
        "volumes": volumes,
        "avg_speeds": avg_speeds,
        "over_speed_volumes": over_speed_volumes
    }
    return jsonify(data)

# Endpoint to search for a license plate (search across last 7 days)
@app.route('/search_license', methods=['GET'])
def search_license():
    query_str = request.args.get('query', '')
    # Remove spaces and tab characters, then uppercase
    query_normalized = query_str.replace(" ", "").replace("\t", "").upper()
    today = datetime.date.today()
    dates = [(today - datetime.timedelta(days=i)).strftime("%Y%m%d") for i in range(6, -1, -1)]
    results = []
    conn = mysql.connector.connect(
         host="host.docker.internal",
         user="root",
         password="",
         database="veh_logs"
    )
    cursor = conn.cursor(dictionary=True)
    for d in dates:
        table_name = f"veh_log_{d}"
        # Remove both spaces and tabs from the license field
        sql = f"SELECT track_id, license, class_name, time FROM {table_name} WHERE REPLACE(REPLACE(license, ' ', ''), '\t', '') LIKE %s"
        like_pattern = "%" + query_normalized + "%"
        try:
            cursor.execute(sql, (like_pattern,))
            rows = cursor.fetchall()
            for row in rows:
                row['date'] = d
                results.append(row)
        except mysql.connector.Error:
            continue
    cursor.close()
    conn.close()
    return jsonify(results)

'''@app.route('/search_license', methods=['GET'])
def search_license():
    query_str = request.args.get('query', '')
    query_normalized = query_str.replace(" ", "").upper()
    today = datetime.date.today()
    dates = [(today - datetime.timedelta(days=i)).strftime("%Y%m%d") for i in range(6, -1, -1)]
    results = []
    conn = mysql.connector.connect(
         host="host.docker.internal",
         user="root",
         password="",
         database="veh_logs"
    )
    cursor = conn.cursor(dictionary=True)
    for d in dates:
        table_name = f"veh_log_{d}"
        sql = f"SELECT track_id, license, class_name, time FROM {table_name} WHERE REPLACE(license, ' ', '') LIKE %s"
        like_pattern = "%" + query_normalized + "%"
        try:
            cursor.execute(sql, (like_pattern,))
            rows = cursor.fetchall()
            for row in rows:
                row['date'] = d
                results.append(row)
        except mysql.connector.Error:
            continue
    cursor.close()
    conn.close()
    return jsonify(results)
'''
# Endpoint to search by time range (search across tables for days in range)
@app.route('/search_time', methods=['GET'])
def search_time():
    start = request.args.get('start')
    end = request.args.get('end')
    if not start or not end:
        return jsonify({"error": "Start and end times required"}), 400

    # Convert ISO 8601 (e.g., "2025-04-06T16:00") to "YYYY-MM-DD HH:MM:SS"
    if "T" in start:
        start = start.replace("T", " ")
    if "T" in end:
        end = end.replace("T", " ")
    # Append seconds if missing (i.e., only HH:MM provided)
    if len(start.split(":")) == 2:
        start += ":00"
    if len(end.split(":")) == 2:
        end += ":00"

    try:
        start_dt = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return jsonify({"error": "Incorrect date/time format. Use YYYY-MM-DD HH:MM:SS"}), 400

    # Calculate the date range (assuming the logs are split per day)
    delta = (end_dt.date() - start_dt.date()).days
    dates = [(start_dt.date() + datetime.timedelta(days=i)).strftime("%Y%m%d") for i in range(delta+1)]
    results = []
    conn = mysql.connector.connect(
         host="host.docker.internal",
         user="root",
         password="",
         database="veh_logs"
    )
    cursor = conn.cursor(dictionary=True)
    for d in dates:
        table_name = f"veh_log_{d}"
        sql = f"SELECT * FROM {table_name} WHERE time BETWEEN %s AND %s"
        try:
            cursor.execute(sql, (start, end))
            rows = cursor.fetchall()
            for row in rows:
                row['date'] = d
                results.append(row)
        except mysql.connector.Error:
            continue
    cursor.close()
    conn.close()
    return jsonify(results)

# Endpoint to get over speeding vehicles (avg speed > 40 km/h) for the last 7 days
@app.route('/dashboard_over_speed', methods=['GET'])
def dashboard_over_speed():
    today = datetime.date.today()
    # Last 7 days (from 6 days ago until today)
    dates = [(today - datetime.timedelta(days=i)).strftime("%Y%m%d") for i in range(6, -1, -1)]
    results = []
    try:
        connection = mysql.connector.connect(
            host="host.docker.internal",
            user="root",
            password="",      # Adjust if needed
            database="veh_logs"
        )
    except mysql.connector.Error as e:
        return jsonify({"error": str(e)}), 500

    cursor = connection.cursor(dictionary=True)
    for d in dates:
        table_name = f"veh_log_{d}"
        # Query records where the average speed (converted from string) is greater than 40
        sql = f"SELECT * FROM {table_name} WHERE CAST(REPLACE(avg_speed, ' km/h','') AS DECIMAL(10,2)) > 40"
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
            for row in rows:
                row['date'] = d
                results.append(row)
        except mysql.connector.Error:
            # Skip tables that do not exist or error out
            continue
    cursor.close()
    connection.close()
    return jsonify(results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



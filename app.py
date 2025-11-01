from flask import Flask, render_template, Response, request, redirect
import cv2
import os
import pickle
import numpy as np
import face_recognition
import pandas as pd
from datetime import datetime

app = Flask(__name__)

DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pickle"
ATTENDANCE_XLSX = "attendance.xlsx"

# Ensure directories and files exist
os.makedirs(DATASET_DIR, exist_ok=True)

def ensure_excel():
    if not os.path.exists(ATTENDANCE_XLSX):
        df = pd.DataFrame(columns=["name", "roll", "date", "time"])
        df.to_excel(ATTENDANCE_XLSX, index=False)
    else:
        try:
            df = pd.read_excel(ATTENDANCE_XLSX)
            required_cols = ["name", "roll", "date", "time"]
            if not all(col in df.columns for col in required_cols):
                df = pd.DataFrame(columns=required_cols)
                df.to_excel(ATTENDANCE_XLSX, index=False)
        except Exception:
            df = pd.DataFrame(columns=["name", "roll", "date", "time"])
            df.to_excel(ATTENDANCE_XLSX, index=False)

def encode_faces():
    image_paths = []
    for person_folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, person_folder)
        if os.path.isdir(folder_path):
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(folder_path, fname))

    known_encodings = []
    known_names = []

    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, boxes)
        name = os.path.basename(os.path.dirname(path))
        for enc in encs:
            known_encodings.append(enc)
            known_names.append(name)

    data = {"encodings": known_encodings, "names": known_names}
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

def mark_attendance(name):
    ensure_excel()

    if "_" in name:
        display_name, roll = name.rsplit("_", 1)
    else:
        display_name, roll = name, ""

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    df = pd.read_excel(ATTENDANCE_XLSX)
    if not all(col in df.columns for col in ["name", "roll", "date", "time"]):
        df = pd.DataFrame(columns=["name", "roll", "date", "time"])

    already = df[(df["name"] == display_name) & (df["date"] == date_str)]
    if len(already) == 0:
        new = pd.DataFrame([{
            "name": display_name,
            "roll": roll,
            "date": date_str,
            "time": time_str
        }])
        df = pd.concat([df, new], ignore_index=True)
        df.to_excel(ATTENDANCE_XLSX, index=False)
        return f"Attendance marked for {display_name}"
    return None

def get_today_attendance():
    ensure_excel()
    df = pd.read_excel(ATTENDANCE_XLSX)
    today = datetime.now().strftime("%Y-%m-%d")
    if "date" not in df.columns:
        return []
    df_today = df[df["date"] == today]
    return df_today.to_dict(orient="records")

# Video Streaming
camera = cv2.VideoCapture(0)
recognized_names = set()

def gen_frames():
    global recognized_names
    recognized_names = set()

    if not os.path.exists(ENCODINGS_FILE):
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No encodings found. Encode faces first.", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    else:
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]

        while True:
            success, frame = camera.read()
            if not success:
                break
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            boxes = face_recognition.face_locations(rgb_small, model="hog")
            encodings = face_recognition.face_encodings(rgb_small, boxes)

            for enc, box in zip(encodings, boxes):
                matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
                name = "Unknown"
                if True in matches:
                    best_idx = np.argmin(face_recognition.face_distance(known_encodings, enc))
                    if matches[best_idx]:
                        name = known_names[best_idx]

                top, right, bottom, left = [coord * 4 for coord in box]

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if name != "Unknown" and name not in recognized_names:
                    mark_attendance(name)
                    recognized_names.add(name)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask Routes
@app.route("/")
def index():
    attendance = get_today_attendance()
    return render_template("index.html", attendance=attendance, message=None)

@app.route("/capture", methods=["POST"])
def capture():
    name = request.form.get("name")
    roll = request.form.get("roll")
    if not name or not roll:
        return "Missing name or roll", 400
    folder = os.path.join(DATASET_DIR, f"{name}_{roll}")
    os.makedirs(folder, exist_ok=True)

    file = request.files.get('image')
    if file:
        count = len(os.listdir(folder))
        path = os.path.join(folder, f"{name}_{roll}_{count}.jpg")
        file.save(path)
        return redirect("/")
    return "No image uploaded", 400

@app.route("/encode_faces", methods=["POST"])
def encode_faces_route():
    encode_faces()
    return redirect("/")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
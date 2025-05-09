from flask import Flask, request
import numpy as np
import face_recognition
import cv2
import os
from datetime import datetime
import csv
import threading

app = Flask(__name__)

# 🧠 Load known faces
known_faces = {
    "atul": "0901EC221022",
    "abhiraj": "0901EC221005",
}

known_encodings = []
known_names = []

for name in known_faces:
    path = f"known_faces/{name}.jpg"  # Ensure this folder and images exist
    if os.path.exists(path):
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)
        else:
            print(f"⚠️ No face found in image: {path}")
    else:
        print(f"❌ Image not found: {path}")

detected_today = {}
latest_frame = None  # Global for OpenCV display

# 📁 Attendance folder
attendance_folder = "attendance_records"
os.makedirs(attendance_folder, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    global latest_frame
    try:
        raw_data = request.data
        frame = np.frombuffer(raw_data, dtype=np.uint8)
        frame = frame.reshape((240, 320, 2))  # For RGB565 format

        # Convert RGB565 to BGR image
        bgr_image = convert_rgb565_to_bgr(frame)

        # Resize for faster processing
        small_frame = cv2.resize(bgr_image, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        names_in_frame = []

        for encoding, loc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]

                # Save attendance only once per day
                if name not in detected_today:
                    now = datetime.now().strftime("%H:%M:%S")
                    detected_today[name] = now
                    save_attendance(name, now)
                    print(f"🧠 Face matched: {name} at {now}")

            names_in_frame.append((name, loc))

        # Draw rectangles on full-size frame
        scale = 4  # Because we resized to 1/4
        for name, (top, right, bottom, left) in names_in_frame:
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale
            cv2.rectangle(bgr_image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(bgr_image, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        latest_frame = bgr_image.copy()

        return "✅ Frame processed", 200

    except Exception as e:
        print(f"❌ Error: {e}")
        return "❌ Server error", 500

def convert_rgb565_to_bgr(rgb565_image):
    r = ((rgb565_image[:, :, 0] & 0xF8) >> 3).astype(np.uint8)
    g = (((rgb565_image[:, :, 0] & 0x07) << 3) | ((rgb565_image[:, :, 1] & 0xE0) >> 5)).astype(np.uint8)
    b = (rgb565_image[:, :, 1] & 0x1F).astype(np.uint8) << 3
    return cv2.merge((b, g, r))

def save_attendance(name, time):
    roll = known_faces.get(name, "Unknown")
    date_str = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(attendance_folder, f"attendance_{date_str}.csv")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Roll No", "Name", "Time"])
        writer.writerow([roll, name, time])
    print(f"📝 Attendance marked: {name} ({roll}) at {time}")

def display_video():
    global latest_frame
    while True:
        if latest_frame is not None:
            cv2.imshow("📷 ESP32-CAM Live Feed", latest_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    threading.Thread(target=display_video, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)

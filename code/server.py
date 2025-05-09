from flask import Flask, request
import numpy as np
import cv2
import face_recognition
import traceback

app = Flask(__name__)

# Constants based on your ESP32-CAM settings
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

@app.route('/')
def index():
    return "ESP32-CAM Face Recognition Server Running..."

@app.route('/upload', methods=['POST'])
def upload_frame():
    try:
        raw_data = request.data
        expected_size = FRAME_WIDTH * FRAME_HEIGHT * 2

        if not raw_data or len(raw_data) != expected_size:
            print(f"❌ Invalid frame size: {len(raw_data)} bytes, expected: {expected_size}")
            return "Invalid frame size", 400

        # Convert raw RGB565 to numpy array
        rgb565_array = np.frombuffer(raw_data, dtype=np.uint16).reshape((FRAME_HEIGHT, FRAME_WIDTH))

        # Convert to BGR image for OpenCV
        bgr_frame = cv2.cvtColor(rgb565_array, cv2.COLOR_BGR5652BGR)

        # Optional: Save frame (for debugging)
        # cv2.imwrite("latest_frame.jpg", bgr_frame)

        # Resize frame for faster face detection
        small_frame = cv2.resize(bgr_frame, (0, 0), fx=0.25, fy=0.25)

        # Run face detection
        face_locations = face_recognition.face_locations(small_frame)
        if face_locations:
            print(f"✅ Face(s) detected: {len(face_locations)}")
        else:
            print("❌ No face detected")

        return "Frame received", 200

    except Exception as e:
        print("🔥 Server Error:")
        traceback.print_exc()
        return "Server error", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

import cv2
import face_recognition
import csv
import os
import numpy as np
from datetime import datetime

# Predefined list of individuals with roll numbers
all_individuals = {
    "dhiraj": "0901AD221022",
    "ayush": "0901AD221018",
    "chetan": "0901AD221024",
    "Person D": "0901AD221004",
    "Person E": "0901AD221029",
    "Person F": "0901AD221045",
}

# Known images for face recognition
known_images = [
    (r'C:\Users\DELL\Desktop\minor project\known_faces\chetan.jpg',"chetan"),
    (r'C:\Users\DELL\Desktop\minor project\known_faces\atul.jpg',"atul")
]

# Prepare known encodings and names
known_encodings = []
known_names = []

for image_path, name in known_images:
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(name)
    else:
        print(f"Warning: No face found in image: {image_path}")

if not known_encodings:
    print("No known faces were loaded. Exiting.")
    exit(1)

# Capture video from webcam
video_capture = cv2.VideoCapture(0)  # Change to 1 if needed

# Track detected names with timestamps
detected_names = {}

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Detect faces and their encodings in the frame
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare detected face with known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        name = "Unknown"

        # Find the best match
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        # Record the detection
        if name != "Unknown" and name not in detected_names:
            detected_names[name] = datetime.now().strftime("%H:%M:%S")

        # Scale back face location for original frame size
        top, right, bottom, left = face_location
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display name
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show video feed
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()

# Prepare attendance record with roll numbers
attendance = []
for name, roll_number in all_individuals.items():
    status = 'P' if name in detected_names else 'A'
    time_stamp = detected_names.get(name, "N/A")
    attendance.append([roll_number, name, status, time_stamp])

# Create an "attendance_records" directory if it doesn't exist
folder_name = "attendance_records"
os.makedirs(folder_name, exist_ok=True)

# Generate a unique CSV filename based on the current date
current_datetime = datetime.now().strftime("%Y-%m-%d")
csv_filename = os.path.join(folder_name, f"attendance_{current_datetime}.csv")

# Save attendance to CSV
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Roll Number", "Name", "Attendance", "Time"])  # Header
    writer.writerows(attendance)

print(f"✅ Attendance saved to {csv_filename}")

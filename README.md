# 📷 Face Recognition Attendance System using ESP32-CAM

This project is an IoT-based **Face Recognition Attendance System** built using an **ESP32-CAM** module and a **Python Flask server**. It captures live images, detects and recognizes faces, and stores attendance records automatically in an Excel-compatible `.csv` file format.

---

## 🚀 Features

- 📸 Face detection and recognition using ESP32-CAM and OpenCV
- 📡 Real-time image capture and upload via HTTP
- 🧠 Face recognition with `face_recognition` Python library
- ✅ Attendance logging with timestamps
- 🧾 Attendance stored in `.csv` (can be opened in Excel)
- 🧵 Multi-threaded video display using OpenCV
- 🔒 Only marks attendance once per person per day

---

## 🧰 Tech Stack

### Hardware
- ESP32-CAM (AI Thinker module)
- FTDI Programmer

### Software
- Arduino IDE (ESP32 Board support)
- Python 3 with Flask
- OpenCV
- face_recognition
- NumPy
- CSV (for Excel-compatible attendance storage)



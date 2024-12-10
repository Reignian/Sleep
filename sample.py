import cv2
import tkinter as tk
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import time
import threading
from playsound import playsound

# Initialize face recognizer and load classifiers
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
leftEyeCascade = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
rightEyeCascade = cv2.CascadeClassifier("haarcascade_righteye_2splits.xml")

# Initialize Mediapipe face mesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Initialize global variables
cap = None
video_active = False
eyes_closed_start = None  # Track when eyes started being closed
alarm_active = False  # Track if alarm is currently playing
alarm_thread = None  # Thread for continuous alarm

# Function to calculate the Eye Aspect Ratio (EAR)
def calculate_ear(eye_points, face_landmarks):
    points = []
    for point in eye_points:
        landmark = face_landmarks.landmark[point]
        points.append([landmark.x, landmark.y])
    points = np.array(points)

    vertical_dist1 = np.linalg.norm(points[1] - points[5])
    vertical_dist2 = np.linalg.norm(points[2] - points[4])
    horizontal_dist = np.linalg.norm(points[0] - points[3])

    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

# Function to play alarm sound continuously in a separate thread
def play_continuous_alarm():
    global alarm_active
    while alarm_active:
        playsound('alarmsound.mp3', block=True)
        if not alarm_active:
            break

# Function to handle the alarm logic
def handle_alarm():
    global alarm_active, alarm_thread
    if not alarm_active or (alarm_thread and not alarm_thread.is_alive()):
        alarm_active = True
        alarm_thread = threading.Thread(target=play_continuous_alarm)
        alarm_thread.daemon = True
        alarm_thread.start()

# Function to capture and analyze each frame for eye status
def capture_img_sample(img):
    global eyes_closed_start, alarm_active
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = faceCascade.detectMultiScale(gray_img, 1.1, 10)

    # Convert the image to RGB for Mediapipe
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)

    LEFT_EYE_POINTS = [362, 385, 387, 263, 373, 380]  # Left eye landmarks
    RIGHT_EYE_POINTS = [33, 160, 158, 133, 153, 144]  # Right eye landmarks
    EAR_THRESHOLD = 0.3  # Adjust this threshold based on testing
    eyes_closed = False

    if len(faces) == 0:
        # No face detected, play alarm sound
        handle_alarm()
        cv2.putText(img, "No Face Detected", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Detect eyes within the face region
            roi_gray = gray_img[y:y + h, x:x + w]
            left_eyes = leftEyeCascade.detectMultiScale(roi_gray)
            right_eyes = rightEyeCascade.detectMultiScale(roi_gray)

            # Draw rectangles around detected left eyes
            for (ex, ey, ew, eh) in left_eyes:
                cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

            # Draw rectangles around detected right eyes
            for (ex, ey, ew, eh) in right_eyes:
                cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

                left_ear = calculate_ear(LEFT_EYE_POINTS, face_landmarks)
                right_ear = calculate_ear(RIGHT_EYE_POINTS, face_landmarks)
                avg_ear = (left_ear + right_ear) / 2.0

                h, w, _ = img.shape
                cv2.putText(img, f"Left EAR: {left_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, f"Right EAR: {right_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, f"Avg EAR: {avg_ear:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if avg_ear < EAR_THRESHOLD:
                    cv2.putText(img, "Eyes Closed", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if eyes_closed_start is None:
                        eyes_closed_start = time.time()
                    elif time.time() - eyes_closed_start > 2:
                        handle_alarm()
                else:
                    cv2.putText(img, "Eyes Open", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    eyes_closed_start = None
                    alarm_active = False

    return img

# Function to start video capture
def start_capture():
    global cap, video_active
    if not video_active:
        for i in range(2):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                video_active = True
                show_frame()
                return
        print("Error: Could not initialize any camera")

# Function to stop video capture
def stop_capture():
    global cap, video_active
    video_active = False
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

# Function to continuously show frames in main Tkinter window
def show_frame():
    global cap
    if video_active and cap.isOpened():
        success, img = cap.read()
        if success:
            img = capture_img_sample(img)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            camera_label.config(image=img_tk)
            camera_label.image = img_tk
            root.after(10, show_frame)

# Setup Tkinter window
root = tk.Tk()
root.title("Sleep Detection System")
root.geometry("800x600")
root.config(bg="#f0f0f0")

camera_frame = tk.Frame(root, bg="#f0f0f0")
camera_frame.pack(pady=10)

camera_label = tk.Label(camera_frame, text="Click 'Start Camera' to begin", bg="#f0f0f0", font=('Arial', 12))
camera_label.pack()

button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack(pady=10)

start_button = tk.Button(button_frame, text="Start Camera", command=start_capture, bg="#4CAF50", fg="white", padx=20)
start_button.pack(side=tk.LEFT, padx=5)

stop_button = tk.Button(button_frame, text="Stop Camera", command=stop_capture, bg="#f44336", fg="white", padx=20)
stop_button.pack(side=tk.LEFT, padx=5)

root.protocol("WM_DELETE_WINDOW", lambda: (stop_capture(), root.destroy()))
root.mainloop()

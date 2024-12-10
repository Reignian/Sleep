import cv2
import tkinter as tk
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np

# Initialize face recognizer and load classifiers
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Initialize global variables
cap = None
video_active = False

# Function to calculate the Eye Aspect Ratio (EAR)
def calculate_eye_aspect_ratio(eye_landmarks):
    if len(eye_landmarks) < 6:
        return 0  # Return 0 or some default value if there are not enough landmarks
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])  # Vertical distance
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])  # Vertical distance
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)  # Calculate EAR
    return ear

# Function to capture and analyze each frame for eye status
def capture_img_sample(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = faceCascade.detectMultiScale(gray_img, 1.1, 10)
    coordinate = []

    # Detect face and draw rectangle
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, "Face", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
        coordinate = [x, y, w, h]

        # Convert the image to RGB for Mediapipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)

        # Draw landmarks on the face
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    h, w, _ = img.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(img, (cx, cy), 1, (0, 255, 0), -1)  # Green dots for landmarks

                # Get left and right eye landmarks (33-133 for left, 362-463 for right)
                left_eye_landmarks = []
                right_eye_landmarks = []
                for i in range(33, 133):
                    left_eye_landmarks.append([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y])
                for i in range(362, 463):
                    right_eye_landmarks.append([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y])

                # Calculate EAR for left and right eyes
                left_eye_ear = calculate_eye_aspect_ratio(np.array(left_eye_landmarks))
                right_eye_ear = calculate_eye_aspect_ratio(np.array(right_eye_landmarks))

                # Calculate average EAR for both eyes
                average_ear = (left_eye_ear + right_eye_ear) / 2.0

                # Display EAR values on the frame
                cv2.putText(img, f"Left EAR: {left_eye_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, f"Right EAR: {right_eye_ear:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, f"Average EAR: {average_ear:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Eye detection within the face region
                roi_gray = gray_img[y:y + h, x:x + w]
                eyes = eyeCascade.detectMultiScale(roi_gray)
                eye_id = 1

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 0, 255), 2)

                    # If EAR is below a threshold, consider the eyes as closed
                    if average_ear > 1.35:  # Threshold for closed eyes, adjust based on testing
                        cv2.putText(img, "Closed", (x + ex, y + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(img, "Open", (x + ex, y + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    eye_id += 1

# Function to start video capture
def start_capture():
    global cap, video_active
    if not video_active:
        cap = cv2.VideoCapture(0)
        video_active = True
        show_frame()

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
    if video_active:
        success, img = cap.read()
        if success:
            capture_img_sample(img)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            camera_label.config(image=img_tk)
            camera_label.image = img_tk
        root.after(10, show_frame)

# Setup Tkinter window with improved design
root = tk.Tk()
root.title("Control Panel")
root.geometry("640x480")
root.config(bg="#f0f0f0")

camera_label = tk.Label(root)
camera_label.pack()

# Start capturing immediately
start_capture()

# Exit handling
root.protocol("WM_DELETE_WINDOW", lambda: (stop_capture(), root.destroy()))
root.mainloop()

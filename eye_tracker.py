import cv2
import mediapipe as mp
import pyautogui
import tkinter as tk

# Initialize MediaPipe for face and eye/head tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# For webcam input
cap = cv2.VideoCapture(0)

# Variables to track positions and settings
last_head_y_position = None
last_eye_y_position = None
gesture_threshold = 0.05  # Default sensitivity for gestures
delay = 2000  # Delay in milliseconds to prevent rapid scrolling
mode = "Eye Tracking"  # Default mode

# Calibration variables for eye tracking
calibrated_top_position = None
calibrated_bottom_position = None
is_calibrated = False
tracking_active = False

# GUI for starting/stopping the program
root = tk.Tk()
root.title("Gesture Control")
root.geometry("300x450")

def start_tracking():
    global tracking_active
    tracking_active = True
    start_button.config(text="Stop Tracking", command=stop_tracking)
    print(f"{mode} tracking started")

def stop_tracking():
    global tracking_active
    tracking_active = False
    start_button.config(text="Start Tracking", command=start_tracking)
    print(f"{mode} tracking stopped")

def update_sensitivity(val):
    global gesture_threshold
    gesture_threshold = float(val)
    print(f"Sensitivity updated to: {gesture_threshold}")

def update_delay(val):
    global delay
    delay = int(float(val) * 1000)  # Convert seconds to milliseconds
    print(f"Delay updated to: {delay / 1000} seconds")

def toggle_mode():
    global mode
    if mode == "Head Gestures":
        mode = "Eye Tracking"
    else:
        mode = "Head Gestures"
    mode_label.config(text=f"Mode: {mode}")
    print(f"Switched to {mode}")

def detect_head_position(landmarks):
    head_y_position = landmarks[1].y  # Y-coordinate of the nose tip
    return head_y_position

def detect_eye_position(landmarks):
    eye_y_position = (landmarks[33].y + landmarks[263].y) / 2  # Average Y-position of both eyes
    return eye_y_position

def calibrate_top():
    global calibrated_top_position
    print("Look at the top of the screen...")
    root.after(2000, lambda: read_eye_position("top"))  # Allow time for the user to look at the top

def calibrate_bottom():
    global calibrated_bottom_position
    print("Look at the bottom of the screen...")
    root.after(2000, lambda: read_eye_position("bottom"))  # Allow time for the user to look at the bottom

def read_eye_position(calibrate_type):
    success, image = cap.read()
    if success:
        results = face_mesh.process(cv2.flip(image, 1))
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                eye_position = detect_eye_position(face_landmarks.landmark)
                if calibrate_type == "top":
                    global calibrated_top_position
                    calibrated_top_position = eye_position
                    print(f"Top position calibrated at: {calibrated_top_position}")
                elif calibrate_type == "bottom":
                    global calibrated_bottom_position
                    calibrated_bottom_position = eye_position
                    print(f"Bottom position calibrated at: {calibrated_bottom_position}")

def detect_eye_gesture(current_position, image):
    global calibrated_top_position, calibrated_bottom_position, delay

    if calibrated_top_position is None or calibrated_bottom_position is None:
        print("Please calibrate the eye tracking first.")
        return

    # Calculate closeness to top and bottom points
    proximity_top = (calibrated_top_position - current_position) / (calibrated_top_position - calibrated_bottom_position)
    proximity_bottom = (current_position - calibrated_bottom_position) / (calibrated_top_position - calibrated_bottom_position)

    # Overlay the proximity as a progress bar
    if proximity_top >= 0 and proximity_bottom <= 1:
        overlay_progress_bar(image, proximity_top, proximity_bottom)

    # Improved sensitivity for detection
    if current_position < (calibrated_top_position - 0.03):  # Reduced buffer for more sensitivity
        print("Eyes at top - Scroll up")
        pyautogui.press("up")
        root.after(delay)

    elif current_position > (calibrated_bottom_position + 0.03):  # Reduced buffer for more sensitivity
        print("Eyes at bottom - Scroll down")
        pyautogui.press("down")
        root.after(delay)

def overlay_progress_bar(image, proximity_top, proximity_bottom):
    height, width, _ = image.shape
    # Displaying a vertical bar on the right side to represent proximity to top/bottom
    cv2.rectangle(image, (width - 20, int(proximity_top * height)), (width - 10, height), (0, 0, 255), -1)
    cv2.rectangle(image, (width - 20, 0), (width - 10, int(proximity_bottom * height)), (0, 255, 0), -1)

# GUI buttons and sensitivity/delay controls
start_button = tk.Button(root, text="Start Tracking", command=start_tracking)
start_button.pack(pady=10)

sensitivity_scale = tk.Scale(root, from_=0.01, to=0.1, resolution=0.01, orient="horizontal",
                             label="Sensitivity", command=update_sensitivity)
sensitivity_scale.set(gesture_threshold)
sensitivity_scale.pack(pady=10)

delay_scale = tk.Scale(root, from_=0.5, to=5, resolution=0.5, orient="horizontal",
                       label="Delay (seconds)", command=update_delay)
delay_scale.set(delay / 1000)
delay_scale.pack(pady=10)

# Mode toggle for switching between head gestures and eye tracking
mode_button = tk.Button(root, text="Toggle Mode", command=toggle_mode)
mode_button.pack(pady=10)

mode_label = tk.Label(root, text=f"Mode: {mode}")
mode_label.pack(pady=10)

# Calibration buttons for eye tracking
calibrate_top_button = tk.Button(root, text="Calibrate Top", command=calibrate_top)
calibrate_top_button.pack(pady=10)

calibrate_bottom_button = tk.Button(root, text="Calibrate Bottom", command=calibrate_bottom)
calibrate_bottom_button.pack(pady=10)

# Main loop for webcam processing
def process_webcam():
    global tracking_active, mode

    if tracking_active and cap.isOpened():
        success, image = cap.read()
        if not success:
            return

        image = cv2.flip(image, 1)  # Mirror the image for selfie-view
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if mode == "Eye Tracking":
                    eye_y_position = detect_eye_position(face_landmarks.landmark)
                    detect_eye_gesture(eye_y_position, image)

        # Overlay display on image for proximity bar
        cv2.imshow('Gesture Control', image)

        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            root.quit()
            return

    root.after(10, process_webcam)

# Start webcam processing loop
process_webcam()

# Start the GUI loop
root.mainloop()

import cv2
import mediapipe as mp
import time
import os
import threading

# Create snapshots directory if not exists
if not os.path.exists("snapshots"):
    os.mkdir("snapshots")

# Initialize face and hand detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)
last_snapshot_time = 0
snapshot_triggered = False

# Helper to detect finger states
def get_finger_states(landmarks):
    return [
        landmarks[4].y < landmarks[3].y,    # Thumb
        landmarks[8].y < landmarks[6].y,    # Index
        landmarks[12].y < landmarks[10].y,  # Middle
        landmarks[16].y < landmarks[14].y,  # Ring
        landmarks[20].y < landmarks[18].y   # Pinky
    ]

# Save snapshot
def take_snapshot(frame):
    global last_snapshot_time
    filename = f"snapshots/snapshot_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    print(f"📸 Saved: {filename}")
    last_snapshot_time = time.time()

# Run countdown and take snapshot
def countdown_and_snapshot(frame_copy):
    global snapshot_triggered
    for i in range(3, 0, -1):
        temp = frame_copy.copy()
        cv2.putText(temp, f"Snapshot in {i}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.imshow("Face + Gesture Actions", temp)
        cv2.waitKey(1000)
    take_snapshot(frame_copy)
    snapshot_triggered = False

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Detect hand landmarks
    results = hands.process(rgb)
    gesture = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers = get_finger_states(hand_landmarks.landmark)

            if fingers == [True, False, False, False, False]:
                gesture = "thumbs_up"
            elif all(fingers):
                gesture = "open_palm"
            elif fingers == [False, True, True, False, False]:
                gesture = "peace"  # Not used yet

    # Trigger snapshot on open palm
    if gesture == "open_palm" and not snapshot_triggered:
        # Only allow snapshot every 2 seconds
        if time.time() - last_snapshot_time > 2:
            snapshot_triggered = True
            frame_copy = frame.copy()
            threading.Thread(target=countdown_and_snapshot, args=(frame_copy,)).start()

    # Draw face rectangles and text
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if gesture == "thumbs_up":
            cv2.putText(frame, "😊 Happy!", (x + 10, y + h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Face + Gesture Actions", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

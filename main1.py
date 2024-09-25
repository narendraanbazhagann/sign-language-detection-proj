import cv2
import mediapipe as mp
import math
import time
import random

# Initialize MediaPipe Hand and Drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to randomly return either "Happy" or "Neutral"
def detect_emotion(face_image):
    return random.choice(["Happy", "Neutral"])

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    """Calculate the angle at point b given three points a, b, c."""
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]
    
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    
    cosine_angle = (ba[0]*bc[0] + ba[1]*bc[1]) / (
        math.sqrt(ba[0]**2 + ba[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2) + 1e-6)
    angle = math.acos(cosine_angle)
    return math.degrees(angle)

# Function to detect hand landmarks and return an image with hand keypoints drawn
def detect_hands(image, hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return result, image

# Function to recognize basic signs based on hand landmarks
def recognize_sign(hand_landmarks):
    if hand_landmarks:
        # Get positions of landmarks
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

        # Check if all fingers are folded (fist)
        fingers_folded = all([
            thumb_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y,
            index_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
            middle_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
            ring_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
            pinky_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
        ])

        if fingers_folded:
            return "Fist"

        # Example: Open palm (all fingers extended)
        fingers_extended = all([
            index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
            middle_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            ring_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
            pinky_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y,
            thumb_tip.x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x if thumb_tip.x < wrist.x else thumb_tip.x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x,
        ])

        if fingers_extended:
            return "Wave"

        # Example: Pointing gesture (only index finger extended)
        if (index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
            middle_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
            ring_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
            pinky_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y):
            return "Pointing"

        # Add Thumbs Up gesture recognition
        if (thumb_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y and
            index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
            middle_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
            ring_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
            pinky_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y):
            return "Thumbs Up"

        # Add number signs recognition (0 to 5)
        if (index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
            middle_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
            ring_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
            pinky_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y):
            return "2"

        if (index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
            middle_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
            ring_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
            pinky_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y):
            return "3"

        if (index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
            middle_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
            ring_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
            pinky_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y):
            return "4"

        if (index_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
            middle_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
            ring_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
            pinky_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y):
            return "1"

        if (index_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
            middle_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
            ring_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
            pinky_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y):
            return "0"

        return "Unknown Gesture"

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands for hand detection
with mp_hands.Hands(max_num_hands=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:
    p_time = 0  # Previous time for FPS calculation
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)

        # Detect hands in the frame
        result, annotated_image = detect_hands(image, hands)

        gesture_text = ""

        # Check if hand landmarks are detected and recognize the gesture
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                gesture = recognize_sign(hand_landmarks)
                gesture_text += gesture + " "

        # Detect faces in the frame for emotion detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Iterate through detected faces
        for (x, y, w, h) in faces:
            # Call the emotion detection function
            emotion = detect_emotion(image[y:y+h, x:x+w])

            # Draw rectangle around the face
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Display the emotion label
            cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Calculate FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        # Display gesture and FPS
        if gesture_text:
            cv2.putText(image, gesture_text.strip(), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, f'FPS: {int(fps)}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting image
        cv2.imshow('Sign Language Detection and Emotion Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

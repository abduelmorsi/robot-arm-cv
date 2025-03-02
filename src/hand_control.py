from serial.tools import list_ports
import serial.tools.list_ports
from serial import Serial
import cv2
import mediapipe as mp
import time
import logging

logging.basicConfig(level=logging.INFO)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def find_port():
    ports = list_ports.comports()
    for port in ports:
        if 'Arduino' in port.description or 'CH340' in port.description:
            return port.device
    return None

try:
    arduino_port = find_port()
    if (arduino_port is None):
        print("Arduino not found. Available ports:")
        for port in list_ports.comports():
            print(f"{port.device}: {port.description}")
        arduino_port = 'COM6'  #default port
        
    arduino = Serial(arduino_port, 9600, timeout=1)
    print(f"Connected to Arduino on {arduino_port}")
    time.sleep(2)
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    exit(1)

cap = cv2.VideoCapture(0)

def get_finger_angles(landmarks):
    angles = []
    fingers = [
        (4, 2),   # thumb
        (8, 5),   # index
        (12, 9),  # middle
        (16, 13), # ring
        (20, 17)  # pinky
    ]
    
    for tip, base in fingers:
        # Get vertical distance between tip and base
        tip_y = landmarks.landmark[tip].y
        base_y = landmarks.landmark[base].y
        # When tip is below base (closed), angle = 180
        # When tip is above base (open), angle = 0
        angle = 180 if tip_y > base_y else 0
        angles.append(angle)
    return angles

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        #flip image horizontally
        image = cv2.flip(image, 1)
        
        # convert image to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                angles = get_finger_angles(hand_landmarks)
                
                # Visualization
                for i, angle in enumerate(angles):
                    state = "CLOSED" if angle == 180 else "OPEN"
                    color = (0, 0, 255) if angle == 180 else (0, 255, 0)
                    cv2.putText(image, f"Finger {i}: {state}", (10, 30 + i * 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                command = ','.join(map(str, angles)) + '\n'
                arduino.write(command.encode())
                arduino.flush()
                time.sleep(0.01)

        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
except KeyboardInterrupt:
    print("Stopping program...")
finally:
    cap.release()
    cv2.destroyAllWindows()
    if 'arduino' in locals():
        arduino.close()
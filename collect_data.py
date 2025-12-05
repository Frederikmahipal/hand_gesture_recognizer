
# Data Collection Script for Hand Gesture Recognition

# This script collects labeled training data by capturing webcam frames
# and automatically organizing them into train/test splits.

# Controls:
# Press '1' - no_click (open hand - move mouse)
# Press '2' - left_click (Fist - left click)
# Press '3' - right_click (two fingers - right click)
# Press 's' - skip current frame
# Press 'q' - quit

# Images are automatically saved to
# data/train/gesture_name/  (70% by default)
# data/test/gesture_name/   (30% by default)


import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import random
import mediapipe as mp

# Configuration
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
IMG_SIZE = (128, 128)
DEFAULT_SPLIT = 0.7

# Gesture classes
GESTURE_CLASSES = {
    '1': 'no_click',
    '2': 'left_click',
    '3': 'right_click',
}

def setup_data_directories(train_split):
    # Create data directory structure
    DATA_DIR.mkdir(exist_ok=True)
    TRAIN_DIR.mkdir(exist_ok=True)
    TEST_DIR.mkdir(exist_ok=True)
    
    for gesture in GESTURE_CLASSES.values():
        (TRAIN_DIR / gesture).mkdir(parents=True, exist_ok=True)
        (TEST_DIR / gesture).mkdir(parents=True, exist_ok=True)
    
    print(f"Data directories set up (train/test split: {train_split:.0%}/{1-train_split:.0%})")

def save_image(image, gesture_name, train_split):
    # Save image to train or test directory based on split
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    # Decide train or test based on random split
    if random.random() < train_split:
        save_dir = TRAIN_DIR / gesture_name
    else:
        save_dir = TEST_DIR / gesture_name
    
    filename = f"{gesture_name}_{timestamp}.jpg"
    filepath = save_dir / filename
    
    # Resize and save
    resized = cv2.resize(image, IMG_SIZE)
    cv2.imwrite(str(filepath), resized)
    
    return filepath

def main():
    # Use default train/test split
    train_split = DEFAULT_SPLIT
    
    # Setup directories
    setup_data_directories(train_split)
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    frame_count = {gesture: 0 for gesture in GESTURE_CLASSES.values()}
    current_gesture = None
    
    while True:
        ret, frame = cap.read()
        
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks on frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display current gesture
        if current_gesture:
            cv2.putText(frame, f"Recording: {current_gesture}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display gesture instructions on frame
        cv2.putText(frame, "1=No Click  2=Left Click  3=Right Click", 
                   (10, frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame counts
        y_offset = 70
        for gesture, count in frame_count.items():
            cv2.putText(frame, f"{gesture}: {count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
        
        # Display instructions
        cv2.putText(frame, "Press 1-3 to label, 'q' to quit", 
                   (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        cv2.imshow('Hand Gesture Data Collection', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key >= ord('1') and key <= ord('3'):
            gesture_key = chr(key)
            if gesture_key in GESTURE_CLASSES:
                current_gesture = GESTURE_CLASSES[gesture_key]
                filepath = save_image(frame, current_gesture, train_split)
                frame_count[current_gesture] += 1
                print(f"Saved: {filepath.name} ({current_gesture})")
                current_gesture = None
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print("Collection Summary:")
    for gesture, count in frame_count.items():
        print(f"  {gesture}: {count} images")

if __name__ == '__main__':
    main()

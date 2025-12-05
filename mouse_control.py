

# This script performs real-time hand gesture recognition from webcam feed
# and maps predictions to mouse control actions using MediaPipe+KNN model.



# Gestures:
# no_click: Move mouse cursor (open hand, all 5 fingers extended)
# left_click: Left mouse click (fist, all fingers closed)
# right_click: Right mouse click (two fingers - index + middle extended)


import cv2
import numpy as np
import time
from pathlib import Path
import mediapipe as mp
import pickle
import pyautogui

# Configuration
GESTURE_CLASSES = ['no_click', 'left_click', 'right_click']
HISTORY_SIZE = 1
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
ACTION_COOLDOWN = 0.5
MOUSE_SMOOTHING = 0.1
MOUSE_SPEED_MULTIPLIER = 2.5

# Prediction smoothing
prediction_history = []

# Get screen size for mouse control
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

def load_knn_model(model_path, scaler_path):
    # Load KNN model and scaler
    with open(model_path, 'rb') as f:
        knn_model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return knn_model, scaler

def calculate_features_from_landmarks(landmarks, image_shape):
    # Calculate features from hand landmarks
    h, w = image_shape[:2]
    
    # Convert normalized coordinates to pixel coordinates (2D only - we don't need depth)
    points = []
    for landmark in landmarks.landmark:
        x = landmark.x * w
        y = landmark.y * h
        points.append([x, y])
    points = np.array(points)
    
    # Extract finger tips and wrist
    thumb_tip = points[4]
    index_tip = points[8]
    middle_tip = points[12]
    ring_tip = points[16]
    pinky_tip = points[20]
    wrist = points[0]
    
    # Determine if fingers are extended
    thumb_extended = (thumb_tip[1] < points[3][1])
    index_extended = (index_tip[1] < points[6][1])
    middle_extended = (middle_tip[1] < points[10][1])
    ring_extended = (ring_tip[1] < points[14][1])
    pinky_extended = (pinky_tip[1] < points[18][1])
    num_extended = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
    
    # Calculate key distances between fingertips (normalized by image width)
    thumb_index_dist = np.linalg.norm(thumb_tip - index_tip) / w
    thumb_middle_dist = np.linalg.norm(thumb_tip - middle_tip) / w
    index_middle_dist = np.linalg.norm(index_tip - middle_tip) / w
    thumb_pinky_dist = np.linalg.norm(thumb_tip - pinky_tip) / w
    
    # Calculate hand orientation angle from wrist to middle finger
    wrist_to_middle = middle_tip - wrist
    hand_angle = np.arctan2(wrist_to_middle[1], wrist_to_middle[0]) / np.pi
    
    # Calculate hand size (width)
    x_coords = points[:, 0]
    hand_width = (x_coords.max() - x_coords.min()) / w
    
    # Calculate hand center for mouse movement (2D only)
    hand_center = points.mean(axis=0)
    
    features = np.array([
        float(thumb_extended),  # Feature 0: Thumb extended
        float(index_extended),  # Feature 1: Index extended
        float(middle_extended), # Feature 2: Middle extended
        float(ring_extended),   # Feature 3: Ring extended
        float(pinky_extended),  # Feature 4: Pinky extended
        float(num_extended),    # Feature 5: Total extended fingers
        thumb_index_dist,       # Feature 6: Distance thumb-index
        thumb_middle_dist,      # Feature 7: Distance thumb-middle
        index_middle_dist,      # Feature 8: Distance index-middle
        thumb_pinky_dist,       # Feature 9: Distance thumb-pinky
        hand_angle,             # Feature 10: Hand orientation angle
        hand_width,             # Feature 11: Hand width
    ])
    
    return features, hand_center

def smooth_prediction(prediction):
    # Apply temporal smoothing to predictions
    if HISTORY_SIZE <= 1:
        return prediction
    
    prediction_history.append(prediction)
    if len(prediction_history) > HISTORY_SIZE:
        prediction_history.pop(0)
    
    avg_pred = np.mean(prediction_history, axis=0)
    return avg_pred

def get_predicted_gesture(prediction):
    # Get the predicted gesture class and confidence
    class_idx = np.argmax(prediction)
    confidence = float(prediction[class_idx])
    gesture = GESTURE_CLASSES[class_idx]
    return gesture, confidence

def predict_with_knn(knn_model, scaler, hands_detector, frame):
    # Run inference on a frame using KNN model with MediaPipe
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_image)
    
    if not results.multi_hand_landmarks:
        return None, 0.0, np.array([0.33, 0.33, 0.34]), None
    
    # Extract features
    landmarks = results.multi_hand_landmarks[0]
    features, hand_center = calculate_features_from_landmarks(landmarks, frame.shape)
    features_scaled = scaler.transform([features])
    
    # Predict
    prediction = knn_model.predict_proba(features_scaled)[0]
    smoothed_pred = smooth_prediction(prediction)
    gesture, confidence = get_predicted_gesture(smoothed_pred)
    
    # Convert hand center to screen coordinates
    h, w = frame.shape[:2]
    hand_x_norm = hand_center[0] / w
    hand_y_norm = hand_center[1] / h
    
    hand_screen_x = int(hand_x_norm * SCREEN_WIDTH)
    hand_screen_y = int(hand_y_norm * SCREEN_HEIGHT * 1.3)
    
    return gesture, confidence, smoothed_pred, (hand_screen_x, hand_screen_y)

def main():
    # Use default paths and settings
    model_path = Path('models/hand_click_model.pkl')
    scaler_path = Path('models/hand_click_scaler.pkl')
    confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
    mouse_speed = MOUSE_SPEED_MULTIPLIER
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3
    )
    mp_drawing = mp.solutions.drawing_utils
    
    # Load model
    knn_model, scaler = load_knn_model(model_path, scaler_path)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    last_click_time = {}
    fps_start_time = time.time()
    fps_frame_count = 0
    current_mouse_pos = None
    
    # Gesture stability tracking
    gesture_history = []
    gesture_frame_count = {}
    MIN_GESTURE_STABILITY = 3
    MAX_GESTURE_HISTORY = 5
    
    while True:
        ret, frame = cap.read()
        
        # Flip frame horizontally (mirror mode)
        frame = cv2.flip(frame, 1)
        
        # Predict gesture
        gesture, confidence, raw_pred, hand_pos = predict_with_knn(
            knn_model, scaler, hands, frame)
        
        # Draw hand landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
           
      
        # Track gesture stability
        if gesture is not None:
            if gesture not in gesture_frame_count:
                gesture_frame_count[gesture] = 0
            gesture_frame_count[gesture] += 1
            
            for g in GESTURE_CLASSES:
                if g != gesture:
                    gesture_frame_count[g] = 0
            
            if len(gesture_history) == 0 or gesture_history[-1] != gesture:
                gesture_history.append(gesture)
                if len(gesture_history) > MAX_GESTURE_HISTORY:
                    gesture_history.pop(0)
        else:
            gesture_frame_count = {}
        
        # Control mouse based on gesture
        current_time = time.time()
        # No click
        if gesture == 'no_click' and confidence > confidence_threshold and hand_pos:
            # Move mouse cursor
            target_x, target_y = hand_pos
            
            center_x = SCREEN_WIDTH // 2
            center_y = SCREEN_HEIGHT // 2
            
            offset_x = target_x - center_x
            offset_y = target_y - center_y
            
            target_x = center_x + int(offset_x * mouse_speed)
            target_y = center_y + int(offset_y * mouse_speed * 1.5)
            
            target_x = max(0, min(SCREEN_WIDTH - 1, target_x))
            target_y = max(0, min(SCREEN_HEIGHT - 1, target_y))
            
            if current_mouse_pos is None:
                current_mouse_pos = (target_x, target_y)
                pyautogui.moveTo(target_x, target_y, duration=0, _pause=False)
            else:
                current_x, current_y = current_mouse_pos
                new_x = int(current_x * MOUSE_SMOOTHING + target_x * (1 - MOUSE_SMOOTHING))
                new_y = int(current_y * MOUSE_SMOOTHING + target_y * (1 - MOUSE_SMOOTHING))
                current_mouse_pos = (new_x, new_y)
                pyautogui.moveTo(new_x, new_y, duration=0, _pause=False)
        # Left click
        elif gesture == 'left_click' and confidence > confidence_threshold:
            gesture_stable = gesture_frame_count.get('left_click', 0) >= MIN_GESTURE_STABILITY
            previous_was_no_click = len(gesture_history) > 1 and gesture_history[-2] == 'no_click'
            cooldown_passed = ('left_click' not in last_click_time or 
                              (current_time - last_click_time['left_click']) > ACTION_COOLDOWN)
            
            if gesture_stable and previous_was_no_click and cooldown_passed:
                pyautogui.click()
                last_click_time['left_click'] = current_time
                print(f"Left click (left_click gesture, confidence: {confidence:.2f})")

        # Right click
        elif gesture == 'right_click' and confidence > confidence_threshold:
            gesture_stable = gesture_frame_count.get('right_click', 0) >= MIN_GESTURE_STABILITY
            previous_was_no_click = len(gesture_history) > 1 and gesture_history[-2] == 'no_click'
            cooldown_passed = ('right_click' not in last_click_time or 
                              (current_time - last_click_time['right_click']) > ACTION_COOLDOWN)
            no_recent_left_click = ('left_click' not in last_click_time or 
                                   (current_time - last_click_time['left_click']) > ACTION_COOLDOWN * 0.5)
            
            if gesture_stable and previous_was_no_click and cooldown_passed and no_recent_left_click:
                pyautogui.rightClick()
                last_click_time['right_click'] = current_time

        # No hand detected
        elif gesture is None:
            current_mouse_pos = None
        
        # Display frame
        cv2.imshow('Hand Gesture Mouse Control', frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
if __name__ == '__main__':
    main()

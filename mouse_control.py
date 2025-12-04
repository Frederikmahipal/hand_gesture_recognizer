

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
    
    # Convert normalized coordinates to pixel coordinates
    points = []
    for landmark in landmarks.landmark:
        x = landmark.x * w
        y = landmark.y * h
        z = landmark.z * w
        points.append([x, y, z])
    points = np.array(points)
    
    # Extract features from hand landmarks
    wrist = points[0]
    wrist_x_norm = wrist[0] / w
    wrist_y_norm = wrist[1] / h
    wrist_z_norm = wrist[2] / w
    
    hand_center = points.mean(axis=0)
    center_x_norm = hand_center[0] / w
    center_y_norm = hand_center[1] / h
    center_z_norm = hand_center[2] / w
    
    thumb_tip = points[4]
    index_tip = points[8]
    middle_tip = points[12]
    ring_tip = points[16]
    pinky_tip = points[20]
    
    thumb_extended = (thumb_tip[1] < points[3][1])
    index_extended = (index_tip[1] < points[6][1])
    middle_extended = (middle_tip[1] < points[10][1])
    ring_extended = (ring_tip[1] < points[14][1])
    pinky_extended = (pinky_tip[1] < points[18][1])
    
    thumb_index_dist = np.linalg.norm(thumb_tip[:2] - index_tip[:2]) / w
    index_middle_dist = np.linalg.norm(index_tip[:2] - middle_tip[:2]) / w
    middle_ring_dist = np.linalg.norm(middle_tip[:2] - ring_tip[:2]) / w
    ring_pinky_dist = np.linalg.norm(ring_tip[:2] - pinky_tip[:2]) / w
    thumb_pinky_dist = np.linalg.norm(thumb_tip[:2] - pinky_tip[:2]) / w
    
    wrist_to_middle = middle_tip[:2] - wrist[:2]
    hand_angle = np.arctan2(wrist_to_middle[1], wrist_to_middle[0])
    hand_angle_norm = hand_angle / np.pi
    
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    hand_width = (x_coords.max() - x_coords.min()) / w
    hand_height = (y_coords.max() - y_coords.min()) / h
    
    index_base = points[5]
    index_vector = index_tip[:2] - index_base[:2]
    index_angle = np.arctan2(index_vector[1], index_vector[0]) / np.pi
    
    thumb_base = points[2]
    thumb_vector = thumb_tip[:2] - thumb_base[:2]
    thumb_angle = np.arctan2(thumb_vector[1], thumb_vector[0]) / np.pi
    
    hand_depth = points[:, 2].mean() / w
    
    features = np.array([
        wrist_x_norm,
        wrist_y_norm,
        wrist_z_norm,
        center_x_norm,
        center_y_norm,
        center_z_norm,
        thumb_tip[0] / w,
        thumb_tip[1] / h,
        index_tip[0] / w,
        index_tip[1] / h,
        middle_tip[0] / w,
        middle_tip[1] / h,
        float(thumb_extended),
        float(index_extended),
        float(middle_extended),
        float(ring_extended),
        float(pinky_extended),
        thumb_index_dist,
        index_middle_dist,
        middle_ring_dist,
        ring_pinky_dist,
        thumb_pinky_dist,
        hand_angle_norm,
        hand_width,
        hand_height,
        index_angle,
        thumb_angle,
        hand_depth,
    ])
    
    # Return hand center for mouse movement
    hand_center_2d = hand_center[:2]
    
    return features, hand_center_2d

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
        
        # Calculate FPS
        fps_frame_count += 1
        if fps_frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()
        else:
            fps = 0
        
        # Display prediction on frame
        if gesture is not None:
            text = f"{gesture}: {confidence:.2f}"
            color = (0, 255, 0) if confidence > confidence_threshold else (0, 0, 255)
        else:
            text = "No hand detected"
            color = (128, 128, 128)
        cv2.putText(frame, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Show all predictions
        if gesture is not None:
            y_offset = 70
            for i, gesture_name in enumerate(GESTURE_CLASSES):
                pred_val = raw_pred[i]
                color = (0, 255, 0) if i == np.argmax(raw_pred) else (200, 200, 200)
                text = f"{gesture_name}: {pred_val:.2f}"
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 20
        
        # Show FPS
        if fps > 0:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
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
    print("\nExiting...")

if __name__ == '__main__':
    main()

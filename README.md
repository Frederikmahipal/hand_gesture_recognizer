# Hand Gesture Recognition for Mouse Control

A machine learning system that recognizes hand gestures via webcam and controls your mouse cursor in real-time using MediaPipe and K-Nearest Neighbors (KNN) classifier.

## Quick Overview

This project allows you to control your computer mouse using hand gestures:
- **no_click** (Open Hand) -> Move mouse cursor
- **left_click** (thumb) -> Left mouse click
- **right_click** (Two Fingers) -> Right mouse click

---

## Prerequisites

- Python installed
- Webcam

---

## Step-by-Step Setup Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/Frederikmahipal/hand_gesture_recognizer
```

### Step 2: Set up and activate Virtual Environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv

source .venv/bin/activate  
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- MediaPipe (hand detection)
- OpenCV (image processing)
- scikit-learn (KNN classifier)
- PyAutoGUI (mouse control)
- NumPy, Matplotlib, Seaborn (data processing and visualization)
- Jupyter/IPython kernel (for running the training notebook)

### Step 4: Collect Training Data

You need to collect labeled images of hand gestures for training the model.

```bash
python3 collect_data.py
```

**How to use:**
1. Position your hand in front of the webcam
2. Press keys to label gestures:
   - **`1`** -> `no_click` (open hand - move mouse)
   - **`2`** -> `left_click` (gesture - left click)
   - **`3`** -> `right_click` (gesture - right click)
   - **`q`** -> Quit
3. Images are automatically split into `data/train/` (70%) and `data/test/` (30%)

### Step 5: Train the Model

1. Open `training/train_model.ipynb`
2. Run all cells sequentially (use your IDE's "Run All" or run each cell individually)
3. The notebook will:
   - Extract features from your collected images
   - Normalize the features
   - Find the best K value using cross-validation
   - Train the KNN classifier
   - Evaluate on test data
   - Save the model to `models/hand_click_model.pkl` and scaler to `models/hand_click_scaler.pkl`

**Expected output:**
- Training accuracy (should be >95%)
- Test accuracy (should be >95%)
- Confusion matrix visualization

### Step 6: Run Mouse Control

Once the model is trained, run the real-time mouse control:

```bash
python3 mouse_control.py
```

**Controls:**
- **Open hand** Move mouse cursor 
- **fist** Left click 
- **Two fingers** Right click
- **`q`** -> Quit

---

## Project Structure

```
mini_project/
├── collect_data.py              # Data collection script
├── mouse_control.py             # Real-time mouse control
├── training/
│   └── train_model.ipynb        # Training notebook 
├── data/                        # Generated when collecting data
│   ├── train/                   # Training images
│   │   ├── no_click/
│   │   ├── left_click/
│   │   └── right_click/
│   └── test/                    # Test images
│       ├── no_click/
│       ├── left_click/
│       └── right_click/
├── models/                      # Trained models (generated after training)
│   ├── hand_click_model.pkl
│   └── hand_click_scaler.pkl
└── requirements.txt             # Python dependencies
```

---

## Documentation

<details>
<summary><b>collect_data.py</b> - Data collection script</summary>

### Overview
Script for collecting labeled hand gesture images from webcam for training the machine learning model.


### Key Features
- **Real-time webcam capture** with hand landmark visualization
- **Automatic train/test split** - randomly assigns images to train (70%) or test (30%) folders
- **Image preprocessing** - automatically resizes all images to 128×128 pixels for consistency

### Gesture Classes
- **`1`** -> `no_click` (open hand - move mouse)
- **`2`** -> `left_click` (fist - left click)
- **`3`** -> `right_click` (two fingers - right click)

### Keyboard Controls
- **`1-3`**: Label current frame with gesture
- **`q`**: Quit and show collection summary

### Output
Images are saved to:
- `data/train/{gesture_name}/` (70% of images)
- `data/test/{gesture_name}/` (30% of images)

### Key Functions
- `setup_data_directories()`: Creates folder structure for train/test splits
- `save_image()`: Saves resized image to appropriate train/test directory
- `main()`: Main loop for webcam capture and labeling

</details>

<details>
<summary><b>mouse_control.py</b> - Real-Time Mouse Control</summary>

### Overview
Real-time hand gesture recognition system that controls your mouse cursor and clicks using trained KNN model.

### Usage
```bash
python3 mouse_control.py 
```


### Gesture Controls
- **`no_click`** (Open Hand): Move mouse cursor - follows hand position in real-time
- **`left_click`** (Fist): Left mouse click - requires gesture stability (3 frames) and previous gesture to be `no_click`
- **`right_click`** (Two Fingers): Right mouse click - same stability requirements as left click

### Features
- **Real-time inference** at 30+ FPS
- **Gesture stability detection** - prevents accidental clicks during transitions
- **Smooth mouse movement** - configurable speed multiplier and smoothing
- **Full screen coverage** - optimized coordinate mapping to reach entire screen
- **Cooldown system** - prevents rapid multiple clicks

### Key Functions
- `load_knn_model()`: Loads trained KNN model and scaler from pickle files
- `calculate_features_from_landmarks()`: Extracts 12 geometric features from hand landmarks
- `predict_with_knn()`: Runs inference on frame and returns gesture prediction with confidence
- `main()`: Main loop for webcam capture, prediction, and mouse control

### Configuration Constants
- `GESTURE_CLASSES`: List of gesture classes `['no_click', 'left_click', 'right_click']`
- `ACTION_COOLDOWN`: Seconds between click actions (default: 0.5)
- `MOUSE_SMOOTHING`: Smoothing factor for mouse movement (default: 0.1)
- `MOUSE_SPEED_MULTIPLIER`: Speed multiplier (default: 2.5)
- `MIN_GESTURE_STABILITY`: Frames required for gesture to be stable (default: 3)

### Gesture Stability Logic
Clicks only trigger when:
1. Gesture is detected for at least 3 consecutive frames
2. Previous gesture was `no_click` (prevents accidental clicks during transitions)
3. Cooldown period has passed since last click

</details>

<details>
<summary><b>training/train_model.ipynb</b> - Training Notebook</summary>

### Overview
Jupyter notebook containing the complete machine learning pipeline for training the hand gesture recognition model.

### Workflow
1. **Import Libraries**: MediaPipe, scikit-learn, OpenCV, NumPy, Matplotlib
2. **Configuration**: Set up paths, gesture classes, and MediaPipe Hands detector
3. **Feature Extraction**: Extract 12 geometric features from hand landmarks
4. **Data Processing**: Load and process training/test images
5. **Feature Normalization**: StandardScaler for consistent feature scaling
6. **Hyperparameter Tuning**: Cross-validation to find best K value
7. **Model Training**: Train KNN classifier with best K
8. **Model Evaluation**: Test accuracy, confusion matrix, classification report
9. **Model Saving**: Save model and scaler to `models/` directory

### Key Sections
- **Feature Engineering**: Extracts 12 relative features from hand landmarks. Only using the x and y coordinates of the hand landmarks is not enough, because the coordinates change depending on where the person in the frame sits. To make the model reliable, we calculate 12 relative features:
  - **Finger Extension**: Compare the y-position of the fingertip to the knuckle. If the tip is lower than the knuckle, the finger is folded (5 features: one for each finger)
  - **Total Extended Fingers**: Count of how many fingers are extended (1 feature)
  - **Euclidean Distances**: Calculate the physical distance between specific fingers (e.g., Thumb-to-Index, Thumb-to-Middle, Index-to-Middle, Thumb-to-Pinky) to differentiate between a "relaxed hand" and a "fist" (4 features)
  - **Hand Orientation Angle**: Angle from wrist to middle finger (1 feature)
  - **Hand Width**: Normalized width of the hand (1 feature)

- **Hyperparameter Tuning**: Tests K values from 1-15 using 5-fold cross-validation

- **Evaluation Metrics**: 
  - Test accuracy
  - Confusion matrix visualization
  - Classification report (precision, recall, F1-score)

### Output Files
- `models/hand_click_model.pkl`: Trained KNN classifier
- `models/hand_click_scaler.pkl`: StandardScaler for feature normalization
- `training/hand_click_confusion_matrix.png`: Confusion matrix visualization

### Expected Performance
- Training accuracy: >95%
- Test accuracy: >95%
- Inference speed: 30+ FPS

### Notes
- Notebook must be run from `training/` directory (uses `../` paths)
- Images without detected hands are automatically skipped
- Model uses KNN with K=1 (best cross-validation score)

</details>

---

## License

This project is for educational purposes (AAU Machine Learning Mini Project).

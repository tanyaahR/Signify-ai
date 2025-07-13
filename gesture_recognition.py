import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import mediapipe as mp
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Flatten, Dropout, Dense, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set environment variable to suppress oneDNN info messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Directory path to your dataset
data_dir = r"C:\Users\Mysore\Downloads\Sign-Language-Recognition-main\Sign-Language-Recognition-main\archive\ASL"

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,        # Rescale images
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2   # Reserve 20% for validation
)

batch_size = 16  # Reduced batch size for lower memory usage

# Training data generator
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(50, 50),
    batch_size=batch_size,
    class_mode='sparse',
    subset='training',
    shuffle=True
)

# Validation data generator
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(50, 50),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation',
    shuffle=True
)

# Get the number of classes
num_classes = train_generator.num_classes

# Simplified model
model = Sequential([
    Conv2D(16, (3, 3), activation="relu", input_shape=(50, 50, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(32, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Display the model's architecture
print(model.summary())

# Train the model using the generators
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Save the model
model.save("gestures.h5")

# Load the model for prediction
model = load_model("gestures.h5")

# Get class indices and labels
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}

# Example prediction with a sample image
# Replace with the path to your test image
test_img_path = r"C:\Users\Mysore\Downloads\Sign-Language-Recognition-main\Sign-Language-Recognition-main\archive\ASL\Pay\Pay_498.jpg"

# Load and preprocess the test image
img = cv2.imread(test_img_path)
img = cv2.resize(img, (50, 50))
img = img / 255.0
test_img = np.expand_dims(img, axis=0)

# Make a prediction
prediction = model.predict(test_img)
predicted_class = np.argmax(prediction, axis=-1)
print(f"Predicted class: {class_labels[predicted_class[0]]}")

# Initialize Mediapipe and OpenCV for real-time hand gesture detection
mphands = mp.solutions.hands
hand = mphands.Hands(max_num_hands=2, min_detection_confidence=0.75)
mpdraw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)  # Open the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            x_points = []
            y_points = []

            # Collect x and y coordinates of hand landmarks
            for id, lm in enumerate(landmarks.landmark):
                h, w, c = frame.shape
                x_points.append(int(lm.x * w))
                y_points.append(int(lm.y * h))

            # Define the bounding box
            x_min, x_max = min(x_points) - 20, max(x_points) + 20
            y_min, y_max = min(y_points) - 20, max(y_points) + 20

            # Ensure coordinates are within frame boundaries
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, w)
            y_max = min(y_max, h)

            # Draw rectangle around the hand
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Extract the region of interest (ROI)
            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size > 0:
                roi = cv2.resize(roi, (50, 50))
                roi = roi / 255.0
                roi = np.expand_dims(roi, axis=0)

                # Make a prediction
                prediction = model.predict(roi)
                predicted_class = np.argmax(prediction, axis=-1)
                confidence = np.max(prediction)

                # Display the prediction and confidence
                text = f"{class_labels[predicted_class[0]]} ({confidence*100:.2f}%)"
                cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Draw hand landmarks
            mpdraw.draw_landmarks(frame, landmarks, mphands.HAND_CONNECTIONS)

    # Show the frame
    cv2.imshow("Sign Language Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
cap.release()

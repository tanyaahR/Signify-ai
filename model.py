import streamlit as st
from keras.models import load_model
import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize MediaPipe hands module
mphands = mp.solutions.hands
hand = mphands.Hands(max_num_hands=2, min_detection_confidence=0.75)
mpdraw = mp.solutions.drawing_utils

# Streamlit UI setup
st.title("Sign Language Classification")
st.header("Gesture Model")

# Use the absolute path to the directory containing the class labels
classes_dir = "C:/Users/Mysore/Downloads/Sign-Language-Recognition-main/Sign-Language-Recognition-main/archive/asl"
if os.path.exists(classes_dir):
    all_classes = os.listdir(classes_dir)
else:
    st.error("The specified directory does not exist.")
    all_classes = []

@st.cache_resource
def model_upload():
    model_path = "C:/Users/Mysore/Downloads/Sign-Language-Recognition-main/Sign-Language-Recognition-main/gestures.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Model loaded successfully.")
        return model
    else:
        st.error("Model file not found.")
        return None

def predict(model, image):
    img = np.array(image)
    img = cv2.resize(img, (50, 50))  # Resize image to 50x50
    img = img / 255
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    pred = np.argmax(prediction, axis=-1)
    return pred, prediction[0][pred[0]]

run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

if not run:
    camera.release()

while run:
    x_points = []
    y_points = []
    ret, frame = camera.read()
    if not ret:
        st.error("Failed to capture image.")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for id, ld in enumerate(landmarks.landmark):
                h, w, channels = frame.shape
                x_points.append(int(ld.x * w))
                y_points.append(int(ld.y * h))

            a1 = (int(max(y_points) + 30), int(min(y_points) - 30))
            a2 = (int(max(x_points) + 30), int(min(x_points) - 30))
        cv2.rectangle(frame, (a2[1], a1[1]), (a2[0], a1[0]), (0, 255, 0), 3)

        if len(x_points) == 21 or len(x_points) == 42:
            target = frame[a1[1]:a1[0], a2[1]:a2[0]]
            if len(target) > 0:
                model = model_upload()
                if model:
                    p, num = predict(model, target)
                    if p[0] < len(all_classes):
                        cv2.putText(frame, f"{all_classes[p[0]]} {100 * num:.2f}", (80, 80), cv2.FONT_ITALIC, 2, (255, 100, 100), 2)
                    else:
                        st.error("Prediction index out of range.")
                else:
                    st.error("Failed to load the model.")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)

else:
    st.write('Stopped')
    if camera.isOpened():
        camera.release()

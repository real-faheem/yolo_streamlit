import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np

st.title("Vehicle Number Plate Detection with YOLOv11")

option = st.radio("Select input source:", ['Upload Image', 'Use Webcam'])

@st.cache_resource
def load_model():
    # Use the official ultralytics YOLOv5 repo or ultralytics YOLOv8 for newer models
    # yolov5 repo might cause the 'Detect' object has no attribute 'grid' error with latest PyTorch / model versions.
    # Instead, use ultralytics YOLOv8 package to load custom weights if your model is YOLOv11 or newer.
    from ultralytics import YOLO
    model = YOLO('yolov11cus.pt')
    return model

model = load_model()

def detect(image):
    # image: np.array RGB format
    results = model(image)
    result_img = results[0].plot()  # results[0] is the first inference result; .plot() returns annotated image
    return result_img

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an image file (jpg/png/jpeg)", type=['jpg','jpeg','png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        results_img = detect(img_array)
        st.image(results_img, caption="Detected Number Plates")

else:
    run = st.checkbox("Start Webcam")
    cap = cv2.VideoCapture(0)
    frame_window = st.image([])
    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to access webcam")
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_img = detect(img)
        frame_window.image(results_img)
    cap.release()

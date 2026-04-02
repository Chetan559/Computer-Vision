from pathlib import Path
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

os.environ['YOLO_AUTOINSTALL'] = '0'

from ultralytics import YOLO


st.set_page_config(page_title="Vehicle Number Plate Locator", page_icon="🚘", layout="wide")
st.title("Vehicle Number Plate Locator")
st.write("Upload an image, detect the vehicle number plate, and view the result.")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = {
    "PyTorch (.pt)": BASE_DIR / "number_plate_detection.pt",
    "ONNX (.onnx)": BASE_DIR / "number_plate_detection.onnx",
}


@st.cache_resource
def load_model(model_file: str):
    return YOLO(model_file, task='detect')


def draw_boxes_and_crops(image_rgb: np.ndarray, result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return image_rgb, []

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))

    annotated = image_rgb.copy()
    detections = []
    h, w = image_rgb.shape[:2]

    for idx, (box, score) in enumerate(zip(xyxy, conf), start=1):
        x1, y1, x2, y2 = box.astype(int)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"plate {idx}: {score:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        crop = image_rgb[y1:y2, x1:x2]
        if crop.size > 0:
            detections.append({"crop": crop, "confidence": float(score)})

    return annotated, detections


available_model_labels = [label for label, path in MODEL_PATHS.items() if path.exists()]
if not available_model_labels:
    st.error(
        "No model found in Detection folder. Add number_plate_detection.pt or number_plate_detection.onnx."
    )
    st.stop()

model_label = st.selectbox("Select model", available_model_labels)
model_path = MODEL_PATHS[model_label]

uploaded_file = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)

    with st.spinner(f"Running detection using {model_path.name}..."):
        try:
            model = load_model(str(model_path))
            results = model.predict(source=image_rgb, conf=0.25, verbose=False)
        except Exception as exc:
            st.error(f"Model inference failed: {exc}")
            st.stop()

    if not results:
        st.warning("No prediction result returned by model.")
        st.stop()

    annotated, plate_detections = draw_boxes_and_crops(image_rgb, results[0])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detected Plate Location")
        st.image(annotated, caption="Bounding box around number plate", width='stretch')

    with col2:
        st.subheader("Extracted Number Plates")
        if not plate_detections:
            st.warning("No number plate detected in this image.")
        else:
            st.caption(f"Detected {len(plate_detections)} plate(s)")
            for idx, detection in enumerate(plate_detections, start=1):
                st.image(
                    detection["crop"],
                    caption=f"Plate {idx} confidence: {detection['confidence']:.2f}",
                    width='stretch',
                )
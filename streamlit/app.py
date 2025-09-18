import cv2
import time
import streamlit as st
from ultralytics import YOLO

# ------------------
# Streamlit settings
# ------------------
st.set_page_config(page_title="Sign Gesture Detection", page_icon="🖐️", layout="wide")

# Inject custom CSS for modern styling
st.markdown("""
    <style>
    /* Main app background with gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #fafafa;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f4037, #99f2c8);
        color: white;
    }

    /* Titles */
    h1, h2, h3, h4 {
        color: #06d6a0;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #06d6a0;
        color: black;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #118ab2;
        color: white;
        transform: scale(1.05);
    }

    /* Slider */
    .stSlider [role=radiogroup] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }

    /* Detection status text */
    .status-text {
        font-size: 22px;
        font-weight: bold;
        color: #ffd166;
        padding: 8px;
    }

    /* Success and error messages */
    .stSuccess {
        background-color: #06d6a0 !important;
        color: black !important;
    }
    .stError {
        background-color: #ef476f !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🖐️ Real-Time Sign Gesture Detection")

# ------------------
# Sidebar settings (minimal)
# ------------------
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.35, 0.05)
start = st.sidebar.button("▶️ Start Webcam")
stop = st.sidebar.button("⏹️ Stop Webcam")

# ------------------
# Load YOLO model
# ------------------
try:
    model = YOLO("best.pt path")   # but your best.pt path here
except Exception as e:
    st.error(f"Could not load YOLO model: {e}")
    st.stop()

# ------------------
# Webcam loop
# ------------------
frame_window = st.image([])  # placeholder for video
status_placeholder = st.empty()

if start:
    cap = cv2.VideoCapture(0)  # default webcam
    if not cap.isOpened():
        st.error("❌ Cannot open webcam.")
    else:
        st.success("✅ Webcam started. Use Stop to exit.")

        prev_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to grab frame.")
                break

            # Inference
            results = model.predict(frame, conf=conf_thresh, verbose=False)
            annotated = results[0].plot()

            # Get top label
            if results[0].boxes:
                boxes = results[0].boxes
                top_idx = boxes.conf.argmax().item()
                cls_id = int(boxes[top_idx].cls[0])
                label = model.names[cls_id]
                conf = float(boxes[top_idx].conf[0])
                status_placeholder.markdown(
                    f"<div class='status-text'>Top Gesture: {label} ({conf:.2f})</div>",
                    unsafe_allow_html=True
                )
            else:
                status_placeholder.markdown(
                    "<div class='status-text'>Top Gesture: None</div>",
                    unsafe_allow_html=True
                )

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            # Show annotated frame
            frame_window.image(annotated, channels="BGR")

            # Check stop button
            if stop:
                st.warning("⏹️ Webcam stopped.")
                break

        cap.release()

# Sign Language Hand Gesture Detection (YOLOv8)

This project provides a YOLOv8-based object detection model for sign language hand gestures.
The dataset includes **5 classes**:
- Yes
- No
- Hello
- I Love You
- Thank You

## 📂 Dataset
- 125 labeled images
- YOLO format (images + labels + data.yaml)
- Split into train / val / test

Dataset available here: [Kaggle Dataset Link](https://www.kaggle.com/datasets/mhmd1424/sign-language-detection-dataset-5-classes)

## 🚀 Training
```bash
pip install -r requirements.txt

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=50, imgsz=640, batch=8)
```

## 🖥️ Running the Streamlit App

A simple **Streamlit demo** is included (`app.py`) so you can test the trained model locally.

### Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   source venv/bin/activate  # On Linux/Mac


2.Install dependencies:
```bash
pip install -r requirements.txt #might be optional
```
3.Run the Streamlit app:
```bash
streamlit run app.py
```
⚠️ Important
Update the path to your trained model (best.pt) inside app.py before running.

Example:
```python
model = YOLO("runs/detect/train/weights/best.pt")

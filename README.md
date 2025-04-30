# Where's Waldo Object Detection

This project uses deep learning and computer vision techniques to solve *Where's Waldo* puzzles by automatically locating Waldo in complex images.

## 🧠 Project Overview

Two models were implemented and compared:

- **YOLOv8 Nano**: A real-time object detection model requiring annotated images.
- **DINO (Self-Supervised Vision Transformer)**: A transformer-based model from Facebook AI used for feature extraction and similarity search with FAISS.

## 🧪 Models & Methodology

### 🔍 DINO + FAISS
- Images were resized to `224x224` for compatibility.
- Training used Waldo images from a [Kaggle dataset](https://www.kaggle.com/datasets/residentmario/wheres-waldo).
- FAISS was used for efficient similarity search.
- Achieved **88% success rate** during training but failed on more complex test images (1/20 successful detections).

### 🎯 YOLOv8 Nano
- Annotated dataset sourced from Roboflow (1,000+ images).
- Addressed dataset imbalance by augmenting underrepresented "not Waldo" images (flipping, rotation, brightness, blurring).
- Achieved:
  - **Precision**: 99.2%
  - **Recall**: 100%
  - **mAP@0.5**: 99.5%
  - **mAP@0.5:0.95**: 86.3%
- Detected Waldo in 6/19 puzzle images—significantly better than DINO.

## ⚙️ Technologies Used
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- FAISS
- DINO Vision Transformer
- Roboflow
- Kaggle

## 📈 Future Improvements
- Use larger YOLO models (e.g., v8-medium or v8-large) for better accuracy (requires more GPU resources).
- Expand dataset with more diverse annotated examples, especially "not Waldo" images.
- Improve DINO model with domain-specific fine-tuning.

## 📸 Dataset Links
- [Waldo Puzzle Images (Kaggle)](https://www.kaggle.com/datasets/residentmario/wheres-waldo)
- [Annotated Waldo Dataset (Roboflow)](https://roboflow.com)

---

*This project was developed to explore and compare object detection techniques using fun and challenging visual data.*

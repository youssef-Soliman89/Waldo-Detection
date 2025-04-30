
import os
import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the YOLOv8 model (choose 'n', 'm', or 'l' depending on your needs)
model = YOLO("yolov8n.pt")  # 'n' = nano (fast), 'm' = medium, 'l' = large

# Train the model on your dataset
model.train(
    data="/content/drive/Waldoman.v3i.yolov8/data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    device="cpu",
    flipud=0.0,  
    fliplr=1.0,  
    mosaic=1.0,  
    mixup=0.2
)
# Load the trained model
trained_model = YOLO("runs/detect/train/weights/best.pt")

model.save("/content/drive/MyDrive/best_model-L.pt")

# Load the trained YOLOv8 model
trained_model = YOLO("/content/drive/MyDrive/best_model-L.pt")

# Define input and output directories
input_folder = "/content/drive/MyDrive/original-images"
save_dir = "/content/drive/MyDrive/wheres_waldo_res"

# Ensure save directory exists
if os.path.exists(save_dir):
    # Clear the directory (delete old images)
    for file in os.listdir(save_dir):
        file_path = os.path.join(save_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)  # Remove the old file
else:
    os.makedirs(save_dir)  # Create directory if it doesn't exist


image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]


for image_file in image_files:
    input_image = os.path.join(input_folder, image_file)

    results = trained_model(input_image, conf=0.1)


    image = cv2.imread(input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding boxes
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores

        # Draw bounding boxes on the image
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 3)  # Yellow box
            cv2.putText(image, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)  # Confidence score in yellow

    # Save the processed image (overwrite previous results)
    output_path = os.path.join(save_dir, image_file)
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Convert back to BGR before saving

    # Show image
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    print(f"✅ Detection complete for {image_file}! Image saved at: {output_path}")

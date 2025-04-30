#any code with comments above it is from copilot, the rest is from the tutorial
import os
import torch
import torchvision.transforms as transforms
import numpy as np
import faiss
import cv2
from PIL import Image
from torchvision import models
import matplotlib.pyplot as plt  

# Load Pretrained DINO Model
model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
model.eval()


waldo_folder = "C:/Users/yhsol/Documents/wheres-waldo/Data/Only waldo"
if not os.path.exists(waldo_folder):#checks if the folder exists
    raise FileNotFoundError(f"❌ The folder '{waldo_folder}' does not exist. Run filtering first!")


waldo_files = sorted(os.listdir(waldo_folder))


def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  
    image_resized = image.resize((224, 224))  
    image_tensor = transforms.ToTensor()(image_resized).unsqueeze(0)

    with torch.no_grad():
        feature = model(image_tensor).squeeze().numpy()
    
    return feature, original_size 

filtered_features = []
waldo_sizes = {}  
for img in waldo_files:
    feature, original_size = extract_features(os.path.join(waldo_folder, img))
    filtered_features.append(feature)
    waldo_sizes[img] = original_size  

filtered_features = np.array(filtered_features)

# Create FAISS Index
index = faiss.IndexFlatL2(filtered_features.shape[1])
index.add(filtered_features)

# Load Puzzle Image (FULL SIZE)
puzzle_path = "C:/Users/yhsol/Documents/wheres-waldo/original-images/20.jpg"
puzzle_cv = cv2.imread(puzzle_path)  
puzzle_pil = Image.open(puzzle_path).convert("RGB")
puzzle_original_size = puzzle_pil.size  
# Resize Puzzle Image
puzzle_resized = puzzle_pil.resize((224, 224))
puzzle_tensor = transforms.ToTensor()(puzzle_resized).unsqueeze(0)

# Extract Features from Puzzle Image
with torch.no_grad():
    puzzle_feature = model(puzzle_tensor).squeeze().numpy()

# Search FAISS for Best Match
D, I = index.search(np.array([puzzle_feature]), 1)
best_match_idx = I[0][0]


if best_match_idx >= len(waldo_files):
    print(f"⚠️ Index {best_match_idx} is out of range! Adjusting...")#if the index is out of range, it will adjust it
    best_match_idx = len(waldo_files) - 1  


waldo_img_name = waldo_files[best_match_idx]
waldo_img_path = os.path.join(waldo_folder, waldo_img_name)
waldo_img = cv2.imread(waldo_img_path)


waldo_original_size = waldo_sizes[waldo_img_name]


waldo_resized = cv2.resize(waldo_img, waldo_original_size)  # Ensure original size
waldo_gray = cv2.cvtColor(waldo_resized, cv2.COLOR_BGR2GRAY)
puzzle_gray = cv2.cvtColor(puzzle_cv, cv2.COLOR_BGR2GRAY)


result = cv2.matchTemplate(puzzle_gray, waldo_gray, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc


h, w = waldo_gray.shape  # Get actual size
h = int(h * 0.8)  # Reduce height by 20%
w = int(w * 0.8)  # Reduce width by 20%
top_left = (top_left[0] + int(0.5 * w), max(0, top_left[1] - int(1.5 * h)))  # Move the box 1.5 times its height upwards and 0.5 times its width to the right

# Draw Red Bounding Box on Full Puzzle Image
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(puzzle_cv, top_left, bottom_right, (0, 0, 255), 5)

# Display the Image
plt.imshow(cv2.cvtColor(puzzle_cv, cv2.COLOR_BGR2RGB))  
plt.axis("off")  
plt.title("Waldo Found!")
plt.show()


output_path = "waldo_detected.jpg"
cv2.imwrite(output_path, puzzle_cv)
print(f"✅ Image saved as {output_path}")#prints the path of the image

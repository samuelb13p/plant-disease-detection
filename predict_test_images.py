import pickle
import yaml
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tabulate import tabulate  # pip install tabulate
import time

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load model and class indices
print("Loading model and configuration...")
model = load_model(config['paths']['model'])

with open(config['paths']['class_indices'], 'rb') as f:
    class_indices = pickle.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}
img_size = tuple(config['image']['size'])

# Folder to scan
folder_path = Path("test_data")

# Start processing
print("Processing images...\nPlease wait while we process the info.")
time.sleep(1)

results = []

for file_path in folder_path.iterdir():
    if file_path.is_file():
        img = image.load_img(file_path, target_size=img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(prediction)
        predicted_label = idx_to_class[predicted_idx]
        confidence = prediction[predicted_idx]

        results.append([file_path.name, predicted_label, f"{confidence:.2%}"])

# Show results
headers = ["Filename", "Predicted Class", "Confidence"]
print(tabulate(results, headers=headers, tablefmt="fancy_grid"))

print("\nDone!")

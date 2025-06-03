import cv2
import pickle
import numpy as np
import yaml
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_path = config['paths']['model']
class_indices_path = config['paths']['class_indices']
img_size = tuple(config['image']['size'])

# Load model and class indices
model = load_model(model_path)
with open(class_indices_path, 'rb') as f:
    class_indices = pickle.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 's' to scan the item. Press 'q' to quit.")

predicted_class = None
confidence = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    cv2.putText(display_frame, "Press 's' to scan the item. Press 'q' to quit.", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # If a prediction was made, show it
    if predicted_class is not None:
        cv2.putText(display_frame, f"Prediction: {predicted_class}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(display_frame, f"Confidence: {confidence:.2%}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    cv2.imshow('Leaf Scanner', display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Take snapshot and predict
        resized = cv2.resize(frame, img_size)
        img_array = img_to_array(resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(prediction)
        predicted_class = idx_to_class[predicted_idx]
        confidence = prediction[predicted_idx]

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

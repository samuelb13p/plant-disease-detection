# 🌿 Plant Disease Detection using CNN

This project uses a **Convolutional Neural Network (CNN)** to detect plant diseases from leaf images. It helps identify whether a leaf is healthy or affected by a specific disease based on visual patterns.

---

## 📦 Dataset

We used a dataset from **Mendeley Data**:

🔗 [Plant Leaf Disease Dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1)

This dataset contains **61,486 images** across **39 different classes**, including both healthy and diseased leaves. It includes various plants such as tomato, apple, corn, grape, strawberry, and more.

> Images were augmented using techniques like flipping, gamma correction, noise injection, PCA color augmentation, rotation, and scaling.

### 📁 How the Dataset is Used

The dataset is already available in the `dataset/` directory and ready for use. The training script automatically detects and uses the image categories inside this folder.

---

## 🏗️ Training the Model

The training script is located in the `scripts/` directory.

### 📌 Steps to Train

From your terminal, run the following command from the project root:

```bash
python scripts/train_model.py
```

> ⚠️ **Note:** Training can take several minutes or hours depending on your system.

### 🎯 Output Files

After training, the following files will be created in the `models/` directory:

- **`model.h5`** – The trained CNN model used for predictions.
- **`class_indices.pkl`** – A dictionary mapping class indices to class names. This is used during prediction to display readable class labels.
- **`history.pkl`** – Contains training history (accuracy and loss per epoch) and is used to plot performance graphs.

---

## 📈 Visualizing Training Metrics

You can visualize the training and validation accuracy/loss using the `print_chart.py` script.

### ▶️ Run the Chart Script

```bash
python scripts/print_chart.py
```

This script reads the `history.pkl` file and generates performance plots like the one below:

![Training and Validation Accuracy and Loss](assets/Figure.png)

- **Left Chart**: Accuracy over epochs (Train vs Validation)
- **Right Chart**: Loss over epochs (Train vs Validation)

This visualization helps understand how well the model is learning and if it's overfitting.

---

## 🧪 Predicting Leaf Diseases

After training, you can test the model using either a folder of images or directly with your camera.

### 📷 Option 1 – Predict Using the Camera

You can use your webcam to take a live picture of a leaf and predict its health status.

#### 📌 Steps to Predict with the Camera

1. Make sure your webcam is connected and working.
2. From the main directory, run:

```bash
python predict_with_camera.py
```

3. Place the leaf in the center of the camera frame with good lighting and a clear background.
4. When ready, press the **`s` key** to capture the image and see the prediction.
5. The model will display the predicted class (e.g., "Tomato___Late_blight") and the confidence percentage.

📸 Example screenshot:

![Camera Prediction Example](assets/Camera_prediction.png)

---

### 🖼️ Option 2 – Predict Using Test Images

1. Place test images in the `testData/` directory.
   - You can use the sample images or add your own.
2. From the main directory, run:

```bash
python predict_test_images.py
```

The model will analyze the images and print the filename, predicted class, and confidence in a table format:

```
╒══════════════════════════════╤════════════════════════════════════════════╤══════════════╕
│ Filename                     │ Predicted Class                            │ Confidence   │
╞══════════════════════════════╪════════════════════════════════════════════╪══════════════╡
│ AppleCedarRust1.JPG          │ Apple___Apple_scab                         │ 75.89%       │
│ AppleCedarRust2.JPG          │ Apple___Cedar_apple_rust                   │ 100.00%      │
╘══════════════════════════════╧════════════════════════════════════════════╧══════════════╛
```

---

## 🗂️ Project Structure

```
plant-disease-detection/
├── assets/                  # Folder to save images and assets
├── dataset/                 # Leaf image categories (already available)
├── models/
│   └── model.h5             # Trained CNN model
│   └── class_indices.pkl    # Mapping of class indices to class names
│   └── history.pkl          # Training history for plotting
├── scripts/
│   └── train_model.py       # Model training script
│   └── print_chart.py       # Script to visualize training metrics
├── testData/                # Images for prediction
├── .gitignore               # Files/folders to ignore in version control
├── config.yaml              # Configuration variables
├── predict_test_images.py   # Script to predict diseases from test images
├── predict_with_camera.py   # Script to predict diseases using the camera
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

---

## 👥 Author

- **Samuel Buendía** – [GitHub](https://github.com/samuelbuendia) · [LinkedIn](https://www.linkedin.com/in/samuelbuendia/) · [Portfolio](https://samuelbuendia.com)

---

## 📃 License

This project is for **academic and educational use**. The dataset is publicly available via [Mendeley Data](https://data.mendeley.com/datasets/tywbtsjrjv/1).

---

Happy coding and good luck detecting plant diseases! 🌱🧠🖼️
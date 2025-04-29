# 🌿 Plant Disease Detection using CNN

This project uses a Convolutional Neural Network (CNN) to detect plant diseases from images of leaves. It helps identify whether a leaf is healthy or affected by a disease based on visual patterns.

## 📦 Dataset

We used a dataset from Mendeley Data:

🔗 [Plant Leaf Disease Dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1)

This dataset contains **61,486 images** across **39 different classes**, including both healthy and diseased leaves. It includes various plants such as tomato, apple, corn, grape, strawberry, and more. Images were augmented using techniques like flipping, gamma correction, noise injection, PCA color augmentation, rotation, and scaling.

### How to use the dataset

Once you download the dataset, move the folders that represent different categories (e.g., `Tomato___healthy`, `Tomato___Early_blight`, etc.) into the `dataset/` directory of this project.

The model will automatically use these folders to train using the pictures inside.

---

## 🏗️ Training the Model

The training script is located in the `scripts/` directory.

### Steps to train:

1. Open a terminal and navigate to the `scripts/` directory:

   ```bash
   cd scripts
   ```

2. Run the training script:

   ```bash
   python train_model.py
   ```

> ⚠️ Training the model can take several hours depending on your system's capabilities.

### Output

After the training finishes, it will create the following files inside the `models/` directory:

- `model.h5`: The trained CNN model.
- `class_indices.pkl`: A dictionary mapping the class indices to class names (used during prediction).

---

## 🧪 Predicting Leaf Diseases

Once your model is trained, you can test it using the `run.py` script.

### Steps to test:

1. Save the images you want to classify in the `testData/` folder. You can add as many images as you'd like.

2. From the **main project directory**, run:

   ```bash
   python run.py
   ```

The script will analyze each image in the `testData/` folder and print the result in the console.

### Example Output

```
testData/test_leaf2.jpg
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 66ms/step
Predicted class: Strawberry___Leaf_scorch
```

---

## 🗂️ Project Structure

```
plant-disease-detection/
├── dataset/               # Downloaded leaf categories go here
├── models/                # Trained model and class index file
├── scripts/
│   └── train_model.py     # Training script
├── testData/              # Images to classify
├── run.py                 # Prediction script
└── README.md              # This file
```

---

### 👥 Authors

- **Samuel Buendía** – [GitHub](https://github.com/samuelbuendia) · [Portfolio](https://samuelbuendia.com)
- **Co-Author Name** – [GitHub](https://github.com/username) · [LinkedIn](https://linkedin.com/in/username)

---

## 📃 License

This project is for academic and educational use. The original dataset is made available by the authors on [Mendeley Data](https://data.mendeley.com/datasets/tywbtsjrjv/1).

---

Happy coding and good luck detecting plant diseases! 🌱🧠🖼️
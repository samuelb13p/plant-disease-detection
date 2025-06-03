import os
import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Paths
dataset_path = config['paths']['dataset']
model_path = config['paths']['model']
history_path = config['paths']['history']
class_indices_path = config['paths']['class_indices']

img_size = tuple(config['image']['size'])
batch_size = config['image']['batch_size']
val_split = config['image']['validation_split']

# Data Preprocessing
datagen = ImageDataGenerator(
    rescale=config['augmentation']['rescale'],
    validation_split=val_split,
    rotation_range=config['augmentation']['rotation_range'],
    zoom_range=config['augmentation']['zoom_range'],
    horizontal_flip=config['augmentation']['horizontal_flip']
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Save class indices
with open(class_indices_path, 'wb') as f:
    pickle.dump(train_data.class_indices, f)

# Load Pretrained Base Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base_model.trainable = False

# Custom classifier
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(config['model']['dense_units'], activation='relu')(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

# Full model
model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer=Adam(learning_rate=config['model']['learning_rate']),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
history = model.fit(train_data, validation_data=val_data, epochs=config['model']['epochs'])

# Save model
model.save(model_path)

# Save training history
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)

# Evaluate and print report
val_data.reset()
y_pred = model.predict(val_data)
y_pred = np.argmax(y_pred, axis=1)
y_true = val_data.classes
print(classification_report(y_true, y_pred, target_names=list(val_data.class_indices.keys())))

# Plot Accuracy & Loss
def plot_accuracy_and_loss(history_data):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history_data['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history_data['val_accuracy'], label='Val Accuracy', color='orange')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history_data['loss'], label='Train Loss', color='green')
    plt.plot(history_data['val_loss'], label='Val Loss', color='red')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Display the curves
plot_accuracy_and_loss(history.history)

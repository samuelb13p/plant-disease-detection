import pickle
import matplotlib.pyplot as plt

# Load the saved history
with open('./models/history.pkl', 'rb') as f:
    history_data = pickle.load(f)

# Plot function
def plot_accuracy_and_loss(history):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history['val_accuracy'], label='Val Accuracy', color='orange')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss', color='green')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Use the plot function
plot_accuracy_and_loss(history_data)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import numpy as np 

def build_model(input_shape: tuple, num_classes: int) -> Sequential:
    """
    Builds a convolutional neural network model for image classification.

    Args:
        input_shape (tuple): The shape of the input data.
        num_classes (int): The number of classes in the dataset.

    Returns:
        Sequential: The compiled Keras model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(),
        loss=SparseCategoricalCrossentropy(),
        metrics=[SparseCategoricalAccuracy()]
    )

    return model

def train_model(model: Sequential, train_data: np.ndarray, train_labels: np.ndarray, val_data: np.ndarray, val_labels: np.ndarray, epochs: int = 50, batch_size: int = 32) -> Sequential:
    """
    Trains the Keras model on the training data and evaluates it on the validation data.

    Args:
        model (Sequential): The Keras model to be trained.
        train_data (np.ndarray): The training data.
        train_labels (np.ndarray): The training labels.
        val_data (np.ndarray): The validation data.
        val_labels (np.ndarray): The validation labels.
        epochs (int, optional): The number of epochs to train the model. Defaults to 50.
        batch_size (int, optional): The batch size for training. Defaults to 32.

    Returns:
        Sequential: The trained Keras model.
    """
    model.fit(
        train_data, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_data, val_labels),
        verbose=1
    )

    return model

def evaluate_model(model: Sequential, test_data: np.ndarray, test_labels: np.ndarray) -> None:
    """
    Evaluates the trained model on the test data and prints the performance metrics.

    Args:
        model (Sequential): The trained Keras model.
        test_data (np.ndarray): The test data.
        test_labels (np.ndarray): The test labels.
    """
    loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

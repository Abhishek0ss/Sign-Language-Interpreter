
import os
import json
import pickle
import sqlite3
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Flatten,
    Conv2D, MaxPooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


# Training settings
IMAGE_SIZE = 50
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 0.001
RANDOM_SEED = 42

MAPPING_PATH = "class_mapping.json"


# Save training results and graphs
def save_training_history(history):

    history_df = pd.DataFrame(history.history)
    history_df.to_csv("training_history.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_accuracy.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss.png")
    plt.close()

    print("\nTraining history saved successfully.")
    print("Created: training_history.csv")
    print("Created: training_accuracy.png")
    print("Created: training_loss.png")


# Detect actual gesture IDs from folders
def get_gesture_ids():

    extensions = ["*.jpg", "*.jpeg", "*.png"]

    gesture_ids = []

    for folder in glob("gestures/*"):

        if not os.path.isdir(folder):
            continue

        folder_name = os.path.basename(folder)

        if not folder_name.isdigit():
            continue

        image_count = sum(
            len(glob(os.path.join(folder, ext)))
            for ext in extensions
        )

        if image_count > 0:
            gesture_ids.append(int(folder_name))

    gesture_ids = sorted(gesture_ids)

    if not gesture_ids:
        raise ValueError("No gesture classes with images found.")

    return gesture_ids


# Get gesture names from SQLite database
def get_gesture_names(gesture_ids):

    names = {}

    if not os.path.exists("gesture_db.db"):
        return {str(gid): f"Gesture {gid}" for gid in gesture_ids}

    connection = sqlite3.connect("gesture_db.db")
    cursor = connection.cursor()

    for gesture_id in gesture_ids:

        cursor.execute(
            "SELECT g_name FROM gesture WHERE g_id = ?",
            (gesture_id,)
        )

        result = cursor.fetchone()

        if result:
            names[str(gesture_id)] = result[0].strip()
        else:
            names[str(gesture_id)] = f"Gesture {gesture_id}"

    connection.close()

    return names


# Create and save database ID to CNN index mapping
def create_class_mapping(gesture_ids):

    db_id_to_model_index = {
        str(gesture_id): index
        for index, gesture_id in enumerate(gesture_ids)
    }

    class_names = get_gesture_names(gesture_ids)

    mapping = {
        "db_id_to_model_index": db_id_to_model_index,
        "model_index_to_db_id": gesture_ids,
        "class_names": [
            class_names[str(gesture_id)]
            for gesture_id in gesture_ids
        ]
    }

    with open(MAPPING_PATH, "w", encoding="utf-8") as file:
        json.dump(mapping, file, indent=4)

    print("\nClass mapping saved:")
    for index, gesture_id in enumerate(gesture_ids):
        print(
            f"Database ID {gesture_id} -> "
            f"CNN index {index} -> "
            f"{class_names[str(gesture_id)]}"
        )

    return db_id_to_model_index, mapping


# Create the CNN model
def cnn_model(num_classes, model_path):

    model = Sequential([
        Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),

        Conv2D(16, (2, 2), activation="relu"),
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same"
        ),

        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(3, 3),
            padding="same"
        ),

        Conv2D(64, (5, 5), activation="relu"),
        MaxPooling2D(
            pool_size=(5, 5),
            strides=(5, 5),
            padding="same"
        ),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint = ModelCheckpoint(
        model_path,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max"
    )

    return model, [checkpoint]


# Load dataset files
def load_pickle(filename):

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"{filename} not found. Run load_images.py first."
        )

    with open(filename, "rb") as file:
        return pickle.load(file)


# Normalize and reshape images
def prepare_images(images):

    images = np.asarray(images)

    if images.ndim != 3 or images.shape[1:] != (
        IMAGE_SIZE, IMAGE_SIZE
    ):
        raise ValueError(f"Invalid image shape: {images.shape}")

    images = images.astype("float32") / 255.0

    return np.expand_dims(images, axis=-1)


# Convert database IDs into contiguous CNN indices
def prepare_labels(labels, mapping, num_classes):

    labels = np.asarray(labels, dtype=np.int32)

    if len(labels) == 0:
        raise ValueError("No labels found.")

    mapped_labels = []

    for label in labels:

        label_key = str(int(label))

        if label_key not in mapping:
            raise ValueError(
                f"Unknown gesture ID {label}. "
                "Check your dataset and gesture folders."
            )

        mapped_labels.append(mapping[label_key])

    mapped_labels = np.asarray(mapped_labels, dtype=np.int32)

    return tf.keras.utils.to_categorical(
        mapped_labels,
        num_classes=num_classes
    )


# Main training function
def train():

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # Detect actual gesture IDs
    gesture_ids = get_gesture_ids()
    num_classes = len(gesture_ids)

    print("\nNumber of gesture classes:", num_classes)

    # Create class mapping
    db_id_to_model_index, class_mapping = create_class_mapping(
        gesture_ids
    )

    # Save model with a new filename
    model_path = f"cnn_model_{num_classes}classes.h5"

    print("\nModel will be saved as:", model_path)

    # Load datasets
    train_images = load_pickle("train_images")
    train_labels = load_pickle("train_labels")

    val_images = load_pickle("val_images")
    val_labels = load_pickle("val_labels")

    test_images = load_pickle("test_images")
    test_labels = load_pickle("test_labels")

    # Check image and label counts
    if len(train_images) != len(train_labels):
        raise ValueError("Training images and labels count mismatch.")

    if len(val_images) != len(val_labels):
        raise ValueError("Validation images and labels count mismatch.")

    if len(test_images) != len(test_labels):
        raise ValueError("Testing images and labels count mismatch.")

    # Prepare images
    train_images = prepare_images(train_images)
    val_images = prepare_images(val_images)
    test_images = prepare_images(test_images)

    # Prepare labels using database ID mapping
    train_labels = prepare_labels(
        train_labels, db_id_to_model_index, num_classes
    )

    val_labels = prepare_labels(
        val_labels, db_id_to_model_index, num_classes
    )

    test_labels = prepare_labels(
        test_labels, db_id_to_model_index, num_classes
    )

    # Display dataset information
    print("\nDataset Shapes:")
    print("Training:", train_images.shape, train_labels.shape)
    print("Validation:", val_images.shape, val_labels.shape)
    print("Testing:", test_images.shape, test_labels.shape)

    # Build CNN model
    model, callbacks = cnn_model(num_classes, model_path)

    model.summary()

    # Train the model
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    # Save training history
    save_training_history(history)

    # Load the best saved model
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "No trained model checkpoint was saved."
        )

    best_model = load_model(model_path)

    # Evaluate using test data
    test_loss, test_accuracy = best_model.evaluate(
        test_images,
        test_labels,
        verbose=1
    )

    print("\nFinal Test Results")
    print("------------------")
    print("Test Loss:", test_loss)
    print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
    print("Test Error: {:.2f}%".format((1 - test_accuracy) * 100))

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    train()

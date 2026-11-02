
import os
import pickle
import numpy as np
import tensorflow as tf

from glob import glob
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


IMAGE_SIZE = 50
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 0.001
RANDOM_SEED = 42


def get_num_of_classes():

    folders = sorted(
        [
            int(os.path.basename(folder))
            for folder in glob("gestures/*")
            if os.path.isdir(folder)
            and os.path.basename(folder).isdigit()
            and glob(os.path.join(folder, "*.jpg"))
        ]
    )

    if not folders:
        raise ValueError("No gesture classes with images found.")

    expected_classes = list(range(max(folders) + 1))

    if folders != expected_classes:
        raise ValueError(
            "Gesture class IDs must be continuous from 0. "
            f"Found: {folders}"
        )

    return len(folders)


def cnn_model(num_classes, model_path):

    model = Sequential()

    model.add(
        Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    )

    model.add(
        Conv2D(
            16,
            (2, 2),
            activation="relu"
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same"
        )
    )

    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu"
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(3, 3),
            padding="same"
        )
    )

    model.add(
        Conv2D(
            64,
            (5, 5),
            activation="relu"
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(5, 5),
            strides=(5, 5),
            padding="same"
        )
    )

    model.add(Flatten())

    model.add(
        Dense(
            128,
            activation="relu"
        )
    )

    model.add(Dropout(0.2))

    model.add(
        Dense(
            num_classes,
            activation="softmax"
        )
    )

    optimizer = Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
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


def load_pickle(filename):

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Dataset file not found: {filename}. "
            "Run load_images.py first."
        )

    with open(filename, "rb") as file:
        return pickle.load(file)


def prepare_images(images):

    images = np.asarray(images)

    expected_shape = (IMAGE_SIZE, IMAGE_SIZE)

    if images.ndim != 3 or images.shape[1:] != expected_shape:
        raise ValueError(
            f"Expected images with shape (N, {IMAGE_SIZE}, "
            f"{IMAGE_SIZE}), but got {images.shape}"
        )

    images = images.astype("float32") / 255.0

    images = np.expand_dims(images, axis=-1)

    return images


def prepare_labels(labels, num_classes):

    labels = np.asarray(labels, dtype=np.int32)

    if len(labels) == 0:
        raise ValueError("No labels found.")

    if np.min(labels) < 0 or np.max(labels) >= num_classes:
        raise ValueError(
            "Dataset labels do not match the detected classes."
        )

    return tf.keras.utils.to_categorical(
        labels,
        num_classes=num_classes
    )


def train():

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    num_classes = get_num_of_classes()

    print("\nNumber of gesture classes:", num_classes)

    # Use a separate model file for each class count.
    model_path = f"cnn_model_{num_classes}classes.h5"

    print("Model will be saved as:", model_path)

    # Load training data
    train_images = load_pickle("train_images")
    train_labels = load_pickle("train_labels")

    # Load validation data
    val_images = load_pickle("val_images")
    val_labels = load_pickle("val_labels")

    # Load test data
    test_images = load_pickle("test_images")
    test_labels = load_pickle("test_labels")

    # Prepare image data
    train_images = prepare_images(train_images)
    val_images = prepare_images(val_images)
    test_images = prepare_images(test_images)

    # Prepare labels
    train_labels = prepare_labels(
        train_labels,
        num_classes
    )

    val_labels = prepare_labels(
        val_labels,
        num_classes
    )

    test_labels = prepare_labels(
        test_labels,
        num_classes
    )

    print("\nDataset shapes:")

    print("Training images:", train_images.shape)
    print("Training labels:", train_labels.shape)

    print("Validation images:", val_images.shape)
    print("Validation labels:", val_labels.shape)

    print("Testing images:", test_images.shape)
    print("Testing labels:", test_labels.shape)

    # Build CNN
    model, callbacks_list = cnn_model(
        num_classes,
        model_path
    )

    model.summary()

    # Train
    model.fit(
        train_images,
        train_labels,
        validation_data=(
            val_images,
            val_labels
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list
    )

    # Load the best validation checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "No trained model checkpoint was saved."
        )

    best_model = load_model(model_path)

    # Evaluate on the separate test set
    test_loss, test_accuracy = best_model.evaluate(
        test_images,
        test_labels,
        verbose=1
    )

    print("\nFinal Test Results")
    print("------------------")

    print("Test Loss:", test_loss)

    print(
        "Test Accuracy: {:.2f}%".format(
            test_accuracy * 100
        )
    )

    print(
        "Test Error: {:.2f}%".format(
            (1 - test_accuracy) * 100
        )
    )

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    train()

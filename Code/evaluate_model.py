
import os
import pickle
import sqlite3
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


IMAGE_SIZE = 50


def load_pickle(filename):

    with open(filename, "rb") as f:
        return pickle.load(f)


def get_class_names(num_classes):

    conn = sqlite3.connect("gesture_db.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT g_id, g_name FROM gesture ORDER BY g_id"
    )

    rows = cursor.fetchall()
    conn.close()

    class_names = []

    for class_id in range(num_classes):

        matching = [
            name for g_id, name in rows
            if g_id == class_id
        ]

        if not matching:
            raise ValueError(
                f"Missing database name for class ID {class_id}"
            )

        class_names.append(matching[0].strip())

    return class_names


def main():

    test_images = np.asarray(
        load_pickle("test_images")
    )

    test_labels = np.asarray(
        load_pickle("test_labels"),
        dtype=np.int32
    )

    if len(test_images) != len(test_labels):
        raise ValueError("Test image and label counts differ.")

    test_images = test_images.astype("float32") / 255.0

    test_images = np.expand_dims(
        test_images,
        axis=-1
    )

    model_path = "cnn_model_4classes.h5"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = tf.keras.models.load_model(model_path)

    num_classes = model.output_shape[-1]

    class_names = get_class_names(num_classes)

    if np.min(test_labels) < 0 or np.max(test_labels) >= num_classes:
        raise ValueError("Test labels do not match model outputs.")

    predictions = model.predict(test_images, verbose=1)

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        test_labels,
        predicted_labels
    )

    print("\nMODEL EVALUATION RESULTS")
    print("------------------------")

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nCLASSIFICATION REPORT")

    report = classification_report(
        test_labels,
        predicted_labels,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0
    )

    print(report)

    with open("classification_report.txt", "w") as f:
        f.write(report)

    matrix = confusion_matrix(
        test_labels,
        predicted_labels,
        labels=list(range(num_classes))
    )

    print("\nCONFUSION MATRIX")
    print(matrix)

    np.savetxt(
        "confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d"
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    display.plot(
        cmap="Blues",
        ax=ax,
        xticks_rotation=45,
        colorbar=False
    )

    plt.title("Sign Language Gesture Recognition")
    plt.tight_layout()

    plt.savefig(
        "confusion_matrix.png",
        dpi=300
    )

    plt.show()

    print("\nEvaluation files saved:")
    print("1. classification_report.txt")
    print("2. confusion_matrix.csv")
    print("3. confusion_matrix.png")


if __name__ == "__main__":
    main()

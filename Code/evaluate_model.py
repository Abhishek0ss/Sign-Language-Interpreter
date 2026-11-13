
import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# -----------------------------
# 1. SETTINGS
# -----------------------------

MODEL_PATH = "cnn_model_7classes.h5"

TEST_IMAGES_PATH = "test_images"
TEST_LABELS_PATH = "test_labels"

IMAGE_SIZE = 50

# CNN output order
CLASS_NAMES = [
    "A",
    "B",
    "C",
    "D",
    "Best of Luck",
    "Love",
    "Victory"
]

# Original database IDs to CNN output indices
ID_MAPPING = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    36: 4,
    41: 5,
    44: 6
}


# -----------------------------
# 2. LOAD PICKLE FILES
# -----------------------------

def load_pickle(filename):

    with open(filename, "rb") as file:
        return pickle.load(file)


# -----------------------------
# 3. CONVERT TEST LABELS
# -----------------------------

def convert_labels(labels, num_classes):

    labels = np.asarray(labels)

    # Handle one-hot encoded labels
    if labels.ndim == 2:

        if labels.shape[1] != num_classes:
            raise ValueError(
                "One-hot labels do not match model outputs."
            )

        labels = np.argmax(labels, axis=1)

    # Handle integer labels
    elif labels.ndim == 1:

        unique_labels = set(labels.tolist())

        # Convert database IDs if present
        if unique_labels.issubset(set(ID_MAPPING.keys())):

            # Only remap if labels contain non-contiguous IDs
            if any(label not in range(num_classes)
                   for label in unique_labels):

                labels = np.array([
                    ID_MAPPING[int(label)]
                    for label in labels
                ])

    else:
        raise ValueError("Unexpected test label format.")

    # Final validation
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(
            "Test labels do not match model output classes."
        )

    return labels.astype(int)


# -----------------------------
# 4. MAIN EVALUATION
# -----------------------------

def main():

    # Load test data
    test_images = np.asarray(
        load_pickle(TEST_IMAGES_PATH)
    )

    test_labels = np.asarray(
        load_pickle(TEST_LABELS_PATH)
    )

    print("Test images:", test_images.shape)
    print("Original test labels:", test_labels.shape)

    # Load trained model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    num_classes = model.output_shape[-1]

    if num_classes != len(CLASS_NAMES):
        raise ValueError(
            "The model does not have seven output classes."
        )

    # Convert test labels
    test_labels = convert_labels(
        test_labels,
        num_classes
    )

    # Prepare images
    test_images = test_images.astype("float32") / 255.0

    if test_images.ndim == 3:
        test_images = np.expand_dims(
            test_images, axis=-1
        )

    if test_images.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 1):
        raise ValueError(
            "Test images must have shape (N, 50, 50, 1)."
        )

    if len(test_images) != len(test_labels):
        raise ValueError(
            "Test image and label counts differ."
        )

    # Predict
    predictions = model.predict(
        test_images,
        verbose=1
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    # Calculate accuracy
    accuracy = accuracy_score(
        test_labels,
        predicted_labels
    )

    print("\nMODEL EVALUATION RESULTS")
    print("------------------------")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Error: {(1 - accuracy) * 100:.2f}%")
    print(f"Test Images: {len(test_labels)}")

    # Classification report
    report = classification_report(
        test_labels,
        predicted_labels,
        labels=list(range(num_classes)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    print("\nCLASSIFICATION REPORT")
    print("---------------------")
    print(report)

    # Save report
    with open(
        "classification_report.txt",
        "w"
    ) as file:
        file.write(report)

    # Confusion matrix
    matrix = confusion_matrix(
        test_labels,
        predicted_labels,
        labels=list(range(num_classes))
    )

    print("\nCONFUSION MATRIX")
    print("----------------")
    print(matrix)

    # Save confusion matrix
    np.savetxt(
        "confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d"
    )

    # Plot confusion matrix
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES
    )

    fig, ax = plt.subplots(figsize=(10, 8))

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

    print("\nEvaluation completed!")
    print("Generated files:")
    print("1. classification_report.txt")
    print("2. confusion_matrix.csv")
    print("3. confusion_matrix.png")


# -----------------------------
# 5. RUN PROGRAM
# -----------------------------

if __name__ == "__main__":
    main()


import cv2
import numpy as np
import pickle
import os

from glob import glob
from collections import Counter
from sklearn.model_selection import train_test_split


IMAGE_SIZE = 50
RANDOM_STATE = 42


def load_images():

    images_labels = []

    image_paths = sorted(glob("gestures/*/*.jpg"))

    if not image_paths:
        raise ValueError("No gesture images found inside gestures/.")

    print("Loading gesture images...")

    invalid_images = []

    for image_path in image_paths:

        folder_name = os.path.basename(
            os.path.dirname(image_path)
        )

        if not folder_name.isdigit():
            print("Skipping invalid folder:", folder_name)
            continue

        label = int(folder_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            invalid_images.append(image_path)
            continue

        if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
            print("Incorrect image size:", image_path)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        images_labels.append(
            (img.astype(np.uint8), label)
        )

    if not images_labels:
        raise ValueError("No valid gesture images could be loaded.")

    if invalid_images:
        print("\nInvalid images skipped:", len(invalid_images))

    return images_labels


def save_pickle(filename, data):

    with open(filename, "wb") as file:
        pickle.dump(data, file)

    print("Saved:", filename)


def split_dataset(images, labels):

    # First split: 5/6 training and 1/6 temporary
    train_images, temp_images, train_labels, temp_labels = (
        train_test_split(
            images,
            labels,
            test_size=1/6,
            random_state=RANDOM_STATE,
            stratify=labels
        )
    )

    # Second split: divide temporary data equally
    test_images, val_images, test_labels, val_labels = (
        train_test_split(
            temp_images,
            temp_labels,
            test_size=0.5,
            random_state=RANDOM_STATE,
            stratify=temp_labels
        )
    )

    return (
        train_images,
        train_labels,
        test_images,
        test_labels,
        val_images,
        val_labels
    )


def main():

    images_labels = load_images()

    print("\nTotal images loaded:", len(images_labels))

    images = [item[0] for item in images_labels]
    labels = [item[1] for item in images_labels]

    # Display class distribution
    class_counts = Counter(labels)

    print("\nDataset class distribution:")

    for class_id in sorted(class_counts):
        print(
            "Class ID:",
            class_id,
            "| Images:",
            class_counts[class_id]
        )

    # Check minimum samples for stratified splitting
    for class_id, count in class_counts.items():

        if count < 6:
            raise ValueError(
                f"Class {class_id} has only {count} images. "
                "At least 6 images per class are needed "
                "for this stratified split."
            )

    (
        train_images,
        train_labels,
        test_images,
        test_labels,
        val_images,
        val_labels
    ) = split_dataset(images, labels)

    print("\nDataset split completed.")

    print("Training images:", len(train_images))
    print("Training labels:", len(train_labels))

    print("Testing images:", len(test_images))
    print("Testing labels:", len(test_labels))

    print("Validation images:", len(val_images))
    print("Validation labels:", len(val_labels))

    # Save using the original filenames
    save_pickle("train_images", tuple(train_images))
    save_pickle("train_labels", tuple(train_labels))

    save_pickle("test_images", tuple(test_images))
    save_pickle("test_labels", tuple(test_labels))

    save_pickle("val_images", tuple(val_images))
    save_pickle("val_labels", tuple(val_labels))

    print("\nAll dataset files saved successfully.")


if __name__ == "__main__":
    main()

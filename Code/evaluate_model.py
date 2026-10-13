import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report


# --------------------------------------------------
# IMAGE SIZE
# --------------------------------------------------

def get_image_size():

    import cv2

    img = cv2.imread(
        "gestures/0/100.jpg",
        0
    )

    return img.shape


image_x, image_y = get_image_size()


# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------

with open("test_images", "rb") as f:

    test_images = np.array(
        pickle.load(f)
    )


with open("test_labels", "rb") as f:

    test_labels = np.array(
        pickle.load(f),
        dtype=np.int32
    )


# --------------------------------------------------
# PREPARE TEST IMAGES
# --------------------------------------------------

test_images = np.reshape(
    test_images,
    (
        test_images.shape[0],
        image_x,
        image_y,
        1
    )
)


# Normalize exactly like training
test_images = (
    test_images.astype("float32") / 255.0
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

model = tf.keras.models.load_model(
    "cnn_model_keras2.h5"
)


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

predictions = model.predict(
    test_images,
    verbose=1
)


predicted_labels = np.argmax(
    predictions,
    axis=1
)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

print()
print("==============================")
print("CONFUSION MATRIX")
print("==============================")


cm = confusion_matrix(
    test_labels,
    predicted_labels
)


print(cm)


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

print()
print("==============================")
print("CLASSIFICATION REPORT")
print("==============================")


print(
    classification_report(
        test_labels,
        predicted_labels,
        target_names=[
            "A",
            "B",
            "C",
            "D"
        ]
    )
)


# --------------------------------------------------
# TEST ACCURACY
# --------------------------------------------------

accuracy = np.mean(
    predicted_labels == test_labels
)


print()
print("==============================")
print("FINAL TEST ACCURACY")
print("==============================")


print(
    "Test Accuracy: {:.2f}%".format(
        accuracy * 100
    )
)

import cv2
import pickle
import subprocess
import numpy as np
import sqlite3
import json
import os

from collections import deque
from tensorflow.keras.models import load_model



MODEL_PATH = "cnn_model_7classes.h5"
MAPPING_PATH = "class_mapping.json"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(MAPPING_PATH):
    raise FileNotFoundError(f"Mapping file not found: {MAPPING_PATH}")

model = load_model(MODEL_PATH)

with open(MAPPING_PATH, "r", encoding="utf-8") as file:
    class_mapping = json.load(file)

MODEL_INDEX_TO_DB_ID = class_mapping["model_index_to_db_id"]

if model.output_shape[-1] != len(MODEL_INDEX_TO_DB_ID):
    raise ValueError(
        "Model output count does not match class mapping."
    )



CONFIDENCE_THRESHOLD = 0.60
HISTORY_SIZE = 6
MIN_STABLE_COUNT = 3
COOLDOWN_FRAMES = 15

IMAGE_SIZE = 50

# Region of Interest
ROI_X, ROI_Y, ROI_W, ROI_H = 300, 100, 300, 300



def speak_text(text):

    text = text.strip()

    if not text:
        print("No text to speak.")
        return

    safe_text = text.replace("'", "''")

    ps_script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Speak('" + safe_text + "'); "
        "$s.Dispose()"
    )

    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print("Speech completed:", text)
        else:
            print("Speech error:", result.stderr)

    except subprocess.TimeoutExpired:
        print("Speech timed out.")

    except Exception as e:
        print("Speech error:", repr(e))



def get_hand_hist():

    if not os.path.exists("hist"):
        raise FileNotFoundError("Hand histogram file 'hist' not found.")

    with open("hist", "rb") as file:
        return pickle.load(file)


hist = get_hand_hist()



def get_pred_text_from_db(db_gesture_id):

    conn = sqlite3.connect("gesture_db.db")

    try:
        cursor = conn.execute(
            "SELECT g_name FROM gesture WHERE g_id = ?",
            (int(db_gesture_id),)
        )

        result = cursor.fetchone()

    finally:
        conn.close()

    if result:
        return result[0].strip()

    return "Unknown"



def keras_process_image(img):

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img, dtype=np.float32)

    img = img / 255.0

    img = np.reshape(
        img,
        (1, IMAGE_SIZE, IMAGE_SIZE, 1)
    )

    return img


def keras_predict(image):

    processed_image = keras_process_image(image)

    prediction = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return confidence, predicted_class


def get_database_id(model_index):

    if model_index < 0 or model_index >= len(MODEL_INDEX_TO_DB_ID):
        raise ValueError(f"Invalid CNN output index: {model_index}")

    return int(MODEL_INDEX_TO_DB_ID[model_index])


def get_img_contour_thresh(img):

    img = cv2.flip(img, 1)
    img = cv2.resize(img, (640, 480))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    dst = cv2.calcBackProject(
        [hsv],
        [0, 1],
        hist,
        [0, 180, 0, 256],
        1
    )

    disc = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (10, 10)
    )

    cv2.filter2D(dst, -1, disc, dst)

    blur = cv2.GaussianBlur(dst, (11, 11), 0)
    blur = cv2.medianBlur(blur, 15)

    thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    thresh = cv2.merge((thresh, thresh, thresh))
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    # Extract ROI
    x, y, w, h = ROI_X, ROI_Y, ROI_W, ROI_H
    thresh = thresh[y:y+h, x:x+w]

    contours, _ = cv2.findContours(
        thresh.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE
    )

    return img, contours, thresh


def make_square(img, width, height):

    if width > height:

        padding = (width - height) // 2

        img = cv2.copyMakeBorder(
            img,
            padding,
            width - height - padding,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=0
        )

    elif height > width:

        padding = (height - width) // 2

        img = cv2.copyMakeBorder(
            img,
            0,
            0,
            padding,
            height - width - padding,
            cv2.BORDER_CONSTANT,
            value=0
        )

    return img


def get_stable_prediction(prediction_history):

    if not prediction_history:
        return None

    counts = {}

    for prediction in prediction_history:
        counts[prediction] = counts.get(prediction, 0) + 1

    most_common_class = max(counts, key=counts.get)

    if counts[most_common_class] >= MIN_STABLE_COUNT:
        return most_common_class

    return None



def add_gesture_to_text(recognized_text, gesture_name):

    # Letters are joined together: A + B = AB
    if gesture_name in ["A", "B", "C", "D"]:
        recognized_text += gesture_name


    else:

        if recognized_text and not recognized_text.endswith(" "):
            recognized_text += " "

        recognized_text += gesture_name + " "

    return recognized_text



def main():

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    x, y, w, h = ROI_X, ROI_Y, ROI_W, ROI_H

    prediction_history = deque(maxlen=HISTORY_SIZE)

    stable_text = "No hand"
    stable_confidence = 0.0

    recognized_text = ""

    last_added_class = None
    cooldown = 0

    print("\n======================================")
    print("       SIGN LANGUAGE RECOGNITION")
    print("======================================")

    print("\nRecognizing:")

    for index, db_id in enumerate(MODEL_INDEX_TO_DB_ID):
        gesture_name = get_pred_text_from_db(db_id)
        print(f"CNN index {index}: {gesture_name}")

    print("\nControls:")
    print("S = Space")
    print("B = Backspace")
    print("C = Clear text")
    print("V = Speak")
    print("Q = Quit")

    print("\nHold a gesture inside the green box.\n")

    try:

        while True:

            ret, frame = cam.read()

            if not ret:
                print("Could not read webcam frame.")
                break

            frame = cv2.resize(frame, (640, 480))

            valid_prediction = False
            current_prediction = None

            img, contours, thresh = get_img_contour_thresh(frame)

            # Detect largest hand contour
            if len(contours) > 0:

                contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(contour)

                if contour_area > 10000:

                    x1, y1, w1, h1 = cv2.boundingRect(contour)

                    hand_img = thresh[
                        y1:y1+h1,
                        x1:x1+w1
                    ]

                    hand_img = make_square(
                        hand_img,
                        w1,
                        h1
                    )

                    # Predict gesture
                    confidence, pred_class = keras_predict(hand_img)

                    if confidence >= CONFIDENCE_THRESHOLD:

                        prediction_history.append(pred_class)
                        valid_prediction = True

                        current_prediction = get_stable_prediction(
                            prediction_history
                        )

                        if current_prediction is not None:

                            db_id = get_database_id(current_prediction)

                            stable_text = get_pred_text_from_db(db_id)
                            stable_confidence = confidence

                    else:
                        prediction_history.clear()

            # Reset if no valid hand prediction
            if not valid_prediction:

                prediction_history.clear()

                stable_text = "No hand"
                stable_confidence = 0.0

                last_added_class = None

            # Cooldown handling
            if cooldown > 0:
                cooldown -= 1

            # Add recognized gesture to text
            if current_prediction is not None:

                if (
                    current_prediction != last_added_class
                    and cooldown == 0
                ):

                    db_id = get_database_id(current_prediction)

                    gesture_name = get_pred_text_from_db(db_id)

                    if gesture_name != "Unknown":

                        recognized_text = add_gesture_to_text(
                            recognized_text,
                            gesture_name
                        )

                        print("Recognized:", gesture_name)

                        last_added_class = current_prediction
                        cooldown = COOLDOWN_FRAMES

            # Draw ROI
            cv2.rectangle(
                img,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            # Display gesture
            cv2.putText(
                img,
                "Gesture: " + stable_text,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            # Display confidence
            cv2.putText(
                img,
                "Confidence: {:.1f}%".format(
                    stable_confidence * 100
                ),
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Display recognized text
            display_text = recognized_text[-25:]

            cv2.putText(
                img,
                "Text: " + display_text,
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            # Display controls
            cv2.putText(
                img,
                "S:Space  B:Backspace  C:Clear  Q:Quit  V:Speak",
                (20, 460),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            # Show webcam and threshold
            cv2.imshow("Sign Language Recognition", img)
            cv2.imshow("Threshold", thresh)

            key = cv2.waitKey(1) & 0xFF

            # Quit
            if key == ord("q"):
                break

            # Add space
            elif key == ord("s"):

                if recognized_text and not recognized_text.endswith(" "):
                    recognized_text += " "

            # Backspace
            elif key == ord("b"):

                recognized_text = recognized_text[:-1]

            # Clear text
            elif key == ord("c"):

                recognized_text = ""
                last_added_class = None
                cooldown = 0
                prediction_history.clear()

            # Text-to-speech
            elif key == ord("v"):

                print("Speaking text:", repr(recognized_text))
                speak_text(recognized_text)

    finally:

        cam.release()
        cv2.destroyAllWindows()

    print("\nFinal recognized text:")
    print(recognized_text)

    print("\nRecognition stopped.")


if __name__ == "__main__":
    main()

import cv2
import pickle
#import pyttsx3
import subprocess
import textwrap
import numpy as np
import sqlite3
from collections import deque
from tensorflow.keras.models import load_model




model = load_model("cnn_model_keras2.h5")
# Initialize text-to-speech
def speak_text(text):
    text = text.strip()

    if not text:
        print("No text to speak.")
        return

    # Escape text safely for PowerShell
    safe_text = text.replace("'", "''")

    ps_script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Speak('" + safe_text + "'); "
        "$s.Dispose()"
    )

    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                ps_script
            ],
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

    with open("hist", "rb") as f:
        hist = pickle.load(f)

    return hist


hist = get_hand_hist()




def get_image_size():

    img = cv2.imread(
        "gestures/0/100.jpg",
        0
    )

    return img.shape


image_x, image_y = get_image_size()




def keras_process_image(img):

    img = cv2.resize(
        img,
        (image_x, image_y)
    )

    img = np.array(
        img,
        dtype=np.float32
    )

    # Same normalization used during training
    img = img / 255.0

    img = np.reshape(
        img,
        (1, image_x, image_y, 1)
    )

    return img




def keras_predict(image):

    processed = keras_process_image(image)

    prediction = model.predict(
        processed,
        verbose=0
    )[0]

    pred_class = int(
        np.argmax(prediction)
    )

    probability = float(
        np.max(prediction)
    )

    return probability, pred_class




def get_pred_text_from_db(pred_class):

    conn = sqlite3.connect(
        "gesture_db.db"
    )

    cursor = conn.execute(
        "SELECT g_name FROM gesture WHERE g_id=?",
        (pred_class,)
    )

    row = cursor.fetchone()

    conn.close()

    if row:
        return row[0].strip()

    return "Unknown"




def get_img_contour_thresh(img):

    img = cv2.flip(
        img,
        1
    )

    imgHSV = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    dst = cv2.calcBackProject(
        [imgHSV],
        [0, 1],
        hist,
        [0, 180, 0, 256],
        1
    )

    disc = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (10, 10)
    )

    cv2.filter2D(
        dst,
        -1,
        disc,
        dst
    )

    blur = cv2.GaussianBlur(
        dst,
        (11, 11),
        0
    )

    blur = cv2.medianBlur(
        blur,
        15
    )

    thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    thresh = cv2.merge(
        (thresh, thresh, thresh)
    )

    thresh = cv2.cvtColor(
        thresh,
        cv2.COLOR_BGR2GRAY
    )

    # Hand detection region
    x, y, w, h = 300, 100, 300, 300

    thresh = thresh[
        y:y+h,
        x:x+w
    ]

    contours, _ = cv2.findContours(
        thresh.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE
    )

    return img, contours, thresh



cam = cv2.VideoCapture(0)

if not cam.isOpened():

    print(
        "Error: Could not open webcam."
    )

    exit()




x, y, w, h = 300, 100, 300, 300

CONFIDENCE_THRESHOLD = 0.60

HISTORY_SIZE = 6

MIN_STABLE_COUNT = 3




prediction_history = deque(
    maxlen=HISTORY_SIZE
)


stable_class = None

stable_text = "No hand"

stable_confidence = 0.0




recognized_text = ""

last_added_class = None

# Prevent immediate duplicate letters
cooldown = 0



print()
print("======================================")
print(" SIGN LANGUAGE RECOGNITION")
print("======================================")
print()
print("Recognizing: A, B, C, D")
print()
print("Controls:")
print("  S = Space")
print("  B = Backspace")
print("  C = Clear text")
print("  Q = Quit")
print()
print("Hold a gesture inside the green box.")
print()




while True:

    ret, frame = cam.read()

    if not ret:

        print(
            "Could not read webcam frame."
        )

        break


    frame = cv2.resize(
        frame,
        (640, 480)
    )


    img, contours, thresh = \
        get_img_contour_thresh(frame)


    current_prediction = None

    current_confidence = 0.0


   

    if len(contours) > 0:

        contour = max(
            contours,
            key=cv2.contourArea
        )

        contour_area = cv2.contourArea(
            contour
        )


        if contour_area > 10000:

            x1, y1, w1, h1 = \
                cv2.boundingRect(contour)


            save_img = thresh[
                y1:y1+h1,
                x1:x1+w1
            ]


        

            if w1 > h1:

                border = int(
                    (w1 - h1) / 2
                )

                save_img = cv2.copyMakeBorder(
                    save_img,
                    border,
                    border,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=0
                )


            elif h1 > w1:

                border = int(
                    (h1 - w1) / 2
                )

                save_img = cv2.copyMakeBorder(
                    save_img,
                    0,
                    0,
                    border,
                    border,
                    cv2.BORDER_CONSTANT,
                    value=0
                )


         

            confidence, pred_class = \
                keras_predict(save_img)

            


            current_confidence = confidence


            if confidence >= CONFIDENCE_THRESHOLD:

                prediction_history.append(
                    pred_class
                )


                counts = {}

                for prediction in prediction_history:

                    counts[prediction] = \
                        counts.get(prediction, 0) + 1


                most_common_class = max(
                    counts,
                    key=counts.get
                )


                most_common_count = \
                    counts[most_common_class]


             

                if most_common_count >= MIN_STABLE_COUNT:

                    current_prediction = \
                        most_common_class

                    stable_class = \
                        most_common_class

                    stable_text = \
                        get_pred_text_from_db(
                            stable_class
                        )

                    stable_confidence = \
                        confidence


            else:

                prediction_history.clear()


    else:

        prediction_history.clear()


    # ==================================================
    # ADD LETTER TO TEXT
    # ==================================================

    if current_prediction is not None:

        if cooldown > 0:

            cooldown -= 1


        # Only add when:
        # 1. A gesture is stable
        # 2. It is different from the previous letter
        # 3. Cooldown has finished

        if (
            current_prediction != last_added_class
            and cooldown == 0
        ):

            letter = get_pred_text_from_db(
                current_prediction
            )

            if letter in ["A", "B", "C", "D"]:

                recognized_text += letter

                last_added_class = \
                    current_prediction

                cooldown = 15


    else:

        # Once the hand disappears,
        # allow the same gesture again.

        if len(prediction_history) == 0:

            last_added_class = None


 

    cv2.rectangle(
        img,
        (x, y),
        (x+w, y+h),
        (0, 255, 0),
        2
    )




    cv2.putText(
        img,
        "Gesture: " + stable_text,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )



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


 

    display_text = recognized_text

    if len(display_text) > 25:

        display_text = \
            display_text[-25:]


    cv2.putText(
        img,
        "Text: " + display_text,
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    cv2.putText(
        img,
        "S:Space  B:Backspace  C:Clear  Q:Quit  V:Speak",
        (20, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Sign Language Recognition",
        img
    )

    cv2.imshow(
        "Threshold",
        thresh
    )




    key = cv2.waitKey(1) & 0xFF


    # Quit
    if key == ord("q"):

        break


    # Space
    elif key == ord("s"):

        if (
            len(recognized_text) > 0
            and not recognized_text.endswith(" ")
        ):

            recognized_text += " "


    # Backspace
    elif key == ord("b"):

        recognized_text = \
            recognized_text[:-1]


    # Clear
    elif key == ord("c"):

        recognized_text = ""

        last_added_class = None

        prediction_history.clear()

    #speak    

    elif key == ord("v"):
        print("V pressed. Text:", repr(recognized_text))
        speak_text(recognized_text)   



cam.release()

cv2.destroyAllWindows()

print()
print("Final recognized text:")
print(recognized_text)
print()
print("Recognition stopped.")
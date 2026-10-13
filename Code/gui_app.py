import os
import cv2
import pickle
import sqlite3
import subprocess
import base64
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from collections import deque, Counter
from tensorflow.keras.models import load_model
from tkinter import messagebox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

MODEL_PATH = "cnn_model_keras2.h5"
HIST_PATH = "hist"
DATABASE_PATH = "gesture_db.db"
SAMPLE_PATH = os.path.join("gestures", "0", "100.jpg")

CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
ROI_X, ROI_Y, ROI_W, ROI_H = 300, 100, 300, 300
MASK_WIDTH, MASK_HEIGHT = 300, 300
CONFIDENCE_THRESHOLD = 0.80
HISTORY_SIZE = 6
MIN_STABLE_COUNT = 5
MIN_HAND_AREA = 6000
MIN_HAND_WIDTH = 60
MIN_HAND_HEIGHT = 60
MIN_MASK_WHITE_RATIO = 0.05
MAX_MASK_WHITE_RATIO = 0.85

for file_path in (MODEL_PATH, HIST_PATH, DATABASE_PATH, SAMPLE_PATH):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")

print("Loading model...")
model = load_model(MODEL_PATH, compile=False)

with open(HIST_PATH, "rb") as f:
    hist = pickle.load(f)

sample = cv2.imread(SAMPLE_PATH, cv2.IMREAD_GRAYSCALE)
if sample is None:
    raise FileNotFoundError(f"Could not read {SAMPLE_PATH}")
IMAGE_HEIGHT, IMAGE_WIDTH = sample.shape[:2]

with sqlite3.connect(DATABASE_PATH) as conn:
    rows = conn.execute("SELECT g_id, g_name FROM gesture").fetchall()
gesture_names = {int(gid): str(name).strip() for gid, name in rows}


def get_gesture_name(class_id):
    return gesture_names.get(int(class_id), "Unknown")


def predict_gesture(hand_image):
    hand_image = cv2.resize(hand_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    hand_image = hand_image.astype(np.float32) / 255.0
    hand_image = hand_image.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    probs = model.predict(hand_image, verbose=0)[0]
    class_id = int(np.argmax(probs))
    return float(probs[class_id]), class_id


def speak_text(text_to_speak):
    text_to_speak = text_to_speak.strip()
    if not text_to_speak:
        status_var.set("Recognize some text before speaking")
        return
    try:
        encoded = base64.b64encode(text_to_speak.encode("utf-16le")).decode("ascii")
        script = (
            "$b=[Convert]::FromBase64String('" + encoded + "');"
            "$t=[System.Text.Encoding]::Unicode.GetString($b);"
            "Add-Type -AssemblyName System.Speech;"
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "$s.Speak($t);$s.Dispose()"
        )
        status_var.set("Speaking...")
        root.update_idletasks()
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=120, check=True
        )
        status_var.set("Speech completed")
    except Exception as exc:
        status_var.set("Speech failed")
        print("Speech error:", repr(exc))


def process_frame(frame):
    # Same histogram back-projection method as final.py
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    back_projection = cv2.calcBackProject(
        [hsv], [0, 1], hist, [0, 180, 0, 256], 1
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    back_projection = cv2.filter2D(back_projection, -1, kernel)
    blurred = cv2.GaussianBlur(back_projection, (11, 11), 0)
    blurred = cv2.medianBlur(blurred, 15)
    full_mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    height, width = full_mask.shape
    x = min(ROI_X, max(0, width - 1))
    y = min(ROI_Y, max(0, height - 1))
    w = min(ROI_W, width - x)
    h = min(ROI_H, height - y)
    mask = full_mask[y:y + h, x:x + w].copy()

    display_frame = frame.copy()
    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Reject small specks and noisy masks before running the CNN.
    white_ratio = cv2.countNonZero(mask) / float(mask.size)

    if white_ratio < MIN_MASK_WHITE_RATIO or white_ratio > MAX_MASK_WHITE_RATIO:
        return display_frame, mask, None

    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return display_frame, mask, None

    contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(contour)

    if contour_area < MIN_HAND_AREA:
        return display_frame, mask, None

    bx, by, bw, bh = cv2.boundingRect(contour)

    if bw < MIN_HAND_WIDTH or bh < MIN_HAND_HEIGHT:
        return display_frame, mask, None

    box_area = bw * bh
    if box_area <= 0:
        return display_frame, mask, None

    fill_ratio = contour_area / float(box_area)
    if fill_ratio < 0.10 or fill_ratio > 0.95:
        return display_frame, mask, None

    hand_crop = mask[by:by + bh, bx:bx + bw]
    if hand_crop.size == 0:
        return display_frame, mask, None

    side = max(bw, bh)
    square = np.zeros((side, side), dtype=np.uint8)
    offset_x = (side - bw) // 2
    offset_y = (side - bh) // 2
    square[offset_y:offset_y + bh, offset_x:offset_x + bw] = hand_crop
    hand_image = cv2.resize(
        square, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA
    )
    return display_frame, mask, hand_image


root = tk.Tk()
root.title("Sign Language Interpreter")
root.geometry("1450x900")
root.minsize(1100, 700)
root.configure(bg="#172033")

prediction_history = deque(maxlen=HISTORY_SIZE)
recognized_text = ""
last_added_class = None
camera = None
camera_running = False
closed = False

gesture_var = tk.StringVar(value="No hand detected")
confidence_var = tk.StringVar(value="Confidence: --")
status_var = tk.StringVar(value="Starting camera...")

header = tk.Frame(root, bg="#172033")
header.pack(fill="x", padx=18, pady=(12, 6))
tk.Label(
    header, text="SIGN LANGUAGE INTERPRETER",
    font=("Segoe UI", 20, "bold"), fg="white", bg="#172033"
).pack(anchor="w")
tk.Label(
    header, text="Live gesture recognition — trained classes A-D",
    font=("Segoe UI", 10), fg="#aab7cf", bg="#172033"
).pack(anchor="w", pady=(2, 0))

main = tk.Frame(root, bg="#172033")
main.pack(fill="both", expand=True, padx=18, pady=8)
main.grid_rowconfigure(0, weight=1)
main.grid_columnconfigure(0, weight=3, minsize=650)
main.grid_columnconfigure(1, weight=2, minsize=380)

camera_panel = tk.Frame(main, bg="#222e45", padx=12, pady=12)
camera_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
tk.Label(
    camera_panel, text="LIVE CAMERA", font=("Segoe UI", 11, "bold"),
    fg="white", bg="#222e45"
).pack(anchor="w", pady=(0, 8))

video_label = tk.Label(
    camera_panel, bg="#0b1220", text="Opening camera...",
    fg="white", font=("Segoe UI", 12)
)
video_label.pack(fill="both", expand=True)
tk.Label(
    camera_panel, text="Keep your hand inside the green box.",
    font=("Segoe UI", 9), fg="#aab7cf", bg="#222e45"
).pack(anchor="w", pady=(8, 0))

right_panel = tk.Frame(main, bg="#222e45", padx=14, pady=12)
right_panel.grid(row=0, column=1, sticky="nsew")

tk.Label(
    right_panel, text="BLACK AND WHITE THRESHOLD",
    font=("Segoe UI", 11, "bold"), fg="white", bg="#222e45"
).pack(anchor="w", pady=(0, 8))

# A fixed-size image label: no fill/expand that could distort its display.
mask_label = tk.Label(
    right_panel, bg="black", fg="white",
    width=MASK_WIDTH, height=MASK_HEIGHT,
    relief="solid", bd=2, anchor="center"
)
mask_label.pack(anchor="center", pady=(0, 8))
mask_label.configure(width=MASK_WIDTH, height=MASK_HEIGHT)

tk.Label(
    right_panel, textvariable=gesture_var, font=("Segoe UI", 24, "bold"),
    fg="#63e6be", bg="#222e45", wraplength=350
).pack(anchor="w", pady=(4, 2))
tk.Label(
    right_panel, textvariable=confidence_var, font=("Segoe UI", 10),
    fg="white", bg="#222e45"
).pack(anchor="w")

tk.Label(
    right_panel, text="RECOGNIZED TEXT", font=("Segoe UI", 10, "bold"),
    fg="#aab7cf", bg="#222e45"
).pack(anchor="w", pady=(10, 5))

text_box = tk.Text(
    right_panel, height=3, font=("Consolas", 13), wrap="word",
    bg="#101827", fg="white", insertbackground="white",
    relief="flat", padx=8, pady=8, state="disabled"
)
text_box.pack(fill="x")


def refresh_text():
    text_box.configure(state="normal")
    text_box.delete("1.0", "end")
    text_box.insert("1.0", recognized_text)
    text_box.configure(state="disabled")


def add_space():
    global recognized_text
    if recognized_text and not recognized_text.endswith(" "):
        recognized_text += " "
        refresh_text()


def backspace():
    global recognized_text
    recognized_text = recognized_text[:-1]
    refresh_text()


def clear_text():
    global recognized_text, last_added_class
    recognized_text = ""
    last_added_class = None
    prediction_history.clear()
    refresh_text()
    status_var.set("Text cleared")


button_frame = tk.Frame(right_panel, bg="#222e45")
button_frame.pack(fill="x", pady=8)
button_frame.grid_columnconfigure(0, weight=1)
button_frame.grid_columnconfigure(1, weight=1)


def make_button(label, command, row, col, color="#354663"):
    tk.Button(
        button_frame, text=label, command=command,
        font=("Segoe UI", 10, "bold"), bg=color, fg="white",
        activebackground="#496184", activeforeground="white",
        relief="flat", cursor="hand2", padx=8, pady=7
    ).grid(row=row, column=col, sticky="ew", padx=4, pady=3)


make_button("Speak", lambda: speak_text(recognized_text), 0, 0, "#16803d")
make_button("Space", add_space, 0, 1)
make_button("Backspace", backspace, 1, 0)
make_button("Clear", clear_text, 1, 1, "#b91c1c")

tk.Label(
    right_panel, text="S: Space  |  B: Backspace  |  C: Clear  |  V: Speak  |  Q: Quit",
    justify="left", font=("Segoe UI", 9), fg="white", bg="#222e45",
    wraplength=360
).pack(anchor="w", pady=(2, 6))
tk.Label(
    right_panel, textvariable=status_var, font=("Segoe UI", 9),
    fg="#63e6be", bg="#222e45", wraplength=360
).pack(anchor="w")


def update_camera():
    global last_added_class, recognized_text

    if closed or not camera_running:
        return

    ret, frame = camera.read()
    if not ret or frame is None:
        status_var.set("Could not read camera frame")
        root.after(100, update_camera)
        return

    frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
    display_frame, mask, hand_image = process_frame(frame)

    # Show a second threshold window, as in final.py.
    cv2.imshow("Threshold", mask)
    cv2.waitKey(1)

    prediction = None
    confidence = 0.0

    if hand_image is not None:
        confidence, class_id = predict_gesture(hand_image)
        if confidence >= CONFIDENCE_THRESHOLD:
            prediction_history.append(class_id)
            counts = Counter(prediction_history)
            common_class, count = counts.most_common(1)[0]
            if count >= MIN_STABLE_COUNT:
                prediction = common_class
        else:
            prediction_history.clear()
    else:
        prediction_history.clear()
        last_added_class = None

    if prediction is not None:
        name = get_gesture_name(prediction)
        gesture_var.set(name)
        confidence_var.set(f"Confidence: {confidence * 100:.1f}%")
        if name in ("A", "B", "C", "D") and prediction != last_added_class:
            recognized_text += name
            last_added_class = prediction
            refresh_text()
        status_var.set("Gesture recognized")
    elif hand_image is None:
        gesture_var.set("No hand detected")
        confidence_var.set("Confidence: --")
        status_var.set("No clear hand shape found — show your full hand inside the green box")
    else:
        gesture_var.set("Recognizing...")
        confidence_var.set(f"Confidence: {confidence * 100:.1f}%")
        status_var.set("Waiting for stable prediction")

    rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    camera_image = Image.fromarray(rgb).resize((700, 525), Image.Resampling.LANCZOS)
    camera_photo = ImageTk.PhotoImage(camera_image)
    video_label.configure(image=camera_photo, text="")
    video_label.image = camera_photo

    mask_image = Image.fromarray(mask).convert("L").resize(
        (MASK_WIDTH, MASK_HEIGHT), Image.Resampling.NEAREST
    )
    mask_photo = ImageTk.PhotoImage(mask_image)
    mask_label.configure(image=mask_photo, text="", width=MASK_WIDTH, height=MASK_HEIGHT)
    mask_label.image = mask_photo

    root.after(15, update_camera)


def close_application():
    global closed, camera_running
    if closed:
        return
    closed = True
    camera_running = False
    if camera is not None:
        camera.release()
    cv2.destroyAllWindows()
    root.destroy()


def on_key(event):
    key = event.keysym.lower()
    if key == "v":
        speak_text(recognized_text)
    elif key == "s":
        add_space()
    elif key == "b":
        backspace()
    elif key == "c":
        clear_text()
    elif key == "q":
        close_application()


root.bind_all("<Key>", on_key)
root.protocol("WM_DELETE_WINDOW", close_application)

try:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        messagebox.showerror("Camera Error", "Could not open the webcam.")
        root.destroy()
    else:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        camera_running = True
        status_var.set("Camera started")
        root.after(100, update_camera)
        root.mainloop()
finally:
    camera_running = False
    if camera is not None:
        camera.release()
    cv2.destroyAllWindows()

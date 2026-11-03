
import cv2
import numpy as np
import pickle
import os
import sqlite3


# Image and capture settings
IMAGE_SIZE = 50
TOTAL_IMAGES = 1200

# Region of Interest (ROI)
ROI_X, ROI_Y, ROI_W, ROI_H = 300, 100, 300, 300


# Load hand histogram
def get_hand_hist():

    with open("hist", "rb") as file:
        return pickle.load(file)


# Create gesture folder and database
def init_create_folder_database():

    os.makedirs("gestures", exist_ok=True)

    conn = sqlite3.connect("gesture_db.db")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS gesture (
            g_id INTEGER PRIMARY KEY,
            g_name TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# Create a folder for a gesture
def create_folder(folder_name):

    os.makedirs(folder_name, exist_ok=True)


# Register gesture in database
def store_in_db(g_id, g_name):

    conn = sqlite3.connect("gesture_db.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT g_name FROM gesture WHERE g_id = ?",
        (g_id,)
    )

    result = cursor.fetchone()

    if result:

        if result[0].strip() == g_name.strip():
            print("Gesture already registered:", g_id, g_name)
            conn.close()
            return True

        print("Error: Gesture ID already belongs to", result[0])
        conn.close()
        return False

    cursor.execute(
        "SELECT g_id FROM gesture WHERE LOWER(TRIM(g_name)) = LOWER(?)",
        (g_name.strip(),)
    )

    if cursor.fetchone():
        print("Error: Gesture name already registered.")
        conn.close()
        return False

    cursor.execute(
        "INSERT INTO gesture (g_id, g_name) VALUES (?, ?)",
        (g_id, g_name)
    )

    conn.commit()
    conn.close()

    print("Gesture registered successfully:", g_id, g_name)
    return True


# Collect and save gesture images
def store_images(g_id):

    folder_path = os.path.join("gestures", str(g_id))
    create_folder(folder_path)

    # Check existing images
    existing_images = [
        file for file in os.listdir(folder_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    saved_count = len(existing_images)

    if saved_count >= TOTAL_IMAGES:
        print("This gesture already has", saved_count, "images.")
        return

    # Find the next image number
    image_numbers = [
        int(os.path.splitext(file)[0])
        for file in existing_images
        if os.path.splitext(file)[0].isdigit()
    ]

    next_image_no = max(image_numbers, default=0) + 1

    print("Existing images:", saved_count)
    print("Images remaining:", TOTAL_IMAGES - saved_count)

    # Load histogram
    try:
        hist = get_hand_hist()
    except FileNotFoundError:
        print("Error: hist file not found.")
        return

    # Open webcam
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Cannot open camera.")
        return

    x, y, w, h = ROI_X, ROI_Y, ROI_W, ROI_H

    image_count = saved_count
    capturing = False
    frames = 0

    print("\nControls:")
    print("C - Start/Pause capturing")
    print("Q - Quit collection")

    try:

        while image_count < TOTAL_IMAGES:

            ret, img = cam.read()

            if not ret or img is None:
                print("Unable to read webcam frame.")
                break

            # Flip webcam image
            img = cv2.flip(img, 1)
            img = cv2.resize(img, (640, 480))

            height, width = img.shape[:2]

            if x + w > width or y + h > height:
                print("Error: ROI is outside the camera frame.")
                break

            # Convert image to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Apply histogram backprojection
            dst = cv2.calcBackProject(
                [hsv],
                [0, 1],
                hist,
                [0, 180, 0, 256],
                1
            )

            # Apply filtering
            disc = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (10, 10)
            )

            cv2.filter2D(dst, -1, disc, dst)

            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)

            # Convert image into binary format
            thresh = cv2.threshold(
                blur,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]

            # Extract ROI
            thresh = thresh[y:y+h, x:x+w]

            # Find contours
            contours, _ = cv2.findContours(
                thresh.copy(),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_NONE
            )

            # Capture images after initial frames
            if capturing and frames > 50 and frames % 3 == 0:

                if len(contours) > 0:

                    contour = max(contours, key=cv2.contourArea)

                    if cv2.contourArea(contour) > 10000:

                        x1, y1, w1, h1 = cv2.boundingRect(contour)

                        save_img = thresh[y1:y1+h1, x1:x1+w1]

                        if save_img.size > 0:

                            # Make image square using padding
                            if w1 > h1:

                                padding = (w1 - h1) // 2

                                save_img = cv2.copyMakeBorder(
                                    save_img,
                                    padding,
                                    w1 - h1 - padding,
                                    0,
                                    0,
                                    cv2.BORDER_CONSTANT,
                                    value=0
                                )

                            elif h1 > w1:

                                padding = (h1 - w1) // 2

                                save_img = cv2.copyMakeBorder(
                                    save_img,
                                    0,
                                    0,
                                    padding,
                                    h1 - w1 - padding,
                                    cv2.BORDER_CONSTANT,
                                    value=0
                                )

                            # Resize and save image
                            save_img = cv2.resize(
                                save_img,
                                (IMAGE_SIZE, IMAGE_SIZE)
                            )

                            image_path = os.path.join(
                                folder_path,
                                str(next_image_no) + ".jpg"
                            )

                            if cv2.imwrite(image_path, save_img):
                                image_count += 1
                                next_image_no += 1

            if capturing:
                frames += 1

            # Display ROI and capture status
            cv2.rectangle(
                img,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            status = "CAPTURING" if capturing else "PAUSED"

            cv2.putText(
                img,
                status,
                (30, 60),
                cv2.FONT_HERSHEY_TRIPLEX,
                1,
                (0, 255, 255),
                2
            )

            cv2.putText(
                img,
                f"Images: {image_count}/{TOTAL_IMAGES}",
                (30, 400),
                cv2.FONT_HERSHEY_TRIPLEX,
                1,
                (127, 127, 255),
                2
            )

            cv2.imshow("Capturing gesture", img)
            cv2.imshow("thresh", thresh)

            key = cv2.waitKey(1) & 0xFF

            # Start or pause capturing
            if key == ord('c'):

                capturing = not capturing
                frames = 0

                print("Capturing started." if capturing else "Capturing paused.")

            # Stop collection
            elif key == ord('q'):

                print("Collection stopped by user.")
                break

    finally:

        cam.release()
        cv2.destroyAllWindows()

    print("\nCollection completed.")
    print("Total images saved:", image_count)
    print("Images remaining:", TOTAL_IMAGES - image_count)


# Main function
if __name__ == "__main__":

    init_create_folder_database()

    try:
        g_id = int(input("Enter gesture no.: "))

        if g_id < 0:
            print("Error: Gesture ID cannot be negative.")
            raise SystemExit

    except ValueError:
        print("Error: Gesture ID must be a number.")
        raise SystemExit

    g_name = input("Enter gesture name/text: ").strip()

    if not g_name:
        print("Error: Gesture name cannot be empty.")
        raise SystemExit

    if store_in_db(g_id, g_name):
        store_images(g_id)

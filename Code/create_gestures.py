
import cv2
import numpy as np
import pickle
import os
import sqlite3

image_x, image_y = 50, 50
total_pics = 1200

# Region of interest
ROI_X, ROI_Y, ROI_W, ROI_H = 300, 100, 300, 300


def get_hand_hist():
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist


def init_create_folder_database():
    os.makedirs("gestures", exist_ok=True)

    conn = sqlite3.connect("gesture_db.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gesture (
            g_id INTEGER NOT NULL PRIMARY KEY,
            g_name TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def create_folder(folder_name):
    os.makedirs(folder_name, exist_ok=True)


def store_in_db(g_id, g_name):
    conn = sqlite3.connect("gesture_db.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT g_name FROM gesture WHERE g_id = ?",
        (g_id,)
    )

    result = cursor.fetchone()

    if result is not None:
        if result[0].strip() == g_name.strip():
            print("Gesture already registered:", g_id, g_name)
            conn.close()
            return True

        print("Error: Gesture ID", g_id,
              "already belongs to", result[0])
        conn.close()
        return False

    cursor.execute(
        "SELECT g_id FROM gesture WHERE LOWER(TRIM(g_name)) = LOWER(?)",
        (g_name.strip(),)
    )

    if cursor.fetchone() is not None:
        print("Error: This gesture name is already registered.")
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


def store_images(g_id):
    folder_path = os.path.join("gestures", str(g_id))
    create_folder(folder_path)

    existing_images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    saved_count = len(existing_images)

    if saved_count >= total_pics:
        print("This gesture already has", saved_count, "images.")
        return

    numeric_names = [
        int(os.path.splitext(f)[0])
        for f in existing_images
        if os.path.splitext(f)[0].isdigit()
    ]

    next_image_no = max(numeric_names, default=0) + 1

    print("Existing images:", saved_count)
    print("Images remaining:", total_pics - saved_count)

    try:
        hist = get_hand_hist()
    except FileNotFoundError:
        print("Error: hist file not found.")
        return

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Cannot open camera.")
        return

    x, y, w, h = ROI_X, ROI_Y, ROI_W, ROI_H

    pic_count = saved_count
    flag_start_capturing = False
    frames = 0

    print("\nControls:")
    print("C - Start/Pause capturing")
    print("Q - Quit collection")

    try:
        while pic_count < total_pics:

            ret, img = cam.read()

            if not ret or img is None:
                print("Unable to read webcam frame.")
                break

            img = cv2.flip(img, 1)

            height, width = img.shape[:2]

            if x + w > width or y + h > height:
                print("Error: ROI is outside the camera frame.")
                break

            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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

            cv2.filter2D(dst, -1, disc, dst)

            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)

            thresh = cv2.threshold(
                blur,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]

            thresh = thresh[y:y+h, x:x+w]

            contours, _ = cv2.findContours(
                thresh.copy(),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_NONE
            )

            if flag_start_capturing and frames > 50 and frames % 3 == 0:

                if len(contours) > 0:

                    contour = max(
                        contours,
                        key=cv2.contourArea
                    )

                    if cv2.contourArea(contour) > 10000:

                        x1, y1, w1, h1 = cv2.boundingRect(contour)

                        save_img = thresh[
                            y1:y1+h1,
                            x1:x1+w1
                        ]

                        if save_img.size > 0:

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

                            save_img = cv2.resize(
                                save_img,
                                (image_x, image_y)
                            )

                            image_path = os.path.join(
                                folder_path,
                                str(next_image_no) + ".jpg"
                            )

                            success = cv2.imwrite(
                                image_path,
                                save_img
                            )

                            if success:
                                pic_count += 1
                                next_image_no += 1

            if flag_start_capturing:
                frames += 1

            cv2.rectangle(
                img,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            status = "CAPTURING" if flag_start_capturing else "PAUSED"

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
                f"Images: {pic_count}/{total_pics}",
                (30, 400),
                cv2.FONT_HERSHEY_TRIPLEX,
                1,
                (127, 127, 255),
                2
            )

            cv2.imshow("Capturing gesture", img)
            cv2.imshow("thresh", thresh)

            keypress = cv2.waitKey(1) & 0xFF

            if keypress == ord('c'):
                flag_start_capturing = not flag_start_capturing
                frames = 0

                if flag_start_capturing:
                    print("Capturing started.")
                else:
                    print("Capturing paused.")

            elif keypress == ord('q'):
                print("Collection stopped by user.")
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()

    print("\nCollection completed.")
    print("Total images saved:", pic_count)
    print("Images remaining:", total_pics - pic_count)


if __name__ == "__main__":

    init_create_folder_database()

    try:
        g_id = int(input("Enter gesture no.: "))
    except ValueError:
        print("Error: Gesture ID must be a number.")
        raise SystemExit

    g_name = input("Enter gesture name/text: ").strip()

    if not g_name:
        print("Error: Gesture name cannot be empty.")
        raise SystemExit

    if store_in_db(g_id, g_name):
        store_images(g_id)

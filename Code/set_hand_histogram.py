
import cv2
import numpy as np
import pickle


# Build squares for selecting the hand region
def build_squares(img):

    x, y = 420, 140
    w, h = 10, 10
    gap = 10

    rows = []

    for i in range(10):

        row = []

        for j in range(5):

            square = img[y:y+h, x:x+w].copy()
            row.append(square)

            cv2.rectangle(
                img,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                1
            )

            x += w + gap

        rows.append(np.hstack(row))

        x = 420
        y += h + gap

    return np.vstack(rows)


# Capture and calculate hand histogram
def get_hand_hist():

    # Open webcam
    cam = cv2.VideoCapture(1)

    if not cam.isOpened():
        cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Cannot open webcam.")
        return

    x, y, w, h = 300, 100, 300, 300

    capture_histogram = False
    save_histogram = False

    hist = None

    print("\nControls:")
    print("C - Capture hand histogram")
    print("S - Save histogram and exit")

    try:

        while True:

            ret, img = cam.read()

            if not ret or img is None:
                print("Unable to read webcam frame.")
                break

            # Flip and resize webcam image
            img = cv2.flip(img, 1)
            img = cv2.resize(img, (640, 480))

            # Convert image to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            key = cv2.waitKey(1) & 0xFF

            # Capture histogram using selected hand region
            if key == ord('c'):

                hand_img = build_squares(img.copy())

                hsv_crop = cv2.cvtColor(hand_img, cv2.COLOR_BGR2HSV)

                hist = cv2.calcHist(
                    [hsv_crop],
                    [0, 1],
                    None,
                    [180, 256],
                    [0, 180, 0, 256]
                )

                cv2.normalize(
                    hist,
                    hist,
                    0,
                    255,
                    cv2.NORM_MINMAX
                )

                capture_histogram = True
                print("Hand histogram captured.")

            # Apply histogram backprojection
            if capture_histogram:

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

                # Apply thresholding
                thresh = cv2.threshold(
                    blur,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1]

                cv2.imshow("Thresh", thresh)

            #if not save_histogram:
            display_img = img.copy()

                # Draw selection squares before histogram capture
            if not capture_histogram:
                build_squares(display_img)

            cv2.imshow("Set hand histogram", display_img)

            # Save histogram and exit
            if key == ord('s'):

                if hist is None:
                    print("Please capture the hand histogram first.")
                    continue

                save_histogram = True
                break

    finally:

        cam.release()
        cv2.destroyAllWindows()

    # Save histogram
    if save_histogram and hist is not None:

        with open("hist", "wb") as file:
            pickle.dump(hist, file)

        print("\nHand histogram saved successfully as 'hist'.")


# Run program
if __name__ == "__main__":
    get_hand_hist()

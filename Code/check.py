
import os
def count_gesture_images():

    for folder in sorted(os.listdir("gestures")):

        folder_path = os.path.join("gestures", folder)

        if os.path.isdir(folder_path):

            images = [
                file for file in os.listdir(folder_path)
                if file.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            print(f"Class {folder}: {len(images)} images")


if __name__ == "__main__":
    count_gesture_images()

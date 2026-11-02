
import os

for folder in sorted(os.listdir("gestures")):
    path = os.path.join("gestures", folder)

    if os.path.isdir(path):
        images = [
            f for f in os.listdir(path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"Class {folder}: {len(images)} images")

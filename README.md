# Sign Language Interpreter Using Deep Learning

A real-time sign language recognition application built using Python, OpenCV, TensorFlow/Keras, and Tkinter. The system uses a trained Convolutional Neural Network (CNN) to recognize hand gestures from a webcam and convert recognized gestures into text and speech.

## Features

- **Real-Time Hand Tracking:** Captures live video using a webcam.
- **Hand Segmentation:** Uses histogram back-projection and thresholding to separate the hand from the background.
- **Gesture Recognition:** Uses a trained CNN model to classify hand gestures.
- **Black-and-White Threshold View:** Displays the segmented hand mask for visual feedback.
- **Text Generation:** Adds recognized letters to a text area.
- **Text-to-Speech:** Reads recognized text aloud using Windows PowerShell speech synthesis.
- **Text Controls:** Supports adding spaces, deleting characters, clearing text, and speaking the generated text.
- **Graphical User Interface:** Provides a Tkinter-based interface for camera preview and gesture recognition.

## Technologies Used

- Python
- TensorFlow and Keras
- OpenCV
- NumPy
- Pillow
- Tkinter
- SQLite
- PowerShell System.Speech

## Project Structure

```text
Sign-Language-Interpreter-using-Deep-Learning/
│
├── Code/
│   ├── gestures/
│   ├── gui_app.py
│   ├── final.py
│   ├── cnn_model_train.py
│   ├── create_gestures.py
│   ├── display_gestures.py
│   ├── evaluate_model.py
│   ├── load_images.py
│   ├── set_hand_histogram.py
│   ├── Rotate_images.py
│   ├── cnn_model_keras2.h5
│   ├── gesture_db.db
│   ├── hist
│   ├── Install_Packages.txt
│   └── Install_Packages_gpu.txt
│
├── .gitignore
├── LICENSE
└── README.md
```

Additional training data files may be present in the `Code` directory.

## Requirements

- Python 3.11 (the project was tested with Python 3.11.4)
- A working webcam
- Windows for the included PowerShell text-to-speech feature
- Git, if you want to clone the repository

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the example URL with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it in Windows Command Prompt:

```bat
venv\Scripts\activate.bat
```

Or in PowerShell, if script execution is permitted:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Move into the project code directory:

```bash
cd Code
```

Install the required packages:

```bash
pip install numpy opencv-python tensorflow pillow scikit-learn h5py
```

If you use a different Python or TensorFlow version, package compatibility may vary.

## Running the Application

From inside the `Code` directory, run:

```bash
python gui_app.py
```

The application opens a graphical interface with a live webcam preview, a black-and-white threshold view, the current gesture, and the recognized text.

Keep your hand inside the green rectangle displayed in the camera preview.

To run the original recognition script instead, use:

```bash
python final.py
```

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `S` | Add a space |
| `B` | Delete the last character |
| `C` | Clear the recognized text |
| `V` | Speak the recognized text |
| `Q` | Quit the application |

The GUI also provides buttons for these actions.

## How It Works

1. The webcam captures a video frame.
2. OpenCV converts the frame to HSV color space.
3. Histogram back-projection helps segment the hand from the background.
4. Gaussian blur, median blur, and Otsu thresholding produce a black-and-white mask.
5. The largest suitable hand contour is extracted and resized to match the CNN input dimensions.
6. The trained CNN predicts the gesture class.
7. Stable predictions are added to the recognized text.
8. The text-to-speech function reads the generated text aloud when requested.

## Current Limitations

- The current trained model is configured for the gestures **A, B, C, and D**.
- Recognition quality depends on lighting, background conditions, hand positioning, and the quality of the segmentation mask.
- The model must be trained with additional gesture classes to support a larger alphabet or complete sign language.
- The included speech function uses Windows PowerShell and may require adaptation for other operating systems.
- This is a prototype and is not a complete sign language translation system.

## Future Improvements

- Expand the dataset to include all alphabet gestures.
- Improve hand segmentation under different lighting conditions.
- Add support for dynamic gestures and complete words.
- Improve recognition accuracy with additional training data.
- Support more languages and text-to-speech engines.
- Add evaluation metrics and more comprehensive testing.

## License

See the [LICENSE](LICENSE) file for the applicable license terms.

## Acknowledgements

This project builds on an existing deep-learning sign language interpreter project and has been adapted and tested as part of a learning project. Refer to the original repository and its license when documenting or redistributing the original work.

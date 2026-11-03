# Sign Language Interpreter Using Deep Learning

A real-time Sign Language Interpreter built using Python, OpenCV, and Convolutional Neural Networks (CNN). The system uses a webcam to detect hand gestures, recognize predefined signs, convert them into text, and provide speech output.

The project aims to make communication easier between people who use sign language and those who do not understand it.

## Features

- Real-time hand gesture detection using a webcam.
- Hand segmentation using HSV histogram backprojection.
- Gesture classification using a Convolutional Neural Network (CNN).
- Recognition of seven trained gesture classes.
- Conversion of recognized gestures into text.
- Text-to-speech output.
- Stable prediction using a sliding prediction history.
- Keyboard controls for text editing and speech output.

## Supported Gestures

The current trained model recognizes the following classes:

| Gesture | Database ID |
|---|---:|
| A | 0 |
| B | 1 |
| C | 2 |
| D | 3 |
| Best of Luck | 36 |
| Love | 41 |
| Victory | 44 |

**Note:** The current model classifies static gesture images. Recognition of multi-sign phrases and dynamic gestures requires sequence-based recognition.

## Technologies Used

- **Programming Language:** Python 3.11
- **Deep Learning:** TensorFlow / Keras
- **Computer Vision:** OpenCV
- **Numerical Processing:** NumPy
- **Data Processing:** Pandas
- **Data Visualization:** Matplotlib
- **Database:** SQLite
- **Speech Output:** Windows PowerShell System.Speech

## CNN Architecture

The project uses a Sequential CNN architecture for gesture classification.

| Layer | Configuration |
|---|---|
| Input | 50 × 50 × 1 |
| Conv2D | 16 filters, 2 × 2 |
| MaxPooling2D | 2 × 2 |
| Conv2D | 32 filters, 3 × 3 |
| MaxPooling2D | 3 × 3 |
| Conv2D | 64 filters, 5 × 5 |
| MaxPooling2D | 5 × 5 |
| Flatten | Feature extraction |
| Dense | 128 neurons, ReLU |
| Dropout | 0.2 |
| Output | 7 neurons, Softmax |

**Training configuration:**

- Epochs: 15
- Batch size: 64
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Categorical Crossentropy

## Dataset

The dataset contains 8,400 gesture images distributed across seven classes.

Each class contains 1,200 images.

| Dataset | Images |
|---|---:|
| Training | 7,000 |
| Validation | 700 |
| Testing | 700 |
| **Total** | **8,400** |

Images are converted into grayscale and resized to 50 × 50 pixels before being passed to the CNN.

## Model Performance

The current training run produced the following results:

| Metric | Result |
|---|---:|
| Training Accuracy (final epoch) | 99.76% |
| Best Validation Accuracy | 98.00% |
| Test Accuracy | 98.00% |
| Test Loss | 0.1007 |
| Total Parameters | 65,207 |

**Evaluation note:** These results are from a random image-level dataset split. Similar frames from the same recording session may appear in different subsets. Therefore, the reported accuracy does not establish generalization to unseen people, environments, or recording sessions.

## Project Structure

```text
Sign-Language-Interpreter-using-Deep-Learning/
│
├── Code/
│   ├── gestures/
│   │   ├── 0/
│   │   ├── 1/
│   │   ├── 2/
│   │   ├── 3/
│   │   ├── 36/
│   │   ├── 41/
│   │   └── 44/
│   │
│   ├── create_gestures.py
│   ├── load_images.py
│   ├── cnn_model_train.py
│   ├── evaluate_model.py
│   ├── final.py
│   ├── set_hand_histogram.py
│   ├── gesture_db.db
│   ├── hist
│   ├── class_mapping.json
│   └── cnn_model_7classes.h5
│
├── README.md
├── LICENSE
└── .gitignore
```

Additional generated files include training history, accuracy graphs, loss graphs, and dataset pickle files.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Sign-Language-Interpreter-using-Deep-Learning.git
```

### 2. Navigate to the project directory

```bash
cd Sign-Language-Interpreter-using-Deep-Learning
cd Code
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, use Command Prompt:

```cmd
venv\Scripts\activate.bat
```

### 5. Install dependencies

```bash
python -m pip install tensorflow opencv-python numpy pandas matplotlib
```

## Running the Project

Run the following scripts from the `Code` directory.

### Step 1: Set up hand histogram

```bash
python set_hand_histogram.py
```

This captures the hand color histogram used for segmentation.

### Step 2: Collect gesture images

```bash
python create_gestures.py
```

Enter the gesture ID and gesture name when prompted. The script captures and stores gesture images.

### Step 3: Load and split the dataset

```bash
python load_images.py
```

This loads the gesture images, preprocesses them, and creates training, validation, and testing datasets.

### Step 4: Train the CNN model

```bash
python cnn_model_train.py
```

The model is trained and saved as:

```text
cnn_model_7classes.h5
```

The training process also generates accuracy and loss graphs.

### Step 5: Run real-time sign language recognition

```bash
python final.py
```

Make sure the trained model, `class_mapping.json`, `hist`, and `gesture_db.db` are available in the working directory.

## Keyboard Controls

| Key | Function |
|---|---|
| S | Insert a space |
| B | Delete the last character |
| C | Clear recognized text |
| V | Convert recognized text into speech |
| Q | Exit the application |

## How It Works

1. The webcam captures the user's hand gestures.
2. OpenCV processes the frames and segments the hand using histogram backprojection.
3. The detected hand region is extracted and resized to 50 × 50 pixels.
4. The CNN predicts the gesture class.
5. A confidence threshold and prediction history help stabilize recognition.
6. The predicted class is mapped to its corresponding database gesture ID.
7. The recognized gesture is displayed as text.
8. The text can be converted into speech using the system's speech synthesizer.

## Limitations

- The current model recognizes only the seven trained classes.
- Recognition performance depends on lighting, background, camera quality, and hand positioning.
- The model is trained for static gesture images.
- Dynamic signs and complete multi-sign phrases require a sequence-based deep learning model.
- The current evaluation uses a random image-level split and requires further testing with unseen signers and recording sessions.
- Speech output currently uses Windows PowerShell System.Speech.

## Future Enhancements

- Expand the dataset to cover more Indian Sign Language (ISL) signs.
- Implement dynamic gesture recognition using LSTM or other sequence-based models.
- Improve recognition across different users and lighting conditions.
- Add sentence-level sign recognition.
- Improve text-to-speech support and multilingual output.
- Evaluate performance using independent signers and recording sessions.

## Contribution

Contributions, suggestions, and improvements are welcome.

If you would like to contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## License

This project is intended for educational and research purposes. Please refer to the repository's LICENSE file for licensing information.
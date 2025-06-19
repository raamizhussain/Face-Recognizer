# Face Recognition with OpenCV and LBPH
This project demonstrates face recognition using OpenCV's LBPH (Local Binary Patterns Histograms) algorithm. The system is trained on a dataset of 5 celebrities and can recognize faces from new input images.

## ✅ Features
Uses Haar Cascade Classifier for face detection.

Uses LBPH Face Recognizer for training and prediction.

Trains on grayscale cropped face regions to improve accuracy.

Saves model and data for later reuse.

Predicts and displays the name of the person in test images along with confidence score.

## 👨‍🏫 Classes
The model is trained to recognize the following celebrities:

Anushka Sharma

Cristiano Ronaldo

Deepika Padukone

Shahrukh Khan

Virat Kohli

## 🚀 How to Run
1. Clone this repository
git clone https://github.com/yourusername/face-recognition-opencv.git
cd face-recognition-opencv

2. Prepare the Dataset
Place the dataset in a folder named Photos, structured like this:

Photos/
├── Anushka Sharma/
├── Cristiano Ronaldo/
├── Deepika Padukone/
├── Shahrukh Khan/
└── Virat Kohli/
Each folder should contain ~30 frontal face images of the respective celebrity.

3. Train the Model
Open and run train1.ipynb to train the model:

Detects faces using haar_face.xml.

Extracts and stores grayscale ROI (Region of Interest).

Trains an LBPH model and saves:

face_trained.yml

features.npy

labels.npy

4. Test the Model
Run recognizer1.ipynb to test recognition on a new image:

Loads the trained model.

Detects faces in a new input image.

Predicts identity and confidence.

Displays the image with bounding boxes and labels.

## 🧠 Requirements
Python 3.x

OpenCV (opencv-python, opencv-contrib-python)

NumPy

Install dependencies:
pip install opencv-python opencv-contrib-python numpy
📷 Example Output
When a known face is detected:

Deepika Padukone with a confidence of 48.63
<img src="example_output.jpg" alt="Detected Face" width="500"/>
📌 Notes
haar_face.xml must be in the project root or correctly referenced.

Confidence value: lower means more confident (typical range is 0–100).

Try to use clear, frontal images for better results.

## 📃 License
This project is for educational use.

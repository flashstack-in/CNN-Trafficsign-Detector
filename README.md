#  Traffic Sign Recognition using CNN

This project uses a Convolutional Neural Network (CNN) to detect and classify traffic signs in real-time using a webcam. It is trained on the GTSRB (German Traffic Sign Recognition Benchmark) dataset.

##  Features
- Real-time traffic sign recognition using OpenCV
- Pre-trained CNN model loaded from `.keras` file
- Live video capture and prediction overlay
- Classifies 43 traffic sign types

##  Technologies Used
- TensorFlow / Keras
- OpenCV
- NumPy
- Python 3

##  Files
- `traffic_sign_detection.py`: Main Python script for real-time detection
- `traffic_sign_cnn_model.keras`: Pre-trained CNN model file
- `requirements.txt`: Python dependencies

##  How to Run

 **Clone this repository**
```bash
git clone https://github.com/flashstack-in/traffic-sign-recognition.git
cd traffic-sign-recognition
```

 **Install dependencies**
```bash
pip install -r requirements.txt
```

 **Run the script**
```bash
python traffic_sign_detection.py
```
Make sure your webcam is connected. Press `q` to exit the video window.

## Dataset
Trained on [GTSRB Dataset](https://benchmark.ini.rub.de/gtsrb_news.html), containing 43 traffic sign classes.

##  Author
**Abhiraj**  
Electrical Engineering Student  



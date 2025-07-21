import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model("traffic_sign_cnn_model.keras")

# Define class labels
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited', 17: 'No entry', 18: 'General caution',
    19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians',
    28: 'Children crossing', 29: 'Bicycles crossing', 30: 'Beware of ice/snow',
    31: 'Wild animals crossing', 32: 'End speed + passing limits',
    33: 'Turn right ahead', 34: 'Turn left ahead', 35: 'Ahead only',
    36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right',
    39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End no passing vehicle with a weight greater than 3.5 tons'
}

# Preprocess frame for prediction
def preprocess_frame(frame):
    image = cv2.resize(frame, (32, 32))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# OpenCV video capture and real-time prediction loop
def run_detection():
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip frame for natural webcam experience
        frame = cv2.flip(frame, 1)

        # Preprocess and predict
        processed_frame = preprocess_frame(frame)
        predictions = model.predict(processed_frame)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = classes.get(predicted_class, "Unknown")

        # Overlay prediction on frame
        cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow("Captured Frame", frame)
        cv2.waitKey(0)


        # Check for user interrupt (press 'q' to exit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the detection
if __name__ == "__main__":
    run_detection()



import cv2
import numpy as np
import tensorflow as tf

# Step 1: Capture Fingerprint Image
def capture_fingerprint():
    # Open camera
    cap = cv2.VideoCapture(0)

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the captured frame
    cv2.imshow('Captured Image', frame)
    
    # Save the captured image
    cv2.imwrite('fingerprint_image.jpg', frame)

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

# Step 2: Preprocess Fingerprint Image
def preprocess_image(image):
    # Preprocessing steps (e.g., resize, normalize, etc.)
    processed_image = cv2.resize(image, (128, 128))  # Resize image to input size of CNN
    processed_image = processed_image / 255.0  # Normalize pixel values
    return processed_image

# Step 3: Load Pretrained CNN Model
def load_cnn_model():
    # Load pretrained CNN model
    model = tf.keras.models.load_model('fingerprint_cnn_model.h5')
    return model

# Step 4: Perform Fingerprint Recognition
def recognize_fingerprint(image, model):
    # Preprocess image
    processed_image = preprocess_image(image)

    # Reshape image for CNN input
    processed_image = np.expand_dims(processed_image, axis=0)

    # Perform fingerprint recognition using the CNN model
    predicted_id = model.predict(processed_image)

    return predicted_id

# Step 5: Display Recognition Result
def display_recognition_result(predicted_id):
    print("Predicted ID:", predicted_id)

# Main function
def main():
    # Step 1: Capture Fingerprint Image
    capture_fingerprint()

    # Step 2: Load Pretrained CNN Model
    cnn_model = load_cnn_model()

    # Step 3: Load captured image
    fingerprint_image = cv2.imread('fingerprint_image.jpg')

    # Step 4: Perform Fingerprint Recognition
    predicted_id = recognize_fingerprint(fingerprint_image, cnn_model)

    # Step 5: Display Recognition Result
    display_recognition_result(predicted_id)

# Entry point
if __name__ == "__main__":
    main()

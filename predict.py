import cv2
import numpy as np
import tensorflow as tf
import json

def load_model_and_classes():
    # Load the trained model
    model = tf.keras.models.load_model('isl_model.h5')
    
    # Load class names
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    
    return model, class_names

def preprocess_frame(frame, target_size=(224, 224)):
    # Extract ROI and preprocess
    roi = frame[100:400, 100:400]
    roi = cv2.resize(roi, target_size)
    roi = roi / 255.0
    return roi

def predict_gesture(model, frame, class_names):
    # Preprocess the frame
    processed_frame = preprocess_frame(frame)
    
    # Make prediction
    prediction = model.predict(np.expand_dims(processed_frame, axis=0))
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return predicted_class, confidence

def main():
    # Load model and class names
    print("Loading model...")
    model, class_names = load_model_and_classes()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
        
        # Make prediction
        predicted_class, confidence = predict_gesture(model, frame, class_names)
        
        # Display prediction and confidence
        text = f"Prediction: {predicted_class}"
        conf_text = f"Confidence: {confidence:.2f}"
        
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, conf_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('ISL Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
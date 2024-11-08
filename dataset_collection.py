import cv2
import os
import time
import numpy as np
from datetime import datetime

class DatasetCollector:
    def __init__(self):
        self.roi_size = 300
        self.min_brightness = 40
        self.max_brightness = 250
        self.blur_threshold = 100
        self.frame_size = (640, 480)
        
    def create_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def check_image_quality(self, frame):
        """Check if the image meets quality standards"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check brightness
        brightness = np.mean(gray)
        if brightness < self.min_brightness or brightness > self.max_brightness:
            return False, "Poor lighting"
        
        # Check blur using Laplacian variance
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < self.blur_threshold:
            return False, "Image too blurry"
        
        # Check contrast
        contrast = np.std(gray)
        if contrast < 20:
            return False, "Low contrast"
            
        return True, "Good quality"
    
    def enhance_image(self, frame):
        """Enhance image quality"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab_planes = list(cv2.split(lab))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def collect_data(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        
        # Define classes (0-9 and A-Z)
        classes = list(range(10)) + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        # Create dataset directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_dir = f'dataset_{timestamp}'
        self.create_directory(dataset_dir)
        
        # Create metadata file
        metadata_file = os.path.join(dataset_dir, 'metadata.txt')
        
        for class_name in classes:
            class_dir = os.path.join(dataset_dir, str(class_name))
            self.create_directory(class_dir)
            
            print(f"\nCollecting data for class: {class_name}")
            print("Instructions:")
            print("- Keep hand within the green box")
            print("- Press 'c' to capture (only good quality images will be saved)")
            print("- Press 'q' to move to next class")
            print("- Press 'esc' to exit completely")
            
            img_counter = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Define ROI
                center_x = frame.shape[1] // 2
                center_y = frame.shape[0] // 2
                roi_x = center_x - self.roi_size // 2
                roi_y = center_y - self.roi_size // 2
                
                # Extract and draw ROI
                roi = frame[roi_y:roi_y+self.roi_size, 
                          roi_x:roi_x+self.roi_size]
                cv2.rectangle(frame, 
                            (roi_x, roi_y), 
                            (roi_x+self.roi_size, roi_y+self.roi_size), 
                            (0, 255, 0), 2)
                
                # Check image quality
                quality_ok, message = self.check_image_quality(roi)
                
                # Display quality message
                color = (0, 255, 0) if quality_ok else (0, 0, 255)
                cv2.putText(frame, f"Quality: {message}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, color, 2)
                
                # Display frame
                cv2.imshow('Data Collection', frame)
                
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    return
                elif key == ord('q'):
                    break
                elif key == ord('c') and quality_ok:
                    # Save original color image
                    enhanced_roi = self.enhance_image(roi)
                    color_path = os.path.join(class_dir, f'color_{img_counter}.jpg')
                    cv2.imwrite(color_path, enhanced_roi)
                    
                    # Save grayscale version
                    gray_roi = cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2GRAY)
                    gray_path = os.path.join(class_dir, f'gray_{img_counter}.jpg')
                    cv2.imwrite(gray_path, gray_roi)
                    
                    # Log metadata
                    with open(metadata_file, 'a') as f:
                        f.write(f"Class: {class_name}, Image: {img_counter}, "
                               f"Quality: {message}, "
                               f"Timestamp: {datetime.now()}\n")
                    
                    print(f"Images captured for {class_name}: {img_counter + 1}")
                    img_counter += 1
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = DatasetCollector()
    collector.collect_data()
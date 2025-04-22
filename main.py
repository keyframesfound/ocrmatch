import cv2
import json
from easyocr import Reader
import numpy as np
from flask import Flask
import time

app = Flask(__name__)

class OCRMatcher:
    def __init__(self):
        self.cap = None
        self.reader = None
        self.text_data = self.load_text_data()
        self.last_ocr_time = 0
        self.OCR_COOLDOWN = 1.0  # Seconds between OCR attempts
        
    def load_text_data(self):
        try:
            with open('data.json', 'r') as f:
                return json.load(f)['texts']
        except Exception as e:
            print(f"Error loading text data: {e}")
            return []

    def setup_webcam(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open webcam")
            # Set higher resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return True
        except Exception as e:
            print(f"Error setting up webcam: {e}")
            return False

    def setup_ocr(self):
        try:
            self.reader = Reader(['en'], gpu=True)  # Enable GPU if available
            return True
        except Exception as e:
            print(f"Error setting up OCR: {e}")
            return False

    def process_ocr(self, frame):
        try:
            # Convert to grayscale for better OCR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply thresholding to improve text detection
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            results = self.reader.readtext(thresh)
            detected_text = ' '.join([text[1] for text in results])
            
            # Draw rectangles around detected text
            for (bbox, text, prob) in results:
                if prob > 0.5:  # Only show confident detections
                    points = np.array(bbox).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], True, (0, 255, 0), 2)
                    cv2.putText(frame, f"{text} ({prob:.2f})", (int(bbox[0][0]), int(bbox[0][1])-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return detected_text, frame
        except Exception as e:
            print(f"Error performing OCR: {e}")
            return None, frame

    def compare_texts(self, detected_text):
        if not detected_text:
            return False, None
            
        for reference_text in self.text_data:
            # Case insensitive comparison
            if reference_text.lower() in detected_text.lower():
                return True, reference_text
        return False, None

    def run(self):
        if not self.setup_webcam() or not self.setup_ocr():
            return

        print("Controls:")
        print("Press 'o' to perform OCR")
        print("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Show original frame
                cv2.imshow('Webcam Feed', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('o'):
                    current_time = time.time()
                    if current_time - self.last_ocr_time >= self.OCR_COOLDOWN:
                        print("Processing OCR...")
                        detected_text, annotated_frame = self.process_ocr(frame.copy())
                        
                        if detected_text:
                            match_found, matched_text = self.compare_texts(detected_text)
                            if match_found:
                                print(f"Match found: '{matched_text}'")
                                # Draw green border for match
                                cv2.rectangle(annotated_frame, (0, 0), 
                                           (annotated_frame.shape[1], annotated_frame.shape[0]), 
                                           (0, 255, 0), 10)
                            else:
                                print("No match found")
                                # Draw red border for no match
                                cv2.rectangle(annotated_frame, (0, 0), 
                                           (annotated_frame.shape[1], annotated_frame.shape[0]), 
                                           (0, 0, 255), 10)
                        
                        cv2.imshow('OCR Result', annotated_frame)
                        self.last_ocr_time = current_time
                    else:
                        print(f"Please wait {self.OCR_COOLDOWN - (current_time - self.last_ocr_time):.1f} seconds before next OCR")
                
                elif key == ord('q'):
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    matcher = OCRMatcher()
    matcher.run()

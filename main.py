import cv2
import json
from easyocr import Reader
import numpy as np
from flask import Flask
import time
from rapidfuzz import fuzz

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
                data = json.load(f)
                if not isinstance(data, dict) or 'texts' not in data or not isinstance(data['texts'], list):
                    raise ValueError("JSON format invalid: must be an object with a 'texts' list")
                return data['texts']
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
            return []
        matches = []
        for reference_text in self.text_data:
            # Fuzzy match: consider a match if similarity is above 80
            score = fuzz.partial_ratio(reference_text.lower().strip(), detected_text.lower().strip())
            if score > 80:
                matches.append((reference_text, score))
        return matches

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

                cv2.imshow('Webcam Feed', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('o'):
                    current_time = time.time()
                    if current_time - self.last_ocr_time >= self.OCR_COOLDOWN:
                        print("Processing OCR...")
                        detected_text, annotated_frame = self.process_ocr(frame.copy())
                        print(f"Detected text: '{detected_text}'")  # Debug print
                        if detected_text:
                            matches = self.compare_texts(detected_text)
                            if matches:
                                print(f"Matches found: {matches}")
                                cv2.rectangle(annotated_frame, (0, 0), 
                                           (annotated_frame.shape[1], annotated_frame.shape[0]), 
                                           (0, 255, 0), 10)
                            else:
                                print("No match found")
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

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
        self.selecting_region = False
        self.region_start = None
        self.region_end = None
        self.region_selected = False
        self.region_img = None
        
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

    def process_ocr_region(self, frame, region):
        x1, y1 = region[0]
        x2, y2 = region[1]
        x1, x2 = sorted([max(0, x1), max(0, x2)])
        y1, y2 = sorted([max(0, y1), max(0, y2)])
        # Ensure region is within frame bounds
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        # Prevent zero-area or invalid region
        if x2 - x1 < 5 or y2 - y1 < 5:
            print("Selected region is too small or invalid. Please select a larger region.")
            return '', frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else frame
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results = self.reader.readtext(thresh)
        detected_text = ''.join([text[1] for text in results])  # Concatenate as one long string
        for (bbox, text, prob) in results:
            if prob > 0.5:
                points = np.array(bbox).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(roi, [points], True, (0, 255, 0), 2)
        return detected_text, roi

    def compare_texts(self, detected_text):
        if not detected_text:
            return []
        matches = []
        for reference_text in self.text_data:
            # Fuzzy match: consider a match if similarity is above 80
            score = fuzz.partial_ratio(reference_text.lower().strip(), detected_text.lower().strip())
            if score > 65:
                matches.append((reference_text, score))
        return matches

    def region_mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.region_start = (x, y)
            self.region_end = (x, y)
            self.selecting_region = True
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting_region:
            self.region_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.selecting_region:
            self.region_end = (x, y)
            self.selecting_region = False
            self.region_selected = True

    def run(self):
        if not self.setup_webcam() or not self.setup_ocr():
            return

        print("Controls:")
        print("Press 'o' to perform OCR on full frame")
        print("Press 'u' to select a region and perform OCR on it")
        print("Press 'q' to quit")
        
        cv2.namedWindow('Webcam Feed')
        cv2.setMouseCallback('Webcam Feed', self.region_mouse_callback)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                display_frame = frame.copy()
                if self.selecting_region and self.region_start and self.region_end:
                    cv2.rectangle(display_frame, self.region_start, self.region_end, (255, 0, 0), 2)
                elif self.region_selected and self.region_start and self.region_end:
                    cv2.rectangle(display_frame, self.region_start, self.region_end, (0, 255, 255), 2)
                
                cv2.imshow('Webcam Feed', display_frame)
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
                elif key == ord('u'):
                    print("Select a region with your mouse (click and drag). Press 'u' again to OCR the selected region.")
                    self.region_selected = False
                    while True:
                        ret, frame = self.cap.read()
                        if not ret:
                            break
                        display_frame = frame.copy()
                        if self.selecting_region and self.region_start and self.region_end:
                            cv2.rectangle(display_frame, self.region_start, self.region_end, (255, 0, 0), 2)
                        elif self.region_selected and self.region_start and self.region_end:
                            cv2.rectangle(display_frame, self.region_start, self.region_end, (0, 255, 255), 2)
                        cv2.imshow('Webcam Feed', display_frame)
                        k = cv2.waitKey(1) & 0xFF
                        if self.region_selected:
                            print("Region selected. Press 'u' again to OCR this region or 'c' to cancel.")
                            k2 = cv2.waitKey(0) & 0xFF
                            if k2 == ord('u'):
                                region = (self.region_start, self.region_end)
                                detected_text, roi_annotated = self.process_ocr_region(frame.copy(), region)
                                if detected_text == '':
                                    print("No valid region detected. Please select a larger or valid region.")
                                    self.region_selected = False
                                    continue
                                print(f"Detected region text: '{detected_text}'")
                                matches = self.compare_texts(detected_text)
                                if matches:
                                    print(f"Matches found: {matches}")
                                    cv2.rectangle(roi_annotated, (0, 0), (roi_annotated.shape[1], roi_annotated.shape[0]), (0, 255, 0), 10)
                                else:
                                    print("No match found")
                                    cv2.rectangle(roi_annotated, (0, 0), (roi_annotated.shape[1], roi_annotated.shape[0]), (0, 0, 255), 10)
                                cv2.imshow('OCR Result', roi_annotated)
                                cv2.waitKey(0)
                                self.region_selected = False
                                break
                            elif k2 == ord('c'):
                                print("Region selection cancelled.")
                                self.region_selected = False
                                break
                        elif k == ord('q'):
                            return
                        elif k == ord('c'):
                            print("Region selection cancelled.")
                            self.region_selected = False
                            break
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

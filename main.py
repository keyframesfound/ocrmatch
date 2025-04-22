import cv2
from easyocr import Reader
import numpy as np
from flask import Flask

app = Flask(__name__)

# Predefined OCR text
text_data = {
    "text_1": "This is some sample text.",
    "text_2": "Another example of text.",
}

def setup_webcam():
    global cap
    try:
        cap = cv2.VideoCapture(0)
        return cap
    except Exception as e:
        print(f"Error setting up webcam: {e}")

def process_ocr(frame):
    try:
        reader = Reader(['en'])
        text, _ = reader.readtext(frame)
        return '\n'.join(text)
    except Exception as e:
        print(f"Error performing OCR: {e}")
        return None

def compare_texts(text1, text2):
    if text1 is None or text2 is None:
        return False

    text1_lines = [line.strip() for line in text1.split('\n')]
    text2_lines = [line.strip() for line in text2.split('\n')]

    # Check for exact matches
    if set(text1_lines) == set(text2_lines):
        return True

    # Check for partial matches
    for line in text1_lines:
        if line in text2_lines:
            return True

    return False

def display_frame(cap, frame):
    cv2.imshow('Webcam Feed', frame)

def main():
    setup_webcam()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame(cap, frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('o'):
            text = process_ocr(frame)
            data_texts = list(text_data.items())

            match_found = False
            for i, (text_i, text_j) in enumerate(data_texts):
                if compare_texts(text, text_i[0]):
                    print(f"Match found at line {i+1}: '{text_i[0]}'")
                    match_found = True
                    break
            
            if not match_found:
                print("No match found")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import imutils
import pytesseract
import numpy as np
import pandas as pd
import time
import os
import threading

# Add the Tesseract directory to the system PATH
os.environ['PATH'] += r';C:\Program Files\Tesseract-OCR'

# Specify the Tesseract executable path directly
custom_tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to set Tesseract path for the current Jupyter Notebook session
def set_tesseract_path():
    pytesseract.pytesseract.tesseract_cmd = custom_tesseract_path

# Set Tesseract path
set_tesseract_path()

# Global variable to store the captured image
captured_image = None

# Function to capture an image from the camera
def capture_image():
    global captured_image
    camera = cv2.VideoCapture(0)
    
    # Check if the camera is opened successfully
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        # Read a frame from the camera
        ret, frame = camera.read()

        # Check if the frame is read successfully
        if not ret:
            print("Error: Could not read frame.")
            return

        # Store the captured image
        captured_image = frame

    finally:
        # Clean up resources
        camera.release()

# Function to process the captured image
def process_image():
    global captured_image

    # Check if an image is available
    if captured_image is not None:
        # Perform image processing and OCR here
        # ...

        # Reset captured image after processing
        captured_image = None

# Function to periodically capture and process images
def image_processing_loop():
    while True:
        capture_image()
        process_image()
        time.sleep(1)  # Adjust sleep time as needed

# Start the image processing loop in a separate thread
image_processing_thread = threading.Thread(target=image_processing_loop)
image_processing_thread.start()

# Open a connection to the camera
camera = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

try:
    while True:
        # Read a frame from the camera
        ret, frame = camera.read()

        # Check if the frame is read successfully
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the live camera feed
        cv2.imshow("Camera Feed", frame)

        # Break the loop and close the camera feed window if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Clean up resources
    cleanup()
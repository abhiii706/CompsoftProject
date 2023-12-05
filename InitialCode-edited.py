import cv2
import imutils
import pytesseract
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt


# Add the Tesseract directory to the system PATH
os.environ['PATH'] += r';C:\Program Files\Tesseract-OCR'

# Full path to the image file using a raw string
image_path = r'C:\Users\Abhinav\AppData\Local\Temp\16e464e6-82e8-42e5-9071-435bc8ee336b_new.zip.36b\new\images\1.jpg'

# Specify the Tesseract executable path directly
custom_tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to set Tesseract path for the current Jupyter Notebook session
def set_tesseract_path():
    pytesseract.pytesseract.tesseract_cmd = custom_tesseract_path

# Set Tesseract path
set_tesseract_path()
 
# Read the image
img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Check if the image is successfully loaded
if img is not None:
    img = imutils.resize(img, width=500)

    # Display the original image using Matplotlib
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a bilateral filter to reduce noise
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Display the grayscale image
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    plt.show()

    # Use Canny edge detector to find edges
    edged = cv2.Canny(gray, 170, 200)

    # Display the edged image
    plt.imshow(edged, cmap='gray')
    plt.axis('off')
    plt.show()

    # Find contours in the edged image
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Check if contours were found
    if cnts is not None and len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
        NumberPlateCnt = None

        # Iterate through the contours and find the one with four vertices
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                NumberPlateCnt = approx
                break

        # Masking the part other than the number plate
        if NumberPlateCnt is not None:
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
            new_image = cv2.bitwise_and(img, img, mask=mask)

            # Display the final image with the extracted number plate
            plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

            # Configuration for Tesseract OCR
            config = ('-l eng --oem 1 --psm 3')

            # Run Tesseract OCR on the extracted number plate region
            text = str(pytesseract.image_to_string(new_image, config=config))

            # Data is stored in a CSV file
            raw_data = {'date': [time.asctime(time.localtime(time.time()))],
                        'v_number': [text]}

            df = pd.DataFrame(raw_data, columns=['date', 'v_number'])
            
            # Save data to CSV file
            df.to_csv('data.csv', index=False)

            # Print recognized text
            print("Recognized Text:", text)

        else:
            print("No contours with four vertices (number plate) found.")

    else:
        print("No contours found.")

else:
    print(f"Failed to load the image at path: {image_path}")
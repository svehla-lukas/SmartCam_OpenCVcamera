# Imports
import cv2
import numpy as np
import time
from scipy.spatial import distance
from collections import defaultdict

from shapeDetect import shapeDetection
from HandTrackingDraw import HandTrackingDetection
from TextDetection import extractTextFromImage
from GetRectanglePicture import GetRectanglePicture
from utils import initCamera

# Global constants (optional)


# Main function
def main() -> None:
    # shapeDetection()
    # HandTrackingDetection()
    # extractTextFromFrame()

    user = ""
    while user != "q":
        user = input("Press Enter to continue...\n Press q to quit\n")
        if user == "q":
            break
        # Start the camera and process the frames
        print("\nStarting camera. Press 'q' to quit.")
        # Initialize the camera
        cap = initCamera()

        processed_frame = GetRectanglePicture(cap)

        # Check if a frame was returned
        if processed_frame is not None:
            print("\nProcessed frame received. Further processing can be done here.")
            # Example: Save the frame as an image
            # Processed frame saving and text extraction
            cv2.imwrite(
                "processed_frame.jpg", processed_frame
            )  # Save the processed frame
            print("\nFrame saved as 'processed_frame.jpg'.")
            text = extractTextFromImage(
                "processed_frame.jpg", "eng+ces"
            )  # Pass the file path
            # print("\nExtracted text:\n", text)

        else:
            print("\nno Text")


# Classes (optional)

# Main function
if __name__ == "__main__":
    main()

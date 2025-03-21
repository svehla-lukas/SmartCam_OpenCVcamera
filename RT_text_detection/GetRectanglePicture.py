import cv2
from cv2.typing import NumPyArrayNumeric
import numpy as np
from numpy import ndarray
import time
from utils import preprocessFrame


def rotateRectangleWideDown(frame, drawContour=False):
    """
    Detects rectangles in the frame, rotates them to ensure the wide side is at the bottom,
    and returns the cropped rectangle and center point.

    Parameters:
        frame (numpy.ndarray): The input frame from the camera.
        drawContour (bool): Whether to draw the detected rectangle on the frame.

    Returns:
        tuple: (cropped rectangle, center point of the rectangle or None if no rectangle is found)
    """

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 150, 250)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow("blurred", edges)

    for contour in contours:
        # Approximate contour to a polygon
        epsilon = 0.01 * cv2.arcLength(
            contour, True
        )  # Reduce epsilon for finer approximation
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(approx)
        # Check if the polygon is a rectangle (4 corners)
        if 3 < len(approx) < 1000 and area > 5000:
            # Get the bounding rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Determine the width and height of the rectangle
            width, height = rect[1]
            angle = rect[2]
            # Check the ratio of the sides
            ratio = max(width, height) / min(width, height)
            if ratio < 1.2 or ratio > 5.0:
                continue  # Skip contours with unrealistic aspect ratios

            # Check if the rectangle needs rotation
            if angle > 50:
                angle = angle - 90

            # Rotate the image around the rectangle center
            center = tuple(map(int, rect[0]))
            rotation_matrix: (
                cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]]
            ) = cv2.getRotationMatrix2D(center, angle, 1)

            rotated_frame = cv2.warpAffine(
                frame, rotation_matrix, (frame.shape[1], frame.shape[0])
            )

            # Crop the rectangle from the rotated image
            x, y, w, h = cv2.boundingRect(array=box)
            cropped_rectangle = rotated_frame[y : y + h, x : x + w]

            # Resize the cropped rectangle to larger dimensions
            scale_factor = 1.8  # Example: Double the size
            cropped_rectangle = cv2.resize(
                cropped_rectangle, None, fx=scale_factor, fy=scale_factor
            )

            # Draw the rectangle (optional for debugging)
            if drawContour:
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

            return cropped_rectangle, center

    return None, None


def GetRectanglePicture(cap=cv2.VideoCapture(0)):
    """
    Main function to start the camera, detect rectangles, rotate them, and save the cropped rectangle
    only if the rectangle center remains consistent for 1 second.
    """

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    stable_center = None
    stable_start_time = None

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read a frame from the camera.")
            break

        # Process the frame to detect and rotate rectangles
        frame = preprocessFrame(frame)
        cropped_rectangle, center = rotateRectangleWideDown(frame, drawContour=True)

        # Check if the center is stable
        if center is not None:
            if stable_center == center:
                if stable_start_time is None:
                    stable_start_time = time.time()
                elif time.time() - stable_start_time >= 0.5:
                    # Save the cropped rectangle as a JPG image
                    cv2.imwrite("detected_rectangle.jpg", cropped_rectangle)
                    print("Cropped rectangle saved as 'detected_rectangle.jpg'.")
                    print("\n - Press key to continue")

                    if cv2.waitKey(0) != -1:
                        print()

                    cap.release()
                    cv2.destroyAllWindows()
                    return cropped_rectangle
            else:
                stable_center = center
                stable_start_time = None
        else:
            stable_center = None
            stable_start_time = None

        cv2.imshow("Rectangle Rotation", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    return None

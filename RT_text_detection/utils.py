import cv2
import numpy as np


def preprocessFrame(frame):
    """
    Preprocesses the frame by flipping and optionally rotating.
    """
    # Flip frame horizontally (avoids mirror effect)
    # frame = cv2.flip(frame, 1)

    # Rotate frame 90 degrees clockwise
    frame = cropFrameCv2(frame)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def cropFrameCv2(
    frame,
    vertical_start_ratio=0,
    vertical_end_ratio=1,
    horizontal_start_ratio=1 / 3,
    horizontal_end_ratio=2 / 3,
):
    """
    Crop the input frame based on the specified ratios.

    Parameters:
        frame (numpy.ndarray): Input image frame.
        vertical_start_ratio (float): Starting ratio for vertical cropping (0 to 1).
        vertical_end_ratio (float): Ending ratio for vertical cropping (0 to 1).
        horizontal_start_ratio (float): Starting ratio for horizontal cropping (0 to 1).
        horizontal_end_ratio (float): Ending ratio for horizontal cropping (0 to 1).

    Returns:
        numpy.ndarray: Cropped image frame.
    """
    height, width, _ = frame.shape

    # Calculate crop boundaries
    start_y = int(height * vertical_start_ratio)
    end_y = int(height * vertical_end_ratio)
    start_x = int(width * horizontal_start_ratio)
    end_x = int(width * horizontal_end_ratio)

    # Ensure crop boundaries are valid
    if start_y >= end_y or start_x >= end_x:
        print("Warning: Invalid crop boundaries. Returning the original frame.")
        return frame

    # Crop the frame
    cropped_frame = frame[start_y:end_y, start_x:end_x]
    return cropped_frame


def initCamera():
    cap = cv2.VideoCapture(0)

    # Set the resolution to 1080p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Width: 1920 pixels
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Height: 1080 pixels

    # Optionally set the frame rate to 30 FPS
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Verify settings
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(
        f"Camera initializa\nResolution: {int(width)}x{int(height)} at {int(fps)} FPS"
    )

    return cap

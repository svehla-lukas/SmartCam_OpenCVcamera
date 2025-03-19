# Imports
import cv2
import numpy as np
from scipy.spatial import distance
from collections import defaultdict

# Global constants (optional)

# Initialize global variables
prevCenter = [0, 0]
# Flags
flagDetectObjet = True
flagIncrease = False
# counter
counterRectangles = 0
counterSquare = 0
counterCircle = 0
counterDetectObject = 0

# Classes (optional)


# Main function
def shapeDetection() -> None:
    """Main function to capture and process video frames."""
    global flagDetectObjet, counterSquare, counterRectangles, flagIncrease, counterCircle

    delayFrame = 200
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if frame capturing fails

        frame = preprocessFrame(frame)

        # Convert frame to grayscale and apply preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(image=thresh, threshold1=40, threshold2=150)

        if flagDetectObjet:
            processEdges(edges, frame)

            # Display text with detected object counts
            text = f"Rectangle: {counterRectangles}\nSquare: {counterSquare}\nCircles: {counterCircle}"
            multiLineTextCv2(frame, text, font_scale=0.7)

            cv2.imshow("Processed Frame", frame)

        adjustDelay()

        if cv2.waitKey(delayFrame) & 0xFF == ord("q"):  # Exit on 'q'
            break

    cleanup(cap)


def preprocessFrame(frame):
    """Crop, flip, and preprocess the frame."""
    frame = cropFrameCv2(frame)
    return cv2.flip(frame, flipCode=1)


def processEdges(edges, frame):
    """Process edges to detect shapes and display debug information."""
    rgb_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    detectRectanglesAngle(edges, frame)
    # detectRectanglesAngle(edges, rgb_edges)
    cv2.imshow("Edges Debug", rgb_edges)


def adjustDelay():
    """Adjust frame delay based on the flagIncrease flag."""
    global delayFrame, flagIncrease
    if flagIncrease:
        delayFrame = 2000
        flagIncrease = False
    else:
        delayFrame = 200


def cleanup(cap):
    """Release resources and close windows."""
    cap.release()
    cv2.destroyAllWindows()


def multiLineTextCv2(
    image,
    text,
    start_x=10,
    start_y=30,
    line_spacing=20,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.5,
    color=(0, 0, 0),
    thickness=2,
):
    """
    Draws multi-line text on an image.

    Args:
        image: The image where the text will be drawn.
        text: The multi-line text to be displayed (lines separated by '\n').
        start_x: The X-coordinate for the starting position of the text.
        start_y: The Y-coordinate for the starting position of the text.
        line_spacing: Vertical spacing between lines of text.
        font: The font type for the text (default: cv2.FONT_HERSHEY_SIMPLEX).
        font_scale: The size of the text (default: 0.5).
        color: The color of the text in BGR format (default: white).
        thickness: The thickness of the text lines (default: 2).
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (start_x, start_y + i * line_spacing),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


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
        debug (bool): If True, prints debug information.

    Returns:
        numpy.ndarray: Cropped image frame.
    """
    height, width, _ = frame.shape

    # Calculate crop boundaries
    start_y = int(height * vertical_start_ratio)
    end_y = int(height * vertical_end_ratio)
    start_x = int(width * horizontal_start_ratio)
    end_x = int(width * horizontal_end_ratio)

    # Crop the frame
    cropped_frame = frame[start_y:end_y, start_x:end_x]
    return cropped_frame


# Functions


def detectRectanglesAngle(image, drawOnFrame):
    global prevCenter, counterDetectObject, flagIncrease, counterRectangles, counterSquare, counterCircle

    COLORS = {
        "square": (0, 255, 0),  # Green
        "rectangle": (255, 0, 0),  # Blue
        "circle": (0, 255, 255),  # Yellow
    }

    # Get contours from the image
    contours = contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Process each contour
    for contour in contours:
        # Classify shape and get details
        shape, label, color, angle, center = classifyShape(contour, COLORS)
        if label:
            drawShape(drawOnFrame, shape, label, color, angle, center)
            handleDetection(label, angle, center, drawOnFrame)


def classifyShape(contour, colors):
    """Classify the shape based on contour properties."""
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    area = cv2.contourArea(contour)

    if len(approx) == 4 and area > 100:  # Rectangle or Square
        return classifyRectangleOrSquare(contour, approx, colors)

    if area > 100 and perimeter > 0:  # Circle detection
        return classifyCircle(contour, perimeter, area, colors)

    return None, None, None, None, None


def classifyRectangleOrSquare(contour, approx, colors):
    """Classify whether the shape is a rectangle or a square."""
    x, y, w, h = cv2.boundingRect(approx)
    aspectRatio = float(w) / h
    label = "Square" if 0.95 < aspectRatio < 1.05 else "Rectangle"
    color = colors["square"] if label == "Square" else colors["rectangle"]

    rect = cv2.minAreaRect(contour)
    center, angle = rect[0], rect[2]
    if rect[1][0] < rect[1][1]:
        angle = 90 + angle

    return rect, label, color, angle, center


def classifyCircle(contour, perimeter, area, colors):
    """Classify whether the shape is a circle."""
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    if 0.8 < circularity <= 1.2:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        if radius > 10:
            label = "Circle"
            return (center, radius), label, colors["circle"], 0, center

    return None, None, None, None, None


def drawShape(frame, shape, label, color, angle, center):
    """Draw the shape on the frame."""
    if label in ["Square", "Rectangle"]:
        # Check if shape is valid before using cv2.boxPoints
        if shape is not None and len(shape) > 0:
            box = cv2.boxPoints(shape)
            box = np.int32(box)  # Convert to integer points
            cv2.drawContours(frame, [box], 0, color, 3)

            # Display the rotation angle near the shape
            if center is not None and len(center) == 2:
                cv2.putText(
                    frame,
                    f"Angle: {normalizeAngle(angle):.1f}deg",
                    (int(box[3][0]), int(box[3][1])),
                    # (int(center[0] + 10), int(center[1] - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 150, 58),
                    2,
                    cv2.LINE_AA,
                )
        else:
            print("Invalid shape for drawing rectangle or square.")
    elif label == "Circle":
        # Validate that shape contains center and radius
        if isinstance(shape, tuple) and len(shape) == 2:
            center, radius = shape
            cv2.circle(frame, center, radius, color, 3)
        else:
            print("Invalid shape for drawing circle.")

    # Add label to the frame
    if center is not None and len(center) == 2:
        cv2.putText(
            frame,
            label,
            (int(center[0] - 50), int(center[1] - 50)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    else:
        print("Invalid center coordinates for text placement.")


def handleDetection(label, angle, center, frame):
    """Handle detection logic, including counters and messages."""
    global prevCenter, counterDetectObject, flagIncrease, counterRectangles, counterSquare, counterCircle

    tolerance = 5
    if (
        abs(prevCenter[0] - center[0]) > tolerance
        or abs(prevCenter[1] - center[1]) > tolerance
    ):
        counterDetectObject += 1
        if counterDetectObject > 9:
            if label == "Circle":
                counterCircle += 1
            elif label == "Square":
                counterSquare += 1
            elif label == "Rectangle":
                counterRectangles += 1

            counterDetectObject = 0
            prevCenter = center
            flagIncrease = True

            multiLineTextCv2(
                frame,
                "CHECK",
                int(center[0]),
                int(center[1]),
                line_spacing=1,
                color=(0, 255, 255),
            )


def normalizeAngle(angle):
    angle %= 360

    if angle < 45:
        return angle
    elif 45 <= angle < 90:
        return 90 - angle
    elif 90 <= angle < 135:
        return angle - 90
    elif 135 <= angle < 180:
        return 180 - angle
    elif 180 <= angle < 225:
        return angle - 180
    elif 225 <= angle < 270:
        return 270 - angle
    elif 270 <= angle < 315:
        return angle - 270
    else:  # 315 <= angle < 360
        return 360 - angle

import cv2
import pytesseract

# import numpy as np


def cleanDetectedText(detectedText):
    """
    Cleans the detected text by removing excessive spaces and empty lines.

    Parameters:
        detectedText (str): Raw text detected by OCR.

    Returns:
        str: Cleaned text.
    """
    # Split text into lines
    lines = detectedText.splitlines()
    # Remove empty lines and strip excessive spaces
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    # Join the cleaned lines into a single text
    cleaned_text = "\n".join(cleaned_lines)
    return cleaned_text


# def origin_extractTextFromImage(imagePath):
def extractTextFromImage(imagePath, language="eng"):
    """
    Extracts text from a given image file using Tesseract OCR and saves the text to a .txt file.

    Parameters:
        imagePath (str): The file path to the input image.

    Returns:
        str: Detected text in the image.
    """
    next = True
    threst = 140
    maxVal = 200
    # Load the image
    image = cv2.imread(imagePath)
    if image is None:
        raise ValueError(f"Cannot load image from path: {imagePath}")

    # Preprocessing the image
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, thresholdImage = cv2.threshold(
        grayImage, threst, maxVal, cv2.THRESH_BINARY
    )  # Binarization

    # cv2.imshow("gray", grayImage)
    cv2.imshow("threshold", thresholdImage)

    # Text detection
    # lang="ces"
    detectedText = pytesseract.image_to_string(thresholdImage, language)
    detectedText = cleanDetectedText(detectedText)
    # Save the detected text to a .txt file
    textFilePath = imagePath.rsplit(".", 1)[0] + "_output.txt"
    with open(textFilePath, "w", encoding="utf-8") as textFile:
        textFile.write(detectedText)
    # print("\n\n")
    # print(detectedText)
    # print(f"threst: {threst}\nmaxVal: {maxVal}")
    # print("\n\n")
    next = False
    if cv2.waitKey(0) != -1:  # Wait for a key press

        threst += 20  # Increment 'threst' by 20
        if threst > 250:
            next = False  # Assuming 'next' is a control variable

    cv2.destroyAllWindows()  # Close all OpenCV windows

    print(f"Detected text saved to: {textFilePath}")
    return detectedText

    # Setting the path to Tesseract OCR (if needed)
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    )


# def extractTextFromImage(imagePath):
def new_extractTextFromImage(imagePath):
    """
    Extracts text from a given image file using Tesseract OCR and saves the text to a .txt file.

    Parameters:
        imagePath (str): The file path to the input image.

    Returns:
        str: Detected text in the image.
    """
    # Load the image
    image = cv2.imread(imagePath)
    if image is None:
        raise ValueError(f"Cannot load image from path: {imagePath}")

    # Preprocess the image
    processed_image = preprocessImage(image)

    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Text detection
    detected_text = pytesseract.image_to_string(processed_image, lang="eng")
    detected_text = detectedText(detected_text)
    # Save the detected text to a .txt file
    textFilePath = imagePath.rsplit(".", 1)[0] + "_output.txt"
    with open(textFilePath, "w", encoding="utf-8") as textFile:
        textFile.write(detected_text)

    print(f"Detected text saved to: {textFilePath}")
    return detected_text


def extractTextFromFrame(frame):
    """
    Extracts text from a given frame using Tesseract OCR.

    Parameters:
        frame (numpy.ndarray): The input image frame.

    Returns:
        str: Detected text in the frame.
    """
    # Preprocessing the image
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, thresholdFrame = cv2.threshold(
        grayFrame, 150, 255, cv2.THRESH_BINARY
    )  # Binarization

    # Text detection
    detectedText = pytesseract.image_to_string(
        thresholdFrame, lang="eng"
    )  # 'lang' specifies the language
    return detectedText


# Setting the path to Tesseract OCR (if needed)
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)


def preprocessImage(frame):
    """
    Preprocesses the image to enhance text recognition for OCR.

    Parameters:
        frame (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Processed image ready for OCR.
    """
    # Convert to grayscale
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    equalized_image = cv2.equalizeHist(grayFrame)

    # Reduce noise
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

    # Adaptive thresholding for binarization
    processed_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return processed_image


# Setting the path to Tesseract OCR (if needed)
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)

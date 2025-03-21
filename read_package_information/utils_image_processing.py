import cv2
import numpy as np
import pytesseract
from typing import Tuple
from difflib import SequenceMatcher


def detect_edges_and_contours(
    frame_gray: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Processes a grayscale image to detect edges and extract contours.

    Steps performed:
    1. Applies Gaussian blur to reduce noise and smooth the image.
    2. Uses thresholding (binary + adaptive) to enhance edges.
    3. Detects edges using the Canny edge detector.
    4. Finds contours using cv2.findContours() with hierarchical data.

    :param frame_gray: np.ndarray
        Grayscale image input (2D NumPy array).

    :return: tuple[list[np.ndarray], np.ndarray]
        - List of contours (each contour is an array of points).
        - Hierarchy array describing the relationships between contours.
    """
    # 1. Rozmazání obrazu pomocí Gaussova filtru
    #    - Kernel velikosti (5,5) určuje rozsah rozmazání
    #    - SigmaX = 0 znamená, že hodnota se vypočítá automaticky na základě jádra
    blurred = cv2.GaussianBlur(frame_gray, (17, 17), 0)

    # 2.1 Prahování (Thresholding)
    #    - Prahová hodnota: 40 (vše pod touto hodnotou bude černé)
    #    - Maximální hodnota: 250 (vše nad touto hodnotou bude bílé)
    #    - Použitý režim: cv2.THRESH_BINARY (binární prahování)
    _, threshold = cv2.threshold(blurred, 20, 250, cv2.THRESH_BINARY)

    # 2.2 Adaptivní prahování (Adaptive Thresholding)
    #    - Dynamicky nastavuje prahovou hodnotu pro různé oblasti obrazu.
    #    - Použitá metoda: cv2.ADAPTIVE_THRESH_GAUSSIAN_C (používá vážený průměr okolních pixelů s Gaussovským filtrem).
    #    - Typ prahování: cv2.THRESH_BINARY (pixely nad vypočítanou prahovou hodnotou se stanou bílé, ostatní černé).
    #    - Bloková velikost: 11 (velikost oblasti pro výpočet prahu, musí být liché číslo).
    #    - Konstanta C: 2 (hodnota odečtená od vypočítaného prahu pro jemnější úpravy).
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 3. Detekce hran pomocí Canny edge detection
    #    - Dolní práh: 20 (nižší hodnota znamená citlivější detekci hran)
    #    - Horní práh: 250 (vyšší hodnota znamená přísnější detekci hran)
    edges = cv2.Canny(threshold, 20, 500)

    # 4. Hledání kontur
    #    - Metoda: cv2.RETR_TREE (zachování hierarchie kontur)
    #    - Přesnost: cv2.CHAIN_APPROX_SIMPLE (zjednodušení kontur odstraněním zbytečných bodů)
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # cv2.imshow("adaptive_thresh", cv2.resize(adaptive_thresh, (640, 480)))
    # cv2.imshow("threshold", cv2.resize(threshold, (640, 480)))
    # cv2.imshow("threshold", cv2.resize(threshold, (640, 480)))
    # cv2.imshow("edges", cv2.resize(edges, (640, 480)))

    return contours, hierarchy


def get_biggest_polygon(frame_gray: np.ndarray, px_to_mm: float, pixelsArea=200000):
    """
    Finds the largest polygon in the image and returns its center coordinates and rotation angle.

    :param frame_gray: Input grayscale image.
    :param pixelsArea: Minimum polygon area for detection.
    :return: Processed image, center coordinates (x, y) or None if no polygon was found, rotation angle.
    """

    largest_contour = None
    crop_frame = None
    center_y, center_x, angle = None, None, None

    frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    contours, _ = detect_edges_and_contours(frame_gray)
    if not contours:
        print("noneContours")
        return frame_bgr, crop_frame, None, None, None  # No contours found

    largest_contour = max(contours, key=cv2.contourArea, default=None)

    if cv2.contourArea(largest_contour) > pixelsArea:
        # Počítá obvod (perimeter) kontury.
        epsilon = 0.07 * cv2.arcLength(largest_contour, True)
        # Zjednodušuje tvar kontury tím, že odstraní zbytečné body.
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4:  # Ověření, že máme 4-úhelník
            cv2.drawContours(
                frame_bgr, [approx], -1, (100, 255, 100), 2
            )  # Zelený obrys

            # 1. Získání úhlu rotace pomocí `cv2.minAreaRect()`
            rect = cv2.minAreaRect(largest_contour)
            # Oříznutí podle obdélníku
            box = cv2.boxPoints(rect)
            crop_frame = crop_image(frame_gray, box)

            ######################################################
            #  --- Vlozeni textu na puvodni obrazek ---
            (center_x, center_y), (width, height), angle = rect
            center_x, center_y = int(center_x), int(center_y)
            width, height = int(width), int(height)
            # Převod na mm pomocí konstanty px_to_mm
            center_x_mm, center_y_mm = center_x * px_to_mm, center_y * px_to_mm
            width_mm, height_mm = width * px_to_mm, height * px_to_mm
            angle = round(angle, 2)  # Zaokrouhlení na 2 desetinná místa

            # Převod na celá čísla pro lepší čitelnost (u rozměrů)
            center_x_mm, center_y_mm = int(center_x_mm), int(center_y_mm)
            width_mm, height_mm = int(width_mm), int(height_mm)

            # Definování textových popisků
            labels = [
                f"center: x:{center_x}, y:{center_y} [px]",
                f"dimension: w:{width} x h:{height} [px]",
                f"rotation: {angle}deg",
            ]
            labels_mm = [
                f"center: x:{center_x_mm}, y:{center_y_mm} [mm]",
                f"dimension: w:{width_mm} x h:{height_mm} [mm]",
                f"rotation: {angle}deg",
            ]

            # Korekce úhlu pro lepší interpretaci
            if angle < -45:
                angle += 90  # Oprava OpenCV úhlu pro správnou interpretaci

            # Parametry textu
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 2.5
            color = (255, 255, 255)
            thickness = 4
            offset_y = 100  # Počáteční vertikální posun

            # Vykreslení textu
            for i, text in enumerate(labels_mm):
                cv2.putText(
                    frame_bgr,
                    text,
                    (10, offset_y + (i * 100)),
                    font,
                    scale,
                    color,
                    thickness,
                )

            # Červený bod středu
            cv2.circle(frame_bgr, (center_x, center_y), 5, (0, 0, 255), -1)
            ######################################################

            return frame_bgr, crop_frame, center_x, center_y, angle

    return frame_bgr, crop_frame, None, None, None


def crop_image(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Extracts and warps a rotated rectangle from the image, ensuring the bottom side is always longer.

    :param frame: Original image
    :param box: 4 corner points of the rotated rectangle
    :return: Cropped and straightened region
    """
    #     box = [
    #     [200, 50],   # Horní levý
    #     [300, 50],   # Horní pravý
    #     [250, 150],  # Dolní levý
    #     [350, 150]   # Dolní pravý
    # ]
    box = np.array(sorted(box, key=lambda x: x[1]))
    # box = np.array([
    #     [300, 50],   # Nejvyšší bod (nejmenší Y)
    #     [200, 100],  # Druhý nejvyšší bod
    #     [250, 150],  # Třetí nejvyšší bod
    #     [100, 200]   # Nejnižší bod (největší Y)
    # ])

    # Split on upper and lower part
    if box[0][0] > box[1][0]:
        top_right, top_left = box[0], box[1]
    else:
        top_left, top_right = box[0], box[1]

    if box[2][0] > box[3][0]:
        bottom_right, bottom_left = box[2], box[3]
    else:
        bottom_left, bottom_right = box[2], box[3]

    width = int(np.linalg.norm(top_right - top_left))
    height = int(np.linalg.norm(bottom_left - top_left))

    # If the height is greater than the width, swap the axes (rotate by 90°)
    if height > width:
        top_left, top_right, bottom_right, bottom_left = (
            bottom_left,
            top_left,
            top_right,
            bottom_right,
        )
        width, height = height, width  # Swap width and height

    # Destination points for rectification
    dst_pts = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype="float32"
    )

    # Compute the transformation matrix
    src_pts = np.array(
        [top_left, top_right, bottom_right, bottom_left], dtype="float32"
    )
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Perspective transformation to straighten the rectangle
    cropped_frame = cv2.warpPerspective(frame, matrix, (width, height))

    return cropped_frame


def calculate_pixel_size(
    D: float, W_sensor: float = 5.76, f: float = 8, W_res: int = 2592
) -> float:
    """
    Calculates the size of a single pixel in millimeters for a given camera-to-object distance (D).

    :param D: Distance between the camera and the object (mm). Must be > 0.
    :param W_sensor: Sensor width (mm), default 5.76 mm.
    :param f: Lens focal length (mm), default 8 mm.
    :param W_res: Sensor resolution width (px), default 2592 px. Must be > 0.
    :return: Size of a single pixel in millimeters.

    :raises ValueError: If D <= 0 or W_res <= 0.
    """

    # Validate input values
    if D <= 0:
        raise ValueError("❌ Distance D must be greater than zero.")
    if W_res <= 0:
        raise ValueError("❌ Sensor resolution W_res must be greater than zero.")

    # Calculate the horizontal field of view angle (in radians)
    fov_angle = 2 * np.arctan(W_sensor / (2 * f))

    # Compute the field of view width at distance D
    fov_width = 2 * D * np.tan(fov_angle / 2)

    # Compute pixel size in millimeters
    pixel_size = fov_width / W_res

    return pixel_size


def extract_text_from_frame(
    frame: cv2.typing.MatLike,
    position_px: Tuple[int, int, int, int],
    language: str = "eng",
    show_image: bool = False,
) -> str:
    """
    Extracts text from a specific region in the given frame using Tesseract OCR.

    Args:
        frame (cv2.typing.MatLike): Input image/frame from which text will be extracted.
        position_px (Tuple[int, int, int, int]): Coordinates (x, y, width, height) of the text region.
        language (str, optional): Language(s) for OCR processing (e.g., "eng+ces"). Defaults to "eng".
        show_image (bool, optional): If True, displays the extracted text region using OpenCV. Defaults to False.

    Returns:
        str: Extracted and cleaned text from the image.

    Raises:
        ValueError: If the extracted region is empty or invalid.
        TypeError: If the language argument is not a string.
    """
    # Validate language input
    if not isinstance(language, str):
        raise TypeError(
            "❌ The 'language' parameter must be a string (e.g., 'eng' or 'eng+ces')."
        )

    # Extract the region of interest (ROI) from the image
    x, y, w, h = position_px
    frame_text_code = frame[y : y + h, x : x + w]

    if frame_text_code is None or frame_text_code.size == 0:
        raise ValueError(
            "❌ Failed to extract the text region. The selected area is empty."
        )

    # Display the extracted region if enabled
    if show_image:
        cv2.imshow("Extracted Text Frame", frame_text_code)
    # Perform OCR using Tesseract
    custom_config = r"--oem 3 --psm 4"
    image_text = pytesseract.image_to_string(
        image=frame_text_code, lang=language, config=custom_config
    ).strip()

    # Clean output (replace line breaks with spaces and remove excess spaces)
    image_text = " ".join(image_text.split())

    return image_text


def text_similarity(text1: str, text2: str) -> float:
    """
    Computes the similarity percentage between two text strings after normalizing them.

    :param text1: First text string.
    :param text2: Second text string.
    :return: Similarity percentage (0 to 100).
    """

    def normalize(text: str) -> str:
        """Removes spaces for better text comparison."""
        return text.replace(" ", "").replace("-", "")
        # return text.replace(" ", "").replace("-", "")

    text1, text2 = normalize(text1), normalize(text2)
    similarity = SequenceMatcher(None, text1, text2).ratio() * 100
    return round(similarity, 2)  # Round for better readability


def find_first_black_pixel(binary_img: np.ndarray) -> tuple[int, int] | None:
    """
    Finds the first black pixel (text) in a binary image, starting from the top-left corner.

    :param binary_img: Binary image as a NumPy array.
    :return: Coordinates (x, y) of the first detected text pixel, or None if no text is found.
    """
    # Ensure input is a valid NumPy array
    if not isinstance(binary_img, np.ndarray):
        raise TypeError("❌ 'binary_img' must be a NumPy array.")

    # Get non-zero (black) pixels
    non_zero_pixels = cv2.findNonZero(
        255 - binary_img
    )  # Invert image for proper detection

    if non_zero_pixels is None:
        return None  # No text detected

    # Get the first detected text pixel (smallest Y, then smallest X)
    first_pixel = tuple(
        non_zero_pixels[0][0]
    )  # OpenCV stores it as [[[x, y]]], so we extract (x, y)

    return first_pixel


def process_and_find_black_pixel(
    frame: np.ndarray, box: np.ndarray
) -> tuple[int, int] | None:
    """
    Applies Gaussian blur, thresholds the image, crops it to the given bounding box,
    and finds the first text pixel.

    :param frame: Input image (NumPy array).
    :param box: Bounding box to crop the image (NumPy array).
    :return: Coordinates (x, y) of the first detected text pixel, or None if no text is found.
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)

    # Convert to binary image (thresholding)
    _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    # Crop the binary image using the provided box
    cropped_frame = crop_image(binary, box)

    # Find the first text pixel
    first_pixel = find_first_black_pixel(cropped_frame)

    return first_pixel


def create_bounding_box(x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Creates a bounding box (4 corner points) from the given top-left corner and dimensions.

    :param x: X-coordinate of the top-left corner.
    :param y: Y-coordinate of the top-left corner.
    :param width: Width of the bounding box.
    :param height: Height of the bounding box.
    :return: NumPy array of shape (4,2) containing the four corner points.
    """
    box = np.array(
        [
            [x, y],  # Top-left
            [x + width, y],  # Top-right
            [x + width, y + height],  # Bottom-right
            [x, y + height],  # Bottom-left
        ],
        dtype="float32",
    )

    return box


def detect_first_black_pixel(crop_frame, search_area):
    """
    Processes the image to detect the first black pixel in a defined region of interest.

    :param crop_frame: Input image (grayscale)
    :param search_area: Tuple (x, y, width, height) defining the region of interest
    :return: Coordinates of the first black pixel or None if not found
    """
    x, y, width, height = search_area

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(crop_frame, (3, 3), 0)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2
    )

    # Apply another binary threshold
    _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    # Show processed images (optional)
    cv2.imshow("size", crop_frame)

    # Define bounding box based on input search area
    box = create_bounding_box(x, y, width, height)

    # Find the first black pixel in the specified area
    first_pixel = process_and_find_black_pixel(crop_frame, box)

    if first_pixel is not None:
        first_pixel += np.array([x, y])
        print(f"✅ First black pixel found at: {first_pixel}")
        return first_pixel
    else:
        print("❌ No black pixel detected in the given area.")
        return None

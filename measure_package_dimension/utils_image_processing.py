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


def find_largest_contour(frame_gray, pixelsArea=200000):
    """
    Finds the largest polygon in the image and returns its contour if it meets the conditions.

    :param frame_gray: Grayscale input image.
    :param pixelsArea: Minimum polygon area for detection.
    :return: The largest valid contour or None if no suitable polygon is found.
    """

    # Detect edges and contours
    contours, _ = detect_edges_and_contours(frame_gray)

    # Check if any contours were found
    if not contours:
        print("❌ No contours were found.")
        return None

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Verify that the largest contour exists and is not too small
    if largest_contour is None or cv2.contourArea(largest_contour) <= pixelsArea:
        print("⚠️ The largest contour is too small or does not exist.")
        return None

    # Simplify the contour (polygon approximation)
    epsilon = 0.07 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Verify that the contour is a quadrilateral
    if len(approx) == 4:
        return largest_contour  # ✅ Valid contour

    print("❌ The largest object is not a quadrilateral.")
    return None  # If the contour is not a quadrilateral


def draw_contour_info(
    frame_bgr, largest_contour, header="Contour Info", focal_length=200, text_offset=100
):
    """
    Processes the largest contour and draws it along with text information.

    Parameters:
        frame_bgr (numpy.ndarray): Color image where the contour will be drawn.
        largest_contour (numpy.ndarray): The largest detected contour.
        focal_length (float): Focal length of the camera.
        text_offset (int): Offset for text positioning on the image.

    Returns:
        tuple: (modified image, center X, center Y, rotation angle)
    """

    # Draw the contour (green color, thickness 2px)
    cv2.drawContours(frame_bgr, [largest_contour], -1, (100, 255, 100), 2)

    # Compute the minimum bounding rectangle around the contour
    (center_x, center_y), (width, height), angle = cv2.minAreaRect(largest_contour)
    center_x, center_y = int(center_x), int(center_y)
    width, height = int(width), int(height)

    # Convert dimensions to millimeters
    px_to_mm = calculate_pixel_size(focal_length)
    width_mm = round(width * px_to_mm, 2)
    height_mm = round(height * px_to_mm, 2)
    area_rect = round(width_mm * height_mm)

    # Adjust angle (OpenCV angles can be misleading)
    if angle < -45:
        angle += 90

    # Text information for display
    texts = [
        f"{header}",
        f"Distance: {focal_length}mm  ; 1mm =~ {round(1/px_to_mm, 3)} [px]",
        f"Center: ({center_x}, {center_y})",
        f"Size: {width} [px] x {height} [px]",
        f"Size: {width_mm} [mm] x {height_mm} [mm]",
        f"Angle: {round(angle, 2)}°",
    ]

    for i, text in enumerate(texts):
        cv2.putText(
            img=frame_bgr,  # Input image
            text=text,  # Text string
            org=(10, text_offset + i * 100),  # Position (x, y)
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # Font type
            fontScale=2,  # Font size
            color=(255, 255, 255),  # Text color (White in BGR)
            thickness=3,  # Thickness of the text
        )

    # Vykreslení středu kontury (červený bod)
    cv2.circle(frame_bgr, (center_x, center_y), 5, (0, 0, 255), -1)

    return frame_bgr

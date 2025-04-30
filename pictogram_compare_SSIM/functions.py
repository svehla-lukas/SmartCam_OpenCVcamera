import cv2
import numpy as np
import time
import os
from skimage.metrics import structural_similarity as ssim


def get_biggest_polygon(frame_gray, pixelsArea=200000):
    """
    Najde největší polygon v obraze, vrátí jeho střední souřadnice a úhel rotace.

    :param frame_gray: Vstupní obraz v odstínech šedi.
    :param pixelsArea: Minimální plocha polygonu pro detekci.
    :return: Upravený obraz, souřadnice středu (x, y) nebo None, pokud nebyl nalezen polygon, úhel rotace.
    """
    largest_contour = None
    crop_frame = None
    center_y, center_x, angle = None, None, None

    frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    contours, _ = detect_edges_and_contours(frame_gray)
    if not contours:
        print("noneContours")
        return frame_bgr, None, None, None  # No contours found

    # Najdi největší konturu podle plochy
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
            # (center_x, center_y), (width, height), angle = cv2.minAreaRect(largest_contour)
            rect = cv2.minAreaRect(largest_contour)
            (center_x, center_y), (width, height), angle = rect

            # Převod na celá čísla pro lepší čitelnost
            center_x, center_y = int(center_x), int(center_y)
            width, height = int(width), int(height)
            angle = round(angle, 2)  # Zaokrouhlení na 2 desetinná místa

            # Definování textových popisků
            text_1 = f"Stred: ({center_x}, {center_y})"
            text_2 = f"Rozmery: {width} x {height}"
            text_3 = f"Uhel: {angle}deg"

            # Korekce úhlu pro lepší interpretaci
            if angle < -45:
                angle += 90  # Oprava OpenCV úhlu pro správnou interpretaci

            absolute_angle = get_absolute_angle(largest_contour)
            text_4 = f"abs Uhel: {int(absolute_angle)}deg"

            # Umístění textu na levý horní roh
            offsetText = 100
            cv2.putText(
                frame_bgr,
                text_1,
                (10, offsetText + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame_bgr,
                text_2,
                (10, offsetText + 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame_bgr,
                text_3,
                (10, offsetText + 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame_bgr,
                text_4,
                (10, offsetText + 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.7,
                (255, 255, 255),
                2,
            )

            # Červený bod středu
            cv2.circle(frame_bgr, (center_x, center_y), 5, (0, 0, 255), -1)

            # Oříznutí podle obdélníku

            box = cv2.boxPoints(rect)

            # Test
            myAngle = 0
            cf = crop_rotated_rectangle(frame_gray, box)
            absolute_angle = angle  # get_absolute_angle(cf)
            if cf is not None:
                _, dir = extract_red_box_and_find_contours(cf)
                if dir == "down":
                    myAngle += 90
                    cf = cv2.rotate(cf, cv2.ROTATE_90_CLOCKWISE)
                    _, dir = extract_red_box_and_find_contours(cf)
                if dir == "down":
                    myAngle += 90
                    cf = cv2.rotate(cf, cv2.ROTATE_90_CLOCKWISE)
                    _, dir = extract_red_box_and_find_contours(cf)
                if dir == "down":
                    myAngle += 90
                    cf = cv2.rotate(cf, cv2.ROTATE_90_CLOCKWISE)
                    _, dir = extract_red_box_and_find_contours(cf)
                if dir == "down":
                    myAngle += 90
                    cf = cv2.rotate(cf, cv2.ROTATE_90_CLOCKWISE)
                    _, dir = extract_red_box_and_find_contours(cf)
                print("myAngle: ", myAngle)

                text_1 = f"abs Uhel: {int(absolute_angle)}deg"
                text_2 = f"myAngle: {int(myAngle)}deg"
                cv2.putText(
                    cf,
                    text_1,
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    cf,
                    text_2,
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("myAngle", resize_frame(cf, 50))
                # END Test

                x_offset = 4.5
                y_offset = -0.5
                hash_map_pictograms = [
                    # x_left_upper_corner, y_left_upper_corner, sizewidth, sizeheight, valid_name
                    [4 + x_offset, 75 + y_offset, 5.5, 9, "LactoseFree.png"],
                    [10 + x_offset, 75 + y_offset, 5.5, 9, "NoClass2Plastics.png"],
                    [16 + x_offset, 75 + y_offset, 5.5, 9, "DoNotReSterilize.png"],
                    [22 + x_offset, 77 + y_offset, 10, 6, "SterileR.png"],
                    [31 + x_offset, 75 + y_offset, 9, 9, "ReadInstructions.png"],
                    [39 + x_offset, 75 + y_offset, 5.5, 9, "ProtecfromSunlight.png"],
                    [45 + x_offset, 75 + y_offset, 5.5, 9, "ProtectFromMoisture.png"],
                    [51 + x_offset, 75 + y_offset, 9, 11, "StoreTemperature.png"],
                    [59 + x_offset, 75 + y_offset, 5.5, 9, "PhthalatFree.png"],
                    [66 + x_offset, 75 + y_offset, 5.5, 9, "LatexFree.png"],
                    [3 + x_offset, 19 + y_offset, 8, 5, "UDI.png"],
                    [3 + x_offset, 26.5 + y_offset, 8, 5, "MD.png"],
                    [46 + x_offset, 18 + y_offset, 6, 5.5, "REF.png"],
                    [46 + x_offset, 23.5 + y_offset, 6, 5.5, "LOT.png"],
                ]

                for pictogram_values in hash_map_pictograms:
                    x, y, size, sizeheight, valid_name = pictogram_values
                    if sizeheight != 0:
                        pictogram = crop_rectangle(cf, x, y, size, sizeheight)
                    else:
                        pictogram = crop_square(cf, x, y, size)
                    cv2.imshow("origin", pictogram)

                    pictogram = convert_to_black_and_white(pictogram)
                    pictogram = remove_white_borders(pictogram)
                    if pictogram is None or pictogram.size == 0:
                        print("❌ Chyba: Oříznutý obrázek `pictogram` je prázdný!")
                    else:
                        #  Compare with images
                        # source_path = r"C:\Users\Uzivatel\Desktop\svehla mvs hkVision\pictureCompare\source"
                        # source_path = r"C:\Users\Uzivatel\Desktop\svehla mvs hkVision\pictogramDetection\correct_source"
                        source_path = os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), "correct_source"
                        )
                        max_score = match_ssim(pictogram, source_path, valid_name)
                        hit = False
                        if max_score > 0.3:
                            hit = True
                        print(f"{hit} : {max_score:.4f} : {valid_name}")

                        cv2.imshow(f"{valid_name}", pictogram)
                        cv2.waitKey(0)
                        try:
                            cv2.destroyWindow(f"{valid_name}")
                            cv2.destroyWindow("origin")
                            cv2.destroyWindow("SSIM Map")
                            cv2.destroyWindow("Input Image 1")
                            cv2.destroyWindow("Input Image 2")
                        except:
                            pass

                # cf = draw_grid(cf, grid_size=20, color=(0, 255, 0), thickness=2)
                # Draw boxes for each pictogram position
                # --- vykreslení boxů pro piktogramy, kde (x,y) = levý-horní roh v procentech ---
                for x_pct, y_pct, w_pct, h_pct, label in hash_map_pictograms:

                    # převod procent → pixely
                    x_px = round(cf.shape[1] * x_pct / 100)
                    y_px = round(cf.shape[0] * y_pct / 100)

                    w_px = round(cf.shape[1] * w_pct / 100)
                    h_px = (
                        round(cf.shape[0] * h_pct / 100) if h_pct else w_px
                    )  # čtverec, když h_pct == 0

                    # horní-levý a spodní-pravý roh
                    top_left = (x_px, y_px)
                    bottom_right = (x_px + w_px, y_px + h_px)

                    # obdélník
                    cv2.rectangle(cf, top_left, bottom_right, (0, 0, 255), 2)

                    # popisek 10 px nad boxem
                    text_org = (x_px, max(0, y_px - 10))
                    cv2.putText(
                        cf,
                        label,
                        text_org,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

            return cf, center_x, center_y, angle

    return frame_bgr, None, None, None


def detect_edges_and_contours(frame_gray):
    """
    Applies Gaussian blur, thresholding, and edge detection to find contours.

    :param frame_gray: Grayscale image input
    :return: List of contours found in the image
    """
    # 1. Rozmazání obrazu pomocí Gaussova filtru
    #    - Kernel velikosti (5,5) určuje rozsah rozmazání
    #    - SigmaX = 0 znamená, že hodnota se vypočítá automaticky na základě jádra
    blurred = cv2.GaussianBlur(frame_gray, (13, 13), 0)

    # 2.1 Prahování (Thresholding)
    #    - Prahová hodnota: 40 (vše pod touto hodnotou bude černé)
    #    - Maximální hodnota: 250 (vše nad touto hodnotou bude bílé)
    #    - Použitý režim: cv2.THRESH_BINARY (binární prahování)
    _, threshold = cv2.threshold(blurred, 65, 250, cv2.THRESH_BINARY)

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
    # edges = cv2.Canny(threshold, 60, 250)

    # 4. Hledání kontur
    #    - Metoda: cv2.RETR_TREE (zachování hierarchie kontur)
    #    - Přesnost: cv2.CHAIN_APPROX_SIMPLE (zjednodušení kontur odstraněním zbytečných bodů)
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # cv2.imshow("adaptive_thresh", resize_frame(adaptive_thresh, 20))
    cv2.imshow("threshold", resize_frame(threshold, 20))
    # cv2.imshow("edges", resize_frame(edges, 20))

    return contours, hierarchy


def draw_grid(frame, grid_size=10, color=(0, 255, 0), thickness=1):
    """
    Přidá na obrázek mřížku o dané velikosti.

    Args:
        frame (numpy.ndarray): Vstupní obrázek (frame).
        grid_size (int): Počet čar (např. 10 = 10x10).
        color (tuple): Barva čar (BGR, výchozí zelená).
        thickness (int): Tloušťka čar.

    Returns:
        numpy.ndarray: Obrázek s mřížkou.
    """
    height, width = frame.shape[:2]

    # 📌 Výpočet vzdáleností mezi čarami
    x_step = width // grid_size
    y_step = height // grid_size

    grid_frame = frame.copy()

    # 📌 Kreslení svislých čar
    for x in range(x_step, width, x_step):
        cv2.line(grid_frame, (x, 0), (x, height), color, thickness)

    # 📌 Kreslení vodorovných čar
    for y in range(y_step, height, y_step):
        cv2.line(grid_frame, (0, y), (width, y), color, thickness)

    return grid_frame


def resize_frame(frame, scale_percent=50):
    # Změna rozlišení
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    new_size = (width, height)
    frame = cv2.resize(frame, new_size)
    return frame


def get_absolute_angle(contour):
    # Najdeme střed a hlavní osy objektu
    mean, eigenvectors = cv2.PCACompute(
        contour.reshape(-1, 2).astype(np.float32), mean=None
    )

    # První vektor odpovídá hlavní ose objektu
    vector = eigenvectors[0]  # Směr hlavní osy
    angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi  # Převod na stupně

    return angle if angle >= 0 else angle + 180  # Normalizace do 0-180°


def crop_rotated_rectangle(frame, box):
    """
    Extracts and warps a rotated rectangle from the image, ensuring the bottom side is always longer.

    :param frame: Original image
    :param box: 4 corner points of the rotated rectangle
    :return: Cropped and straightened region
    """
    # Seřazení bodů podle Y souřadnice (horní a dolní dvojici)
    box = np.array(sorted(box, key=lambda x: x[1]))  # Seřazení podle Y souřadnice
    # box = np.array([
    #     [300, 50],   # Nejvyšší bod (nejmenší Y)
    #     [200, 100],  # Druhý nejvyšší bod
    #     [250, 150],  # Třetí nejvyšší bod
    #     [100, 200]   # Nejnižší bod (největší Y)
    # ])

    # Rozdělení na horní a dolní část
    if box[0][0] > box[1][0]:  # Seřazení levého a pravého bodu nahoře
        top_right, top_left = box[0], box[1]
    else:
        top_left, top_right = box[0], box[1]

    if box[2][0] > box[3][0]:  # Seřazení levého a pravého bodu dole
        bottom_right, bottom_left = box[2], box[3]
    else:
        bottom_left, bottom_right = box[2], box[3]

    #     box = [
    #     [200, 50],   # Horní levý
    #     [300, 50],   # Horní pravý
    #     [250, 150],  # Dolní levý
    #     [350, 150]   # Dolní pravý
    # ]

    # Výpočet šířky a výšky
    width = int(np.linalg.norm(top_right - top_left))  # Šířka
    height = int(np.linalg.norm(bottom_left - top_left))  # Výška

    # Pokud je výška větší než šířka, prohodíme osy (otočíme o 90°)
    # if height > width:
    #     top_left, top_right, bottom_right, bottom_left = bottom_left, top_left, top_right, bottom_right
    #     width, height = height, width  # Přehodíme šířku a výšku

    # Cílové body pro narovnání
    dst_pts = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype="float32"
    )

    # Vypočítání transformační matice
    src_pts = np.array(
        [top_left, top_right, bottom_right, bottom_left], dtype="float32"
    )
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Perspektivní transformace pro narovnání obdélníku
    cropped_frame = cv2.warpPerspective(frame, matrix, (width, height))

    return cropped_frame


def extract_red_box_and_find_contours(frame_gray, box_threashold=70, offset=10):
    """
    Funkce, která hledá největší obdélník v horní levé části oříznutého snímku
    a poté detekuje kontury uvnitř této oblasti.

    :param frame_gray: Oříznutý snímek (ROI) z hlavního snímku.
    :return: Oříznutý červený obdélník s vykreslenými konturami uvnitř.
    """
    if frame_gray is None:
        print("Chyba: frame_gray je None!")
        return None, ""

    x_pct, y_pct, w_pct, h_pct = 5, 10, 25, 25

    # 0. Ořez obrazu na levý horní obdélník
    h, w = frame_gray.shape[:2]

    # Přepočet procent na pixely
    x = int(w * x_pct / 100)
    y = int(h * y_pct / 100)
    w_crop = int(w * w_pct / 100)
    h_crop = int(h * h_pct / 100)

    # Ořezání
    frame_gray = frame_gray[y : y + h_crop, x : x + w_crop]
    # 1. Převod do barevného BGR (pro vykreslení)
    frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    # 2. Thresholding pro detekci obdélníků
    _, thresh = cv2.threshold(frame_gray, 10, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("cr", thresh)

    # 3. Najdi kontury
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Vyber největší konturu v horní levé části
    selected_contour = None
    max_area = 4000

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x < frame_bgr.shape[1] // 2 and y < frame_bgr.shape[0] // 2:
            area = w * h
            if area > max_area:
                max_area = area
                selected_contour = (x, y, w, h)
                print(max_area)
    print("redRec")
    # 5. Oříznutí červeného obdélníku
    if selected_contour:
        x, y, w, h = selected_contour
        roi = frame_bgr[y : y + h, x : x + w]  # Oříznutí
        print(f"Detekovaný obdélník: {w}x{h}")

        # Vykreslení obdélníku na oříznutou oblast
        cv2.rectangle(roi, (0, 0), (w, h), (0, 0, 255), 2)

        # 6. Převod do grayscale a thresholding
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi_thresh = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY_INV)

        # 7. Najdi kontury uvnitř oříznuté oblasti
        roi_contours, _ = cv2.findContours(
            roi_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 8. Vykreslení kontur
        cv2.drawContours(roi, roi_contours, -1, (0, 255, 255), 2)

        # 9. Podmínka pro up/down
        aspect_ratio = w / h if h != 0 else 1

        if abs(w - box_threashold) < offset and abs(h - box_threashold) < offset:
            return roi, "up"
        else:
            return roi, "down"
    print("No Conture")
    return frame_gray, "down"


def crop_square(frame, x_percent, y_percent, size_percent):
    """
    Vyřízne čtvercový výřez z obrázku na základě procent.

    Args:
        frame (numpy.ndarray): Vstupní obrázek (frame).
        x_percent (float): Levý horní roh X (v procentech, 0-100).
        y_percent (float): Levý horní roh Y (v procentech, 0-100).
        size_percent (float): Velikost čtverce (v procentech šířky nebo výšky).

    Returns:
        numpy.ndarray: Oříznutý čtvercový obrázek.
    """
    height, width = frame.shape[:2]

    # Výpočet velikosti čtverce (menší ze šířky a výšky)
    max_size = min(width, height)
    size = int(max_size * size_percent / 100)

    # Převod procent na pixely
    x = int(width * x_percent / 100)
    y = int(height * y_percent / 100)

    # Zajištění, že výřez zůstane v obrázku
    x = max(0, min(x, width - size))
    y = max(0, min(y, height - size))

    # Oříznutí obrázku
    cropped = frame[y : y + size, x : x + size]

    return cropped


def crop_rectangle(frame, x_percent, y_percent, w_percent, h_percent):
    """
    Vyřízne obdélník z obrázku na základě procent.

    Args:
        frame (numpy.ndarray): Vstupní obrázek (frame z kamery nebo obrázek).
        x_percent (float): Levý horní roh X (v procentech, 0-1).
        y_percent (float): Levý horní roh Y (v procentech, 0-1).
        w_percent (float): Šířka výřezu (v procentech, 0-1).
        h_percent (float): Výška výřezu (v procentech, 0-1).

    Returns:
        numpy.ndarray: Oříznutý obrázek.
    """
    height, width = frame.shape[:2]

    # Převod procent na pixely
    x = int(width * x_percent / 100)
    y = int(height * y_percent / 100)
    w = int(width * w_percent / 100)
    h = int(height * h_percent / 100)

    # Oříznutí obrázku
    cropped = frame[y : y + h, x : x + w]

    return cropped


def convert_to_black_and_white(img):
    # Zvýšení kontrastu pomocí vyrovnání histogramu
    img = cv2.equalizeHist(img)
    # Použití thresholdingu (černobílý převod)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # cv2.imshow("img1", blurred)
    _, binary_img = cv2.threshold(blurred, 40, 250, cv2.THRESH_BINARY)
    # cv2.imshow("img2", img)

    return binary_img


def remove_white_borders(img):
    """Ořízne bílé okraje z obrázku s přesnější detekcí hranic."""

    if img is None:
        raise ValueError("❌ Nelze načíst obrázek")

    # 📌 Převod na stupně šedi, pouze pokud má obrázek 3 kanály (BGR)
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 📌 Prahování (threshold) -> černá & bílá maska
    _, thresh = cv2.threshold(
        gray, 40, 255, cv2.THRESH_BINARY_INV
    )  # Inverze (černé objekty na bílém pozadí)

    # 📌 Najdeme všechny obrysy (kontury)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img  # Pokud nejsou nalezeny žádné kontury, vrať původní obrázek

    # 📌 Najdeme největší obdélník, který obklopuje všechny obrysy
    x, y, w, h = cv2.boundingRect(
        np.vstack(contours)
    )  # Spojí všechny kontury do jednoho pole

    # 📌 Oříznutí obrázku na nalezený obdélník
    cropped = img[y : y + h, x : x + w]

    return cropped


def match_ssim(frame, source_folder, file_name):
    similarity_scores_ssim = []

    source_image_name = os.path.join(source_folder, file_name)

    source_image_path = os.path.join(source_folder, source_image_name)
    similarity_scores_ssim = compare_frame_image_ssim(frame, source_image_path)

    return similarity_scores_ssim


def compare_frame_image_ssim(img1_camera, image2_path: str) -> float:
    # Načtení obrázků v odstínech šedi
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    """ Test """
    # heat, overlay, aligned = ssim_heatmap(img1, img2)

    # cv2.imshow("Heat-mapa", heat)
    # cv2.imshow("Overlay", overlay)
    # cv2.imshow("Aligned img", aligned)
    # cv2.waitKey(0)
    """ END Test """

    img1_camera = remove_white_borders(img1_camera)
    img2 = remove_white_borders(img2)
    # Kontrola, zda mají stejnou velikost
    if img1_camera.shape != img2.shape:
        img2 = cv2.resize(img2, (img1_camera.shape[1], img1_camera.shape[0]))

    # Výpočet SSIM indexu
    # větší win_size → tolerantnější k šumu,
    # menší σ v Gaussiánu → důraz na střed pixelového okna,
    # nastavit K1, K2 (v skimage přes k1, k2) pro extrémní dynamiky.
    # score, _ = ssim(
    #     img1, img2, data_range=255, win_size=11, gaussian_weights=True, full=True
    # )

    # ------------------------------------------------------------
    # Nastavení SSIM parametrů
    # ------------------------------------------------------------
    WIN_SIZE = 5  # větší okno → tolerantnější k lokálnímu šumu
    GAUSSIAN = True  # zapne Gaussovské váhy
    SIGMA = 1.5  # menší σ → větší důraz na střed okna,
    #            větší σ → váhy plošší (≈ rovnoměrné)

    K1 = 0.05  # 0.01 je výchozí, větší hodnota tlumí vliv luminančního rozdílu
    K2 = 0.05  # 0.03 je výchozí, větší hodnota tlumí vliv kontrastu/struktury

    # ------------------------------------------------------------
    # Výpočet SSIM
    # ------------------------------------------------------------
    score, ssim_map = ssim(
        img1_camera,
        img2,
        data_range=255,  # 0–255 (důležité!)
        win_size=WIN_SIZE,
        gaussian_weights=GAUSSIAN,
        sigma=SIGMA,
        k1=K1,
        k2=K2,
        full=True,
    )

    # Visualize the images and SSIM map
    cv2.imshow(
        "img1_camera",
        cv2.resize(img1_camera, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR),
    )
    cv2.imshow(
        "Input origin",
        cv2.resize(img2, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR),
    )
    cv2.imshow(
        "SSIM Map",
        cv2.resize(ssim_map, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR),
    )
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.destroyWindow("origin")
    # cv2.destroyWindow("SSIM Map")
    # cv2.destroyWindow("Input Image 1")
    # cv2.destroyWindow("Input Image 2")

    return score  # Hodnota mezi 0 a 1

    """Maps coordinates (x, y) from one range to another."""


def map_coordinates(x, y):
    # Source coordinate range (camera coordinates)
    src_x_min, src_x_max = 2094, 1116
    src_y_min, src_y_max = 440, 1389

    # Target coordinate range (mapped to 0-200)
    dst_x_min, dst_x_max = 0, 200
    dst_y_min, dst_y_max = 0, 200

    mapped_x = ((x - src_x_min) / (src_x_max - src_x_min)) * (
        dst_x_max - dst_x_min
    ) + dst_x_min
    mapped_y = ((y - src_y_min) / (src_y_max - src_y_min)) * (
        dst_y_max - dst_y_min
    ) + dst_y_min

    return round(mapped_x, 2), round(mapped_y, 2)  # Return rounded values


import cv2
import numpy as np
from skimage.metrics import structural_similarity as ms_ssim


def ssim_heatmap(
    img_cam: np.ndarray,
    img_tmpl: np.ndarray,
    *,
    gaussian_sigma: float = 1.5,
    colormap: int = cv2.COLORMAP_JET,
    overlay_alpha: float = 0.5,
):
    """
    Porovná snímek z kamery se šablonou a vrátí:
      heat_map ...... barevnou mapu rozdílů (uint8 BGR)
      overlay ....... heat-mapu překrytou na zarovnaném snímku (BGR)
      aligned ....... zarovnaný (warpnutý) kamerový snímek v odstínech šedi

    Parameters
    ----------
    img_cam : np.ndarray
        Obrázek z kamery (BGR nebo grayscale).
    img_tmpl : np.ndarray
        Šablona (BGR nebo grayscale).
    gaussian_sigma : float, optional
        Sigma pro Gaussovo vážení v SSIM.
    colormap : int, optional
        OpenCV colormap (např. cv2.COLORMAP_JET, TURBO…).
    overlay_alpha : float, optional
        Průhlednost heat-mapy v overlayi (0–1).

    Returns
    -------
    heat_color : np.ndarray  [H, W, 3] uint8
    overlay     : np.ndarray  [H, W, 3] uint8
    aligned     : np.ndarray  [H, W]    uint8
    """

    # --- převod na šedotón --
    cam_gray = (
        cv2.cvtColor(img_cam, cv2.COLOR_BGR2GRAY) if img_cam.ndim == 3 else img_cam
    )
    tmpl_gray = (
        cv2.cvtColor(img_tmpl, cv2.COLOR_BGR2GRAY) if img_tmpl.ndim == 3 else img_tmpl
    )

    # --- ORB feature matching + homografie --
    orb = cv2.ORB_create(1000)
    k1, d1 = orb.detectAndCompute(cam_gray, None)
    k2, d2 = orb.detectAndCompute(tmpl_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)[:80]

    if len(matches) < 4:
        raise RuntimeError("Nedostatek shod pro výpočet homografie.")

    src_pts = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    aligned = cv2.warpPerspective(
        cam_gray, H, (tmpl_gray.shape[1], tmpl_gray.shape[0]), flags=cv2.INTER_LINEAR
    )

    # --- multi-scale SSIM ---
    _, ssim_map = ms_ssim(
        tmpl_gray, aligned, full=True, gaussian_weights=True, sigma=gaussian_sigma
    )

    # --- mapování na 0–255 a colormap ---
    diff = ((1 - ssim_map) / 2 * 255).astype(np.uint8)  # 0 = shoda, 255 = rozdíl
    heat_color = cv2.applyColorMap(diff, colormap)

    # --- overlay ---
    aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(
        aligned_bgr, 1 - overlay_alpha, heat_color, overlay_alpha, 0
    )

    return heat_color, overlay, aligned

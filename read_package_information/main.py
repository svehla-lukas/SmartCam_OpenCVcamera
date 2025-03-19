from camera_thread import CameraThread
import utils_image_processing as imPr
import utils_files
import time
import cv2


settings = {
    "use_camera": False,
}
flags = {"run_loop": True}

nice_label_database_xlsx = r"C:\Users\Uzivatel\Desktop\Potisky - platnost od 2.10.2024 - Baxter CH REP\Zdroje\Vyrobky z PN + RU, EN.xlsx"
# products_informations_json = r"products_informations.json"
products_informations_json = r"data_TraumastemTafLight.json"


"""Capture a single frame without starting video streaming."""
if __name__ == "__main__":
    if settings["use_camera"]:
        camera = CameraThread()

    px_to_mm = imPr.calculate_pixel_size(D=290)
    print(f"px = {round(px_to_mm, 3)} mm")

    """ Rad json data """
    product_references = utils_files.load_json(products_informations_json)
    # print("product_references", product_references)

    """ Read xlsx data """
    # excel_dict = utils_excel.find_ref_row_in_excel(nice_label_database_xlsx, ref_value)
    # print(excel_dict)
    # print(excel_dict["Skupina Výrobků"])

    excel_dict = utils_files.load_excel_to_dict(nice_label_database_xlsx)
    # print(excel_dict[1]["REF"])

    while flags["run_loop"] == True:  # Infinite loop until 'q' is pressed
        if settings["use_camera"]:
            frame_gray = camera.capture_single_frame()
        else:
            frame_gray = cv2.imread("TraumastemTafLight.png")
            frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_BGR2GRAY)

        #  Detect biggest rectangle and crop
        frame_detect, crop_frame, center_x, center_y, angle = imPr.get_biggest_polygon(
            frame_gray, px_to_mm=px_to_mm, pixelsArea=(100 * 100 / px_to_mm)
        )
        cv2.imshow("origin", cv2.resize(frame_detect, (640, 480)))

        if crop_frame is not None:
            print("frame cropped")
            cv2.imshow("croped", cv2.resize(crop_frame, (640, 480)))

        # Read REF from product
        position_px = tuple(
            int(value / px_to_mm) for value in product_references["REF"]["position_mm"]
        )
        read_REF_code = imPr.extract_text_from_frame(
            crop_frame, position_px=position_px
        )
        # print(read_REF_code)

        # fill json dict from excel_Nicelabel by read REF code
        utils_files.fill_json_from_excel(
            product_json=product_references,
            excel_dict=excel_dict,
            REF_code=read_REF_code,
        )

        # Iterate through json and valid texts
        for key, value in product_references.items():
            # print(value)
            position = value.get("position_mm")
            origin_text = value.get("correct_text", "")
            position_px = tuple(
                int(value / px_to_mm) for value in value.get("position_mm")
            )
            language = value.get("language", "eng") or "eng"
            read_text = imPr.extract_text_from_frame(
                frame=crop_frame,
                position_px=position_px,
                language=language,
                show_image=True,
            )
            similarity = imPr.text_similarity(read_text, origin_text)
            print(f"checking: {key}")
            print(f"origin text: {read_text}")
            print(f"read text  : {origin_text}")
            print(f"similarity {similarity}")
            print("\n")
            cv2.waitKey(0)  # Waits for a key press before closing

        # end loop
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            flags["run_loop"] = False
            cv2.destroyAllWindows()
            print("🔴 'q' pressed. Exiting...")
            break
        else:
            print(f"🔵 Key '{chr(key)}' pressed. Continuing...")
            time.sleep(1)
            continue

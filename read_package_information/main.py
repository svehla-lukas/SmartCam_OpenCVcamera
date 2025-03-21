import utils_image_processing as imPr
import utils_files
import time
import cv2
import numpy as np
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "camera_thread_HKVision")
    )
)
import CameraThread


settings = {
    "use_camera": False,
    "text_compare": True,
}
flags = {"run_loop": True}

nice_label_database_xlsx = r"C:\Users\Uzivatel\Desktop\Potisky - platnost od 2.10.2024 - Baxter CH REP\Zdroje\Vyrobky z PN + RU, EN.xlsx"
# products_information_json = r"products_informations.json"
products_information_json = r"data_TraumastemTafLight.json"


"""Capture a single frame without starting video streaming."""
if __name__ == "__main__":
    if settings["use_camera"]:
        camera = CameraThread()

    px_to_mm = imPr.calculate_pixel_size(D=290)
    print(f"px = {round(px_to_mm, 3)} mm")

    """ Rad json data """
    product_references = utils_files.load_json(products_information_json)
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

        smallest_area = 100 * 100
        #  Detect biggest rectangle and crop
        frame_detect, crop_frame, center_x, center_y, angle = imPr.get_biggest_polygon(
            frame_gray, px_to_mm=px_to_mm, pixelsArea=(smallest_area / px_to_mm)
        )

        cv2.imshow("origin", cv2.resize(frame_detect, None, fx=0.5, fy=0.5))
        if crop_frame is not None:
            box_text_relative_position = [50, 100, 200, 150]
            first_pixel = imPr.detect_first_black_pixel(
                crop_frame, box_text_relative_position
            )

            if first_pixel is not None:
                print(f"✅ First text pixel found at: {first_pixel}")
            else:
                print("❌ No text detected in the image.")
            ### Read REF position from product
            position_px = tuple(
                int(value / px_to_mm)
                for value in product_references["REF"]["position_mm"]
            )
            read_REF_code = imPr.extract_text_from_frame(
                crop_frame, position_px=position_px, show_image=True
            )
            # print(read_REF_code)

            # fill json dict from excel_Nicelabel by read REF code
            utils_files.fill_json_from_excel(
                product_json=product_references,
                excel_dict=excel_dict,
                REF_code=read_REF_code,
            )

            ### Calculate position offset for local package
            position_abs = product_references["product_name"][
                "position_NiceLabel_abs_mm"
            ]
            position_diff = [
                first_pixel[0] * px_to_mm - position_abs[0] - 1,
                first_pixel[1] * px_to_mm - position_abs[1] - 1,
                1,
                1,
            ]

            position_offset = np.array(position_abs) + position_diff
            print(f"position_offset: {position_offset}")

            # Check text on package
            if settings["text_compare"]:
                # Iterate through json and valid texts
                for key, value in product_references.items():
                    # print(value)
                    # position = value.get("position_mm")
                    position_abs = value.get("position_NiceLabel_abs_mm")
                    position_rel = np.array(position_abs) + position_diff
                    position_px = tuple(int(value / px_to_mm) for value in position_rel)
                    # print(f"first_px [mm] :{first_pixel * px_to_mm}")
                    # print(f"position_abs {position_abs}")
                    # print(f"position_diff {position_diff}")
                    # print(f"position_rel: {position_rel}")
                    # print(f"position_px: {position_px}")

                    origin_text = value.get("correct_text", "")
                    language = value.get("language", "eng") or "eng"
                    read_text = imPr.extract_text_from_frame(
                        frame=crop_frame,
                        position_px=position_px,
                        language=language,
                        show_image=True,
                    )
                    similarity = imPr.text_similarity(read_text, origin_text)
                    print(f"checking: {key}")
                    if value.get("excel_colomn_name"):
                        print("get from Excel")
                    print(f"origin text: {origin_text}")
                    print(f"read text  : {read_text}")
                    print(f"similarity {similarity}")
                    # print("\n")
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

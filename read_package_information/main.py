import utils_image_processing as imPr
import utils_files
import time
import cv2
import numpy as np
import sys
import os
import pathlib
import math

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "camera_thread_HKVision")
    )
)
from CameraThread import CameraThread


path_output_database = pathlib.Path.cwd() / "output.json"

settings = {
    "use_camera": True,
    "text_compare": True,
    "snap_pictograms": True,
    "draw_pictograms_on_origin": False,
    "draw_pictograms_on_cropped": True,
}
flags = {"run_loop": True}
camera_heigh = 290

nice_label_database_xlsx = r"C:\Users\Uzivatel\Desktop\Potisky - platnost od 2.10.2024 - Baxter CH REP\Zdroje\Vyrobky z PN + RU, EN.xlsx"
# products_information_json = r"products_informations.json"
products_information_json = r"data_products.json"


""" Read json data """
product_references = utils_files.load_json(products_information_json)

"""Capture a single frame without starting video streaming."""
if __name__ == "__main__":
    if settings["use_camera"]:
        camera = CameraThread()

    px_to_mm = imPr.calculate_pixel_size(D=camera_heigh)
    print(f"px = {round(px_to_mm, 3)} mm")

    # print("product_references", product_references)

    """ Read xlsx data """
    # excel_dict = utils_excel.find_ref_row_in_excel(nice_label_database_xlsx, ref_value)
    # print(excel_dict)
    # print(excel_dict["Skupina Výrobků"])

    excel_dict = utils_files.load_excel_to_dict(nice_label_database_xlsx)

    while flags["run_loop"] == True:  # Infinite loop until 'q' is pressed
        if settings["use_camera"]:
            frame_gray = camera.capture_single_frame()
        else:
            frame_gray = cv2.imread("TraumastemTafLight1.png")
            frame_gray = cv2.imread("TraumastemTafLight0.png")
            frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_BGR2GRAY)

        smallest_area = 100 * 100
        #  Detect biggest rectangle and crop
        frame_detect, crop_frame, crop_frame_box, angle = imPr.get_biggest_polygon(
            frame_gray, px_to_mm=px_to_mm, pixelsArea=(smallest_area / px_to_mm)
        )
        print(px_to_mm)
        if crop_frame is None:
            cv2.imshow("origin", cv2.resize(frame_detect, None, fx=0.5, fy=0.5))
        else:
            # cv2.imshow("origin", cv2.resize(crop_frame, None, fx=1, fy=1))
            ### Detect first black pixel
            box_text_relative_position_px = [50, 50, 400, 400]
            first_pixel_px = imPr.detect_first_black_pixel(
                crop_frame,
                box_text_relative_position_px,
            )

            if first_pixel_px is not None:
                print(f"✅ First text pixel found at: {first_pixel_px}")
            else:
                print("❌ No text detected in the image.")
                break

            ### Read product name from First pixel
            position_px = [
                int(first_pixel_px[0] - 3 / px_to_mm),
                int(first_pixel_px[1] - 3 / px_to_mm),
                int(60 / px_to_mm),
                int(8 / px_to_mm),
            ]
            read_produce_name = imPr.extract_text_from_frame(
                crop_frame, position_px=position_px, show_image=True
            )

            # delete white space
            read_produce_name = "".join(read_produce_name.split()).lower()
            match read_produce_name:
                case "traumastemtaf":
                    print("Detected: Traumastem Taf")
                    product_name = "Traumastem_Taf"
                    cv2.waitKey(0)
                case "traumastemtaflight":
                    print("Detected: Traumastem Taf Light")
                    product_name = "Traumastem_Taf_Light"
                    cv2.waitKey(0)
                case "celstatfibrillar":
                    print("Detected: Celstat Fibrillar")
                    product_name = "Celstat_Fibrillar"
                    cv2.waitKey(0)
                case "traumastemp":
                    print("Detected: Traumastem P")
                    product_name = "Traumastem_P"
                    cv2.waitKey(0)

            print(f"read_produce_name: {read_produce_name}")
            cv2.waitKey(0)

            ### Calculate position offset for local package
            position_abs_mm = product_references[product_name]["texts"]["product_name"][
                "position_NiceLabel_abs_mm"
            ]
            position_offset_mm = [
                first_pixel_px[0] * px_to_mm - position_abs_mm[0],
                first_pixel_px[1] * px_to_mm - position_abs_mm[1],
                1 + 1,
                1 + 1,
            ]

            print(f"position_abs [mm]: {position_abs_mm}")
            print(f"position_first_px [mm]: {first_pixel_px * px_to_mm}")
            print(f"position_offset [mm]: {position_offset_mm}")
            cv2.waitKey(0)

            ### Read REF position from product
            position_px = list(
                int(value / px_to_mm)
                for value in product_references[product_name]["texts_numbers"]["REF"][
                    "position_mm"
                ]
            )
            position_px[0] = int(position_px[0] + position_offset_mm[0] / px_to_mm)
            position_px[1] = int(position_px[1] + position_offset_mm[1] / px_to_mm)

            read_REF_code = imPr.extract_text_from_frame(
                crop_frame, position_px=position_px, show_image=True
            )
            print(f"read_REF_code: {read_REF_code}")
            cv2.waitKey(0)

            # fill json dict from excel_Nicelabel by read REF code
            utils_files.fill_json_from_excel(
                product_json=product_references,
                excel_dict=excel_dict,
                REF_code=read_REF_code,
            )

            if settings["draw_pictograms_on_origin"]:
                for key, item in product_references[product_name][
                    "texts_numbers"
                ].items():
                    # convert z mm na px
                    position_px = list(
                        int(v / px_to_mm) for v in item.get("position_NiceLabel_abs_mm")
                    )
                    # Add box[0] offset coordinates to position_px x,y values
                    if crop_frame_box is not None:
                        position_px = (
                            position_px[0]
                            + int(crop_frame_box[0][0])
                            + int(position_offset_mm[0] / px_to_mm),
                            position_px[1]
                            + int(crop_frame_box[0][1])
                            + int(position_offset_mm[1] / px_to_mm),
                            position_px[2],
                            position_px[3],
                        )
                    print(position_px[0])
                    print(int(crop_frame_box[0][0]))
                    print(int(position_offset_mm[0] / px_to_mm))
                    imPr.draw_rotated_rect(
                        frame_detect, position_px, item.get("name"), angle=angle
                    )
                    # cv2.imshow("origin", cv2.resize(frame_detect, None, fx=1, fy=1))
                    cv2.imshow("origin", cv2.resize(frame_detect, None, fx=0.4, fy=0.4))
            else:
                cv2.imshow("origin", cv2.resize(frame_detect, None, fx=0.4, fy=0.4))

            if settings["draw_pictograms_on_cropped"]:
                crop_frame_temp = crop_frame.copy()

                combined = {
                    **product_references[product_name]["texts_numbers"],
                    **product_references[product_name]["texts"],
                    "QR_code": product_references[product_name]["QR_code"],
                }
                for key, item in combined.items():
                    try:
                        position_px = list(
                            int(v / px_to_mm)
                            for v in item.get("position_mm")
                            # int(v / px_to_mm)
                            # for v in item.get("position_NiceLabel_abs_mm")
                        )
                        # Add box[0] offset coordinates to position_px x,y values
                        if crop_frame_box is not None:
                            position_px = (
                                position_px[0] + int(position_offset_mm[0] / px_to_mm),
                                position_px[1] + int(position_offset_mm[1] / px_to_mm),
                                position_px[2],
                                position_px[3],
                            )

                        imPr.draw_rotated_rect(
                            crop_frame_temp, position_px, item.get("name"), angle=0
                        )
                        # cv2.imshow("origin", cv2.resize(frame_detect, None, fx=1, fy=1))

                        cv2.circle(
                            crop_frame_temp,
                            [
                                int(first_pixel_px[0]),
                                int(first_pixel_px[1]),
                            ],
                            radius=10,
                            color=(0, 0, 0),
                            thickness=4,
                        )
                        cv2.imshow(
                            "cropped", cv2.resize(crop_frame_temp, None, fx=0.4, fy=0.4)
                        )
                    except Exception as e:
                        print("--------------")
                        print("draw_pictograms_on_cropped:")
                        print(f"- Error processing {key}: {e}")
                        print("--------------")

            # Get pictogram images
            # if settings["snap_pictograms"]:
            #     for key, item in product_references[product_name]["pictograms"].items():
            #         print(item.get("position_mm"))

            #         position_mm = item.get("position_mm")

            #         # Převod z mm na px
            #         position_px = tuple(int(v / px_to_mm) for v in position_mm)
            #         x, y, w, h = position_px

            #         # Zkontrolujeme, že výška a šířka nejsou nula (aby nedošlo k chybě)
            #         if w > 0 and h > 0:
            #             frame_text_code = crop_frame[y : y + h, x : x + w]
            #             cv2.imshow(key, frame_text_code)
            #             cv2.waitKey(0)

            frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_BGR2RGB)

            cv2.circle(
                frame_gray,
                [
                    int(crop_frame_box[0][0]),
                    int(crop_frame_box[0][1]),
                ],
                radius=10,
                color=(0, 255, 0),
                thickness=4,
            )

            cv2.circle(
                frame_gray,
                [
                    int(crop_frame_box[0][0] + first_pixel_px[0]),
                    int(crop_frame_box[0][1] + first_pixel_px[1]),
                ],
                radius=10,
                color=(255, 0, 0),
                thickness=4,
            )
            cv2.imshow("origin", cv2.resize(frame_gray, None, fx=0.4, fy=0.4))

            # Check text on package
            if settings["text_compare"]:
                # Iterate through json and valid texts
                combined = {
                    **product_references[product_name]["texts_numbers"],
                    **product_references[product_name]["texts"],
                }
                for key, item in combined.items():
                    # print(item)
                    # position = item.get("position_mm")
                    # position_abs_mm = item.get("position_NiceLabel_abs_mm")
                    position_abs_mm = item.get("position_mm")
                    print(f"posAbs {position_abs_mm}")
                    print(f"posOff {position_offset_mm}")
                    position_rel_mm = np.array(position_abs_mm) + position_offset_mm
                    print(f"posRel {position_rel_mm}")
                    position_px = tuple(
                        int(value / px_to_mm) for value in position_rel_mm
                    )
                    print(f"px {position_px}")

                    origin_text = item.get("correct_text", "")
                    language = item.get("language", "eng") or "eng"
                    read_text = imPr.extract_text_from_frame(
                        frame=crop_frame,
                        position_px=position_px,
                        language=language,
                        show_image=True,
                    )
                    similarity = imPr.text_similarity(read_text, origin_text)
                    print(path_output_database)
                    utils_files.update_json_value(path_output_database, key, read_text)

                    print("\n")
                    print(f"Checking: {key}")
                    if item.get("excel_colomn_name"):
                        print("get from Excel")
                    print(f"origin text: {origin_text}")
                    print(f"read text  : {read_text}")
                    print(f"similarity {similarity}")
                    cv2.waitKey(0)  # Waits for a key press before closing
                    try:
                        cv2.destroyWindow("Extracted Text Frame")
                    except:
                        continue
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

import utils_image_processing as imPr
import time
import cv2
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
    "measure_rectangle": True,
}
flags = {"run_loop": True}


"""Capture a single frame without starting video streaming."""
if __name__ == "__main__":
    if settings["use_camera"]:
        camera = CameraThread()

    px_to_mm = imPr.calculate_pixel_size(D=290)
    print(f"px = {round(px_to_mm, 3)} mm")

    while flags["run_loop"] == True:  # Infinite loop until 'q' is pressed
        if settings["use_camera"]:
            frame_gray = camera.capture_single_frame()
        else:
            frame_gray = cv2.imread("TraumastemTafLight.png")
            frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_BGR2GRAY)

        smallest_area = 100 * 100

        """Measure width and height"""
        largest_contour = None
        if settings["measure_rectangle"] and frame_gray is not None:
            largest_contour = imPr.find_largest_contour(frame_gray)

            frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            if largest_contour is not None:
                frame_bgr = imPr.draw_contour_info(
                    frame_bgr,
                    largest_contour,
                    "biggest object",
                    focal_length=200,
                    text_offset=100,
                )

            cv2.imshow("Measure_distance", cv2.resize(frame_bgr, (640, 480)))

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

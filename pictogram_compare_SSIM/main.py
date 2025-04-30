import numpy as np
from functions import *
import cv2
import time
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "camera_thread_HKVision")
    )
)
from CameraThread import CameraThread

image_package_path = "TraumastemTafLight.png"

settings = {
    "use_camera": True,
}


"""Capture a single frame without starting video streaming."""
if __name__ == "__main__":
    center_x, center_y, angle = 0, 0, 0
    camera = CameraThread()  # No callback needed

    try:
        while True:  # Infinite loop until 'q' is pressed
            time.sleep(1)  # Wait 1 second before capturing the next frame
            if settings["use_camera"]:
                frame = camera.capture_single_frame()  # Capture one frame
                # Save the captured frame
                # filename = f"TraumastemTafLight.png"
                # cv2.imwrite(filename, frame)
                # print(f"📸 Frame saved as {filename}")
            else:
                frame = cv2.imread(image_package_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # capture_on_keypress(frame, "/captured_image.png")

            if frame is not None:
                plg_frame, center_x, center_y, angle = get_biggest_polygon(frame, 10000)
                # 3. Vykreslení textu s informacemi v levém horním rohu
                if angle is not None:
                    text = f"Stred: ({center_x}, {center_y}) | Uhel: {angle:.2f}deg"
                    cv2.putText(
                        plg_frame,
                        text,
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.7,
                        (255, 255, 255),
                        2,
                    )

                cv2.imshow("plg Frame", resize_frame(plg_frame, 40))

                # Print center coordinates
                if center_x is not None and center_y is not None:
                    center_x, center_y = map_coordinates(center_x, center_y)
                    print(f"🟢 Mapped coordinates: ({center_x}, {center_y}, {angle})")
                else:
                    print("⚠ No valid contour detected. Skipping coordinate mapping.")

            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("🔴 'q' pressed. Exiting...")
                break

    except KeyboardInterrupt:
        print("\n🔴 KeyboardInterrupt detected. Exiting...")

    camera.stop()  # Ensure camera resources are released
    cv2.destroyAllWindows()  # Close OpenCV windows

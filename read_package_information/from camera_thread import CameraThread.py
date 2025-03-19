from camera_thread import CameraThread


use_camera = True
use_camera = False


settings = {
        "crop_pictograms": False,
        "check_pictograms_with_NN": False,
        "measure_rectangle": False,
        "read_DMC": False,
        "produce_ident": True,
    }

"""Capture a single frame without starting video streaming."""
if __name__ == "__main__":
    if use_camera:
        camera = CameraThread()  # No callback needed

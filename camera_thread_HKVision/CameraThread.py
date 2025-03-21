# -- coding: utf-8 --
import sys
import threading
import numpy as np
import cv2
from ctypes import *
from MvImport.MvCameraControl_class import *


class CameraThread:
    def __init__(self, callback=None):
        """Initializes the camera and prepares the capture thread."""
        self.cam = None
        self.running = False
        self.thread = None
        self.latest_frame = None  # Stored latest frame
        self.callback = callback  # Optional callback function for image processing

    def initialize_camera(self):
        """Initializes and opens the camera."""
        device_list = MV_CC_DEVICE_INFO_LIST()
        tlayer_type = MV_GIGE_DEVICE | MV_USB_DEVICE

        ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
        if ret != 0 or device_list.nDeviceNum == 0:
            raise RuntimeError("No devices found.")

        print(f"Found {device_list.nDeviceNum} devices.")

        self.cam = MvCamera()
        device_info = cast(
            device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)
        ).contents

        ret = self.cam.MV_CC_CreateHandle(device_info)
        if ret != 0:
            raise RuntimeError(f"Failed to create camera handle! ret[0x{ret:x}]")

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise RuntimeError(f"Failed to open device! ret[0x{ret:x}]")

        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            raise RuntimeError(f"Failed to set trigger mode! ret[0x{ret:x}]")

        self.configuration()

    def configuration(self):
        # 🛠 Set Custom Camera Settings
        ret = self.cam.MV_CC_SetIntValue("Width", 2592)
        if ret != 0:
            print(f"⚠ Failed to set Width! ret[0x{ret:x}]")

        ret = self.cam.MV_CC_SetIntValue("Height", 1944)
        if ret != 0:
            print(f"⚠ Failed to set Height! ret[0x{ret:x}]")

        ret = self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_Mono8)  # MONO8
        if ret != 0:
            print(f"⚠ Failed to set PixelFormat! ret[0x{ret:x}]")

        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", 5000.0)  # Exposure 5ms
        if ret != 0:
            print(f"⚠ Failed to set ExposureTime! ret[0x{ret:x}]")

        ret = self.cam.MV_CC_SetFloatValue("Gain", 1.0)  # Gain level
        if ret != 0:
            print(f"⚠ Failed to set Gain! ret[0x{ret:x}]")

        ret = self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 5.0)  # FPS 5
        if ret != 0:
            print(f"⚠ Failed to set AcquisitionFrameRate! ret[0x{ret:x}]")

        # ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)  # Continuous mode
        # if ret != 0:
        #     print(f"⚠ Failed to set TriggerMode! ret[0x{ret:x}]")
        ret = self.cam.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE)
        if ret != 0:
            print(f"⚠ Failed to set TriggerSource! ret[0x{ret:x}]")

    def start(self):
        """Starts the capture thread and begins grabbing frames."""
        if self.running:
            print("❗ Camera is already running!")
            return

        self.initialize_camera()

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"Failed to start grabbing! ret[0x{ret:x}]")

        self.running = True
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.start()

    def stop(self):
        """Stops the capture thread and closes the camera."""
        if not self.running:
            print("❗ Camera is not active.")
            return

        self.running = False
        self.thread.join()
        self.cam.MV_CC_StopGrabbing()
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()
        print("✅ Camera successfully shut down.")

    def _capture_frames(self):
        """Internal method for continuously capturing frames in the thread."""
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))

        while self.running:
            ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)  # Timeout 1000ms
            if stOutFrame.pBufAddr is not None and ret == 0:
                # Convert to NumPy array
                img_data = np.ctypeslib.as_array(
                    cast(stOutFrame.pBufAddr, POINTER(c_ubyte)),
                    shape=(
                        stOutFrame.stFrameInfo.nHeight,
                        stOutFrame.stFrameInfo.nWidth,
                    ),
                )

                self.latest_frame = img_data.copy()  # Store the latest frame

                # Callback function for external processing
                if self.callback:
                    self.callback(img_data)  # If a callback is provided, use it
                else:
                    cv2.imshow(
                        "Live Video", img_data
                    )  # Otherwise, use default OpenCV display
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break  # Stop when 'q' is pressed

                self.cam.MV_CC_FreeImageBuffer(stOutFrame)
            else:
                print(f"No data [0x{ret:x}]")

        cv2.destroyAllWindows()

    def get_latest_frame(self):
        """Returns the most recently stored frame."""
        return self.latest_frame

    def capture_single_frame(self):
        """Captures a single frame from the camera without starting continuous capture."""
        if self.cam is None:  # Initialize camera only if it's not already created
            self.initialize_camera()

        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))

        ret = self.cam.MV_CC_StartGrabbing()  # Start grabbing for a single frame
        if ret != 0:
            print(f"⚠ Failed to start grabbing! ret[0x{ret:x}]")
            return None

        ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)  # Timeout 1000ms
        if stOutFrame.pBufAddr is not None and ret == 0:
            img_data = np.ctypeslib.as_array(
                cast(stOutFrame.pBufAddr, POINTER(c_ubyte)),
                shape=(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth),
            )

            self.cam.MV_CC_FreeImageBuffer(stOutFrame)  # Release buffer
            self.cam.MV_CC_StopGrabbing()  # Stop grabbing after capturing one frame
            return img_data.copy()  # Return a copy of the frame

        print(f"⚠ Failed to capture frame! ret[0x{ret:x}]")
        self.cam.MV_CC_StopGrabbing()  # Ensure grabbing stops even if frame capture fails
        return None

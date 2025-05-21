import cv2
import numpy as np
import pytest


# ----------  basic image ---------- #
@pytest.fixture()
def blank_frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


# ---------- fixture: synthetic image ----------
@pytest.fixture()
def test_frame():
    """
    Returns a 100 × 100 RGB image:
        left half  = red   (BGR: 0, 0, 255)
        right half = green (BGR: 0, 255, 0)
    """
    rows = np.arange(120, dtype=np.uint8).reshape(-1, 1, 1)
    frame = np.repeat(rows, 90 * 3).reshape(120, 90, 3)
    return frame


# ----------  contours ---------- #
@pytest.fixture()
def rectangle_contour():
    # 80×40 – rectangle
    return np.array([[[0, 0]], [[80, 0]], [[80, 40]], [[0, 40]]], dtype=np.int32)


@pytest.fixture()
def square_contour():
    return np.array([[[0, 0]], [[40, 0]], [[40, 40]], [[0, 40]]], dtype=np.int32)


@pytest.fixture()
def circle_contour():
    # approximate circle with radius 20
    return cv2.ellipse2Poly(
        center=(50, 50), axes=(20, 20), angle=0, arcStart=0, arcEnd=360, delta=5
    )


# ----------  disable GUI ---------- #
@pytest.fixture(autouse=True)
def patch_no_gui(monkeypatch):
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a, **k: 0)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)


# ----------  fake camera ---------- #
class _FakeCam:
    def __init__(self, frame):
        self.frame = frame

    def isOpened(self):
        return True

    def read(self):
        return True, self.frame.copy()

    def release(self):
        pass


@pytest.fixture()
def mock_camera(monkeypatch, blank_frame):
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: _FakeCam(blank_frame))

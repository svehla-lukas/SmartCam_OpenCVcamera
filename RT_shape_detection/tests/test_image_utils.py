import pytest
import numpy as np
import cv2
from RT_shape_detection.shapeDetect import (
    preprocessFrame,
    multiLineTextCv2,
    cropFrameCv2,
)


# ----------  fixture: syntetic picture  ----------
def test_preprocess_frame_shape_and_flip(test_frame):
    """
    Verifies that `preprocessFrame`:
      • crops the frame using the default ratios in `cropFrameCv2`
        (i.e. keeps only the middle third horizontally),
      • flips the result horizontally (flipCode = 1).
    """

    # 1) Expected output if we only crop (no flip)
    expected = cropFrameCv2(test_frame)

    # 2) Call the function under test
    processed = preprocessFrame(test_frame)

    # --- A) Shape check ------------------------------------------------
    assert processed.shape == expected.shape

    # --- B) Flip check -------------------------------------------------
    # A second horizontal flip should restore the cropped image
    double_flipped = cv2.flip(processed, 1)
    assert np.array_equal(double_flipped, expected)


def test_crop_default(test_frame):
    cropped = cropFrameCv2(test_frame)

    # Expect height unchanged, width = 90 * (2/3 - 1/3) = 30
    assert cropped.shape == (120, 30, 3)

    # The slice should equal the middle-third of the source
    expected = test_frame[:, 30:60, :]
    assert np.array_equal(cropped, expected)


@pytest.mark.parametrize(
    "v_start, v_end, h_start, h_end",
    [
        (0.25, 0.75, 0.0, 1.0),  # remove top & bottom quarters
        (0.0, 1.0, 0.0, 0.5),  # left half only
        (0.5, 1.0, 0.5, 1.0),  # bottom-right quadrant
    ],
)
def test_crop_custom(test_frame, v_start, v_end, h_start, h_end):
    cropped = cropFrameCv2(
        test_frame,
        vertical_start_ratio=v_start,
        vertical_end_ratio=v_end,
        horizontal_start_ratio=h_start,
        horizontal_end_ratio=h_end,
    )

    # --- expected shape ---
    h, w, _ = test_frame.shape
    exp_h = int(h * (v_end - v_start))
    exp_w = int(w * (h_end - h_start))
    assert cropped.shape == (exp_h, exp_w, 3)

    # --- expected content ---
    y0, y1 = int(h * v_start), int(h * v_end)
    x0, x1 = int(w * h_start), int(w * h_end)
    expected = test_frame[y0:y1, x0:x1, :]
    assert np.array_equal(cropped, expected)


# 1) MOCK-BASED TEST
"""
Mock-based test for multiLineTextCv2:

- We replace cv2.putText with a fake function that records each call’s text and position.
- Call multiLineTextCv2 with three lines of text.
- Assert that putText was called exactly once per line.
- Verify each call used the correct string and computed coordinates (start_x, start_y + i*line_spacing).

This ensures our function iterates lines correctly and computes the right positions without relying on actual OpenCV drawing.
"""


def test_multiLineTextCv2_called_for_each_line(monkeypatch):
    calls = []

    # nothing paint, only save arguments
    def fake_putText(img, text, org, font, font_scale, color, thickness, line_type):
        calls.append((text, org))

    monkeypatch.setattr(cv2, "putText", fake_putText)

    img = np.zeros((50, 200, 3), dtype=np.uint8)
    multiLineTextCv2(
        img,
        "Line1\nLine2\nLine3",
        start_x=5,
        start_y=10,
        line_spacing=15,
        font_scale=1.0,
        color=(255, 255, 255),
        thickness=1,
    )

    # expect 3 calls, with correct positions
    assert len(calls) == 3
    assert calls[0] == ("Line1", (5, 10 + 0 * 15))
    assert calls[1] == ("Line2", (5, 10 + 1 * 15))
    assert calls[2] == ("Line3", (5, 10 + 2 * 15))


# 2) IMAGE-BASED TEST
def test_multiLineTextCv2_pixels_changed_by_putText():
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    # kreslíme text „A“ na pozici (50,50)

    multiLineTextCv2(
        img, "Test text", start_x=10, start_y=50, line_spacing=10, color=(255, 255, 255)
    )

    assert np.any(img != 0), "Expected some pixels to be drawn, but block is all zeros"

import pytest

# from ..shapeDetect import normalizeAngle
from RT_shape_detection.shapeDetect import normalizeAngle


@pytest.mark.parametrize(
    "input, output",
    [
        (0, 0),
        (44, 44),
        (45, 45),  # edge
        (60, 30),
        (135, 45),
        (225, 45),
        (359, 1),
    ],
)
def test_normalize_angle(input, output):
    assert normalizeAngle(input) == output


def test_detect_rectangles_angle():
    pass

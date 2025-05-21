import pytest
import cv2
from RT_shape_detection.shapeDetect import classifyShape, classifyRectangleOrSquare


def test_classifyShape(rectangle_contour, square_contour, circle_contour):
    COLORS = {
        "square": (0, 255, 0),  # Green
        "rectangle": (255, 0, 0),  # Blue
        "circle": (0, 255, 255),  # Yellow
    }
    shape, label_rectangle, color, angle, center = classifyShape(
        rectangle_contour,
        COLORS,
    )
    shape, label_square, color, angle, center = classifyShape(
        square_contour,
        COLORS,
    )
    shape, label_circle, color, angle, center = classifyShape(
        circle_contour,
        COLORS,
    )
    assert label_rectangle == "Rectangle"
    assert label_square == "Square"
    assert label_circle == "Circle"


def test_classifyRectangleOrSquare(rectangle_contour, square_contour):
    COLORS = {"square": (0, 255, 0), "rectangle": (255, 0, 0)}

    # prepare the 'approx' polygons
    peri_rect = cv2.arcLength(rectangle_contour, True)
    approx_rect = cv2.approxPolyDP(rectangle_contour, 0.02 * peri_rect, True)

    peri_sq = cv2.arcLength(square_contour, True)
    approx_sq = cv2.approxPolyDP(square_contour, 0.02 * peri_sq, True)

    # now call the function with correct types
    _, label_rect, _, _, _ = classifyRectangleOrSquare(
        rectangle_contour, approx_rect, COLORS
    )
    _, label_sq, _, _, _ = classifyRectangleOrSquare(square_contour, approx_sq, COLORS)

    assert label_rect == "Rectangle"
    assert label_sq == "Square"


"""
TO DO:


def test_classify_circle():
    pass
"""

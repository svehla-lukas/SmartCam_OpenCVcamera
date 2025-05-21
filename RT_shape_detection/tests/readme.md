RT_shape_detection/
├─ shapeDetect.py
├─ motionDetect.py
└─ tests/
    ├─ test_math.py              # normalizeAngle
    ├─ test_image_utils.py       # cropFrameCv2, preprocessFrame, multiLineTextCv2
    ├─ test_classify.py          # classifyRectangleOrSquare / Circle / Shape
    ├─ test_logic.py             # adjustDelay, handleDetection
    └─ test_pipeline_small.py    # detectRectanglesAngle with syntetic frame
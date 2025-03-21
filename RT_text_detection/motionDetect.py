import cv2
import numpy as np


def trackObject():
    """Function to track moving objects and display their trajectory."""
    # Initialize camera
    cap = cv2.VideoCapture(0)

    # Check if the camera is accessible
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Initialize variables
    previous_frame = None
    trajectory = np.zeros(
        (480, 640, 3), dtype=np.uint8
    )  # For storing trajectory (adjust size if needed)

    while True:
        # Capture current frame
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame (stream end?). Exiting...")
            break

        # Resize frame for consistency (if needed)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5))

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Save the current frame every second
        if previous_frame is not None:
            # Calculate absolute difference between current and previous frame
            diff = cv2.absdiff(previous_frame, gray_frame)

            # Threshold the difference to highlight significant changes
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

            # Find contours of the moving objects
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Loop through contours to find the largest one (assuming it's the object to track)
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Ignore small changes/noise
                    # Get bounding box of the object
                    x, y, w, h = cv2.boundingRect(contour)
                    center = (x + w // 2, y + h // 2)

                    # Draw a rectangle and trajectory point
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(
                        trajectory, center, 3, (0, 0, 255), -1
                    )  # Draw point on trajectory

        # Show the current frame with trajectory
        combined = cv2.addWeighted(frame, 0.8, trajectory, 0.2, 0)
        cv2.imshow("Object Tracking", combined)

        # Save the trajectory frame
        cv2.imwrite("trajectory_frame.png", combined)

        # Update previous frame
        previous_frame = gray_frame.copy()

        # Clear the trajectory on pressing 'd'
        if cv2.waitKey(30) & 0xFF == ord("d"):
            trajectory = np.zeros((480, 640, 3), dtype=np.uint8)
            print("Trajectory cleared!")

        # Exit on pressing 'q'
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

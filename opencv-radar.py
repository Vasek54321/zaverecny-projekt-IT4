import cv2
import numpy as np
import time

# Constants
FPS = 120  # Frames per second of the video
DISTANCE_PER_PIXEL = 0.05  # Distance in meters represented by one pixel (calibrate this value)

# Initialize video capture (0 for webcam or 'video.mp4' for a video file)
cap = cv2.VideoCapture('Untitled.mp4')

# Background subtractor
backSub = cv2.createBackgroundSubtractorMOG2()

# Variables to store previous position and time
prev_position = None
prev_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fg_mask = backSub.apply(frame)

    # Find contours of the moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Filter out small contours
            continue

        # Get bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)

        # Draw the bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate speed
        current_time = time.time()
        if prev_position is not None and prev_time is not None:
            # Calculate distance in pixels
            distance_pixels = np.linalg.norm(np.array(center) - np.array(prev_position))
            # Convert distance to meters
            distance_meters = distance_pixels * DISTANCE_PER_PIXEL
            # Calculate time difference in hours
            time_diff_hours = (current_time - prev_time) / 3600
            # Calculate speed in km/h
            speed_kmh = distance_meters / 1000 / time_diff_hours if time_diff_hours > 0 else 0

            # Display speed on the frame
            cv2.putText(frame, f'Speed: {speed_kmh:.2f} km/h', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Update previous position and time
        prev_position = center
        prev_time = current_time

    # Show the frame
    cv2.imshow('Car Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
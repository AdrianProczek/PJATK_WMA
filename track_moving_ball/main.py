import cv2
import numpy as np

SCALE_PERCENT = 50

cap = cv2.VideoCapture("movingball.mp4")

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_delay = int(1000 / fps)


def preprocess_frame(frame):
    """
    Processes a film frame: resizes, converts to HSV, applies filters and morphological operations
    """

    width = int(frame.shape[1] * SCALE_PERCENT / 100)
    height = int(frame.shape[0] * SCALE_PERCENT / 100)
    frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return frame_resized, mask


def detect_ball(frame, mask):
    """
    Finds the outline of the ball, draws the center and marks the size
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            cv2.circle(frame, center, radius, (0, 255, 0), 2)

            cv2.circle(frame, center, 5, (255, 0, 0), -1)

            cv2.putText(frame, f"Size: {radius * 2} px", (center[0] - 40, center[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized, mask = preprocess_frame(frame)
    frame_tracked = detect_ball(frame_resized, mask)

    cv2.imshow("Tracking", frame_tracked)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

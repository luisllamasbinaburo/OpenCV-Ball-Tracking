import cv2

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    frame = cv2.medianBlur(frame, 3);
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_hue_range = cv2.inRange(hsvFrame, (0, 100, 100), (10, 255, 255));
    upper_hue_range = cv2.inRange(hsvFrame, (160, 100, 100), (179, 255, 255));
    
    mask = cv2.addWeighted(lower_hue_range, 1.0, upper_hue_range, 1.0, 0.0);
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

    cv2.imshow("Camera", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()


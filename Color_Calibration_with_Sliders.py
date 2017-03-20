import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def nothing(x):
    pass

cv2.namedWindow('Threshold')
cv2.createTrackbar("Hue Lower Bound", 'Threshold', 0, 179, nothing)
cv2.createTrackbar("Hue Upper Bound", 'Threshold', 0, 179, nothing)
cv2.createTrackbar("Saturation Lower Bound", 'Threshold', 0, 255, nothing)
cv2.createTrackbar("Saturation Upper Bound", 'Threshold', 0, 255, nothing)
cv2.createTrackbar("Value Lower Bound", 'Threshold', 0, 255, nothing)
cv2.createTrackbar("Value Upper Bound", 'Threshold', 0, 255, nothing)

while(True):

    ret, frame = cap.read()

#Get the positions of the trackbars

    lowerHue = cv2.getTrackbarPos("Hue Lower Bound", 'Threshold')
    upperHue = cv2.getTrackbarPos("Hue Upper Bound", 'Threshold')
    lowerSaturation = cv2.getTrackbarPos("Saturation Lower Bound", 'Threshold')
    upperSaturation = cv2.getTrackbarPos("Saturation Upper Bound", 'Threshold')
    lowerValue = cv2.getTrackbarPos("Value Lower Bound", 'Threshold')
    upperValue = cv2.getTrackbarPos("Value Upper Bound", 'Threshold')

    if lowerHue > upperHue or lowerSaturation > upperSaturation or lowerValue > upperValue:
        lowerHue-=1

    else:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, (lowerHue, lowerSaturation, lowerValue), (upperHue, upperSaturation, upperValue))
        print (lowerHue, lowerSaturation, lowerValue)
        print (upperHue, upperSaturation, upperValue)
        image, cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) > 0:
            cv2.drawContours(frame, cnts, -1, (0, 255, 0), 2)

    cv2.imshow("Threshold", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.camera.release()
cv2.destroyAllWindows()

import cv2

cap = cv2.VideoCapture(0)

for i in range(0,12):
    while (True):
        ret, frame = cap.read()
        cv2.imshow('Calibration Photo' + str(i), frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.imwrite('calib'+str(i) + '.jpg', frame)

import numpy as np
import cv2
from collections import deque
import time

idealRatio = 0.39  # 4/10.25
ratioTolerance = 0.075

firstErodeKernel = np.ones((5,5),np.uint8)
secondDilateKernal = np.ones((50,50),np.uint8)
thirdErodeKernel = np.ones((50,50),np.uint8)

lower_bound = np.array([85, 0, 50])
upper_bound = np.array([100, 255, 255])

cap = cv2.VideoCapture(1)

_, frame = cap.read()

frameWidth = int(cap.get(3))
print frameWidth
frameHeight = int(cap.get(4))
print frameHeight

while (1):
    start = time.time()

    _, frame = cap.read()

    img = frame

    raw = frame
    cv2.imshow('Original', raw)
    erosion1 = cv2.erode(raw, firstErodeKernel)
    cv2.imshow('First Erosion', erosion1)
    dilate = cv2.dilate(erosion1, secondDilateKernal)
    cv2.imshow('Dilate', dilate)
    erosion2 = cv2.erode(dilate, thirdErodeKernel)
    cv2.imshow('Second Erosion', erosion2)

    frame = erosion2
    img = frame

    goalView = np.zeros((frameHeight, frameWidth, 3), np.uint8)

    ### FINDING AND OUTLINING THE GOAL###

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    goalContourRow = 0

    if len(contours) > 0:

        # Eliminating anything too small and putting the rest in "goalContours"

        #print contours

        #unified = []

        hull = cv2.convexHull(contours[0])

        if len(contours) > 1:
            if len(contours) == 2:
                goal = np.vstack(contours[i] for i in range(0, 2))
                insideArea = cv2.contourArea(contours[0]) + cv2.contourArea(contours[1])
                outline = cv2.convexHull(goal)
                outsideArea = cv2.contourArea(outline)
                ratio = insideArea/(outsideArea + 0.01)
                if ratio >= idealRatio - ratioTolerance and ratio <= idealRatio + ratioTolerance:
                    M = cv2.moments(outline)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    print str(cX) + " " + str(cY)
                    cv2.circle(goalView, (cX, cY), 2, (0, 0, 255), 4)
                    for j in range(0, len(outline)):
                        cv2.line(img, (outline[j - 1][0][0], outline[j - 1][0][1]), (outline[j][0][0], outline[j][0][1]), (0, 255, 0), 2)
                        cv2.line(goalView, (outline[j - 1][0][0], outline[j - 1][0][1]), (outline[j][0][0], outline[j][0][1]), (0, 255, 0), 2)

            #elif len(goalContours) == 3:
            #    blah
            #else:
            #    blah

    cont = cv2.drawContours(frame, contours, -1, (0, 255, 0))

    ###FINDING AND CIRCLING THE CORNERS###

    cv2.imshow('Final Product', img)
    cv2.imshow('Mask', mask)
    cv2.imshow('Goal', goalView)
    #cv2.imshow('Blah', hsv)

    end = time.time()

    fps = 1/(end-start)

    print fps

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
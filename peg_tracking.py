import numpy as np
import cv2
from collections import deque
import time
from networktables import NetworkTable
import subprocess # to run shell commands


table = NetworkTable.getTable("PegCenters")

idealRatio = 0.39  # 4/10.25
ratioTolerance = 0.075

firstErodeKernel = np.ones((5,5),np.uint8)
secondDilateKernal = np.ones((50,50),np.uint8)
thirdErodeKernel = np.ones((50,50),np.uint8)

lower_bound = np.array([80, 0, 40])
upper_bound = np.array([100, 255, 255])

cap = cv2.VideoCapture(0)

_, frame = cap.read()

frameWidth = int(cap.get(3))
print frameWidth
frameHeight = int(cap.get(4))
print frameHeight

areaArray = []

# set webcam exposure controls
# for some reason, it doesn't work if we only run it before this script
# we apparently need to run it within this script
subprocess.call("./config_webcam.sh", shell=True)

while (1):
    start = time.time()

    _, frame = cap.read()

    #img = frame

    #raw = frame
    #cv2.imshow('Original', raw)
    #erosion1 = cv2.erode(raw, firstErodeKernel)
    #cv2.imshow('First Erosion', erosion1)
    #dilate = cv2.dilate(erosion1, secondDilateKernal)
    #cv2.imshow('Dilate', dilate)
    #erosion2 = cv2.erode(dilate, thirdErodeKernel)
    #cv2.imshow('Second Erosion', erosion2)

    #frame = erosion2

    ### FINDING AND OUTLINING THE GOAL###
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 1:
        for c in contours:
            area = cv2.contourArea(c)
            areaArray.append(area)
        sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
        goal = np.vstack(sorteddata[i][1] for i in range(0, 2))
        insideArea = cv2.contourArea(contours[0]) + cv2.contourArea(contours[1])
        outline = cv2.convexHull(goal)
        outsideArea = cv2.contourArea(outline)
        ratio = insideArea/(outsideArea + 0.01)
        if ratio >= idealRatio - ratioTolerance and ratio <= idealRatio + ratioTolerance:
            M = cv2.moments(outline)
            pegCenterX = int(M["m10"] / M["m00"])
            pegCenterY = int(M["m01"] / M["m00"])
            pegCenterX = float(pegCenterX-frameWidth/2)/(frameWidth/2)
            pegCenterY = float(pegCenterY-frameHeight/2)/(frameHeight/2)
            print str(pegCenterX) + " " + str(pegCenterY)
            table.putNumber('pegCenterX', pegCenterX)
            table.putNumber('pegCenterY', pegCenterY)
            for j in range(0, len(outline)):
                cv2.line(frame, (outline[j - 1][0][0], outline[j - 1][0][1]), (outline[j][0][0], outline[j][0][1]), (0, 255, 0), 2)
        else:
            table.putNumber('pegCenterX', -10)
            table.putNumber('pegCenterY', -10)
    else:
        table.putNumber('pegCenterX', -10)
        table.putNumber('pegCenterY', -10)
                

    end = time.time()

    fps = 1/(end-start)

    print fps

    table.putNumber('timestamp', start)

    cv2.imshow('Camera', frame)
    cv2.imshow('Mask', mask)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

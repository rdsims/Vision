import numpy as np
import cv2
#from collections import deque
import time
from networktables import NetworkTable
import subprocess # to run shell commands

# set global flags
displayFlag = False
erodeFlag = False
displayErodeFlag = False

# initialize NetworkTable server
NetworkTable.setServerMode()
NetworkTable.initialize()
table = NetworkTable.getTable("Vision")

# get image frame size
cap = cv2.VideoCapture(0)
_, frame = cap.read()
frameWidth = int(cap.get(3))
print frameWidth
frameHeight = int(cap.get(4))
print frameHeight

# set webcam exposure controls
# for some reason, it doesn't work if we only run it before this script
# we apparently need to run it within this script
subprocess.call("./config_webcam.sh", shell=True)

# set image processing constants
idealRatio = 0.39  # 4/10.25
ratioTolerance = 0.075

firstErodeKernel = np.ones((5,5),np.uint8)
secondDilateKernal = np.ones((50,50),np.uint8)
thirdErodeKernel = np.ones((50,50),np.uint8)

lower_bound = np.array([80, 0, 40])
upper_bound = np.array([100, 255, 255])


while True:
    start = time.time()

    # set some default values for info sent to roboRIO
    targetCenterX = -999
    targetCenterY = -999
    targetArea = -999

    # capture image frame
    _, frame = cap.read()

    if erodeFlag:
        # erode & dilate to remove noise
        # (this takes a lot of processing.  replaced with simpler sort by area, below)
        raw = frame
        img = frame

        erosion1 = cv2.erode(raw, firstErodeKernel)
        dilate = cv2.dilate(erosion1, secondDilateKernal)
        erosion2 = cv2.erode(dilate, thirdErodeKernel)
        frame = erosion2

        if displayErodeFlag:
            cv2.imshow('Original', raw)
            cv2.imshow('First Erosion', erosion1)
            cv2.imshow('Dilate', dilate)
            cv2.imshow('Second Erosion', erosion2)


    # convert RGB image to HSV for filtering
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # filter on HSV
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # find contours of what's left
    image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # process contours to find center of target
    if len(contours) > 1:
        # get area of all contours
        areaArray = []
        for c in contours:
            area = cv2.contourArea(c)
            areaArray.append(area)
            
        # sort contours by area (the true target should be largest, the rest is noise we can ignore
        sortedData = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
        
        # goal consists of the 2 largest contours
        goal = np.vstack(sortedData[i][1] for i in range(0, 2))
        
        # calculate area of positive and negative space in target
        insideArea = cv2.contourArea(contours[0]) + cv2.contourArea(contours[1])
        outline = cv2.convexHull(goal)
        outsideArea = cv2.contourArea(outline)
        
        # filter once more by ratio of area in target
        ratio = insideArea/(outsideArea + 0.01)
        if (ratio >= idealRatio - ratioTolerance) and (ratio <= idealRatio + ratioTolerance):
            # we have found the target.  calculate location of center, normalized to the size of the image frame
            M = cv2.moments(outline)
            targetCenterX = int(M["m10"] / M["m00"])
            targetCenterY = int(M["m01"] / M["m00"])
            targetCenterX = float(targetCenterX-frameWidth/2)/(frameWidth/2)        # normalize
            targetCenterY = float(targetCenterY-frameHeight/2)/(frameHeight/2)      # normalize
            targetArea = outsideArea/(frameHeight*frameWidth)                       # normalize

            # add lines around target display
            if displayFlag:
                for j in range(0, len(outline)):
                    cv2.line(frame, (outline[j - 1][0][0], outline[j - 1][0][1]), (outline[j][0][0], outline[j][0][1]), (0, 255, 0), 2)
                    
            # print target center to screen
            print "X: " + str(targetCenterX) + ", Y: " + str(targetCenterY) + ", Area:" + str(targetArea)

    # calculate the frame rate
    end = time.time()
    fps = 1/(end-start)
    print fps

    # send target information to roboRIO
    table.putNumber('timestamp', start)
    table.putNumber('targetCenterX', targetCenterX)
    table.putNumber('targetCenterY', targetCenterY)
    table.putNumber('targetArea',    targetArea)
    table.putNumber('fps', fps)

    if displayFlag:
        cv2.imshow('Camera', frame)
        cv2.imshow('Mask', mask)

    #k = cv2.waitKey(5) & 0xFF
    #if k == 27:
    #    break

if displayFlag or displayErodeFlag:
    cv2.destroyAllWindows()

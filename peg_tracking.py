import numpy as np
import cv2
import time
from networktables import NetworkTable
import subprocess # to run shell commands
import gtk
import sys

# set global flags
displayFlag = 0
erodeFlag = 0

#try:
#    gtk_init_check(null, null)
#except (NameError):
#    print("Disabling display...")
#    displayFlag = 0
#except:
#    print("Error:", sys.exc_info()[0])

cameraMatrix = np.loadtxt("/home/pi/git/Vision/camera_matrix.txt")
distortionCoefs = np.loadtxt("/home/pi/git/Vision/distortion_coefs.txt")


# initialize NetworkTable server
NetworkTable.setIPAddress("10.96.86.2")
NetworkTable.setClientMode()
NetworkTable.initialize()
table = NetworkTable.getTable("SmartDashboard")

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
subprocess.call("/home/pi/git/Vision/config_webcam.sh", shell=True)

# set image processing constants
idealRatio = 0.39  # 4/10.25
ratioTolerance = 0.200

firstErodeKernel = np.ones((5,5),np.uint8)
secondDilateKernal = np.ones((50,50),np.uint8)
thirdErodeKernel = np.ones((50,50),np.uint8)

lower_bound = np.array([60,   0, 200])
upper_bound = np.array([90, 255, 255])


while (True):
    start = time.time()

    # set some default illegal values for info sent to roboRIO
    targetCenterX = -999
    targetCenterY = -999
    targetWidth = -999
    targetHeight = -999
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

        if displayFlag:
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

        # Draw top four contour outlines: 1 and 2: blue, 3 and 4: red
        #for i in range(0,2):
        #    if displayFlag:
        #        for j in range(0, len(sortedData[i][1])):
        #            cv2.line(frame, (sortedData[i][1][j - 1][0][0], sortedData[i][1][j - 1][0][1]), (sortedData[i][1][j][0][0], sortedData[i][1][j][0][1]), (255, 0, 0), 2)
        #for i in range(2,4):
        #    if displayFlag:
        #        for j in range(0, len(sortedData[i][1])):
        #            cv2.line(frame, (sortedData[i][1][j - 1][0][0], sortedData[i][1][j - 1][0][1]), (sortedData[i][1][j][0][0], sortedData[i][1][j][0][1]), (0, 0, 255), 2)     

        # calculate area of positive and negative space in target
        insideArea = cv2.contourArea(sortedData[0][1]) + cv2.contourArea(sortedData[1][1])
        outline = cv2.convexHull(goal)
        outsideArea = cv2.contourArea(outline)

        # filter once more by ratio of area in target
        ratio = insideArea/(outsideArea + 0.01)
        if (ratio >= idealRatio - ratioTolerance) and (ratio <= idealRatio + ratioTolerance):
            # we have found the target.

            # calculate location of center, normalized to the size of the image frame
            M = cv2.moments(outline)
            x,y,outsideWidth,outsideHeight = cv2.boundingRect(outline)
            targetCenterX = int(M["m10"] / M["m00"])
            targetCenterY = int(M["m01"] / M["m00"])
            targetCenterX = float(targetCenterX-frameWidth/2)/(frameWidth/2)        # normalize
            targetCenterY = float(targetCenterY-frameHeight/2)/(frameHeight/2)      # normalize
            targetArea = float(outsideArea)/(frameHeight*frameWidth)                # normalize
            targetWidth = float(outsideWidth)/frameWidth			    # normalize
            targetHeight = float(outsideHeight)/frameHeight                         # normalize

            # add lines around target display
            if displayFlag:
                for j in range(0, len(outline)):
                    cv2.line(frame, (outline[j - 1][0][0], outline[j - 1][0][1]), (outline[j][0][0], outline[j][0][1]), (0, 255, 0), 2)

	else:
            print "Ratio={:5.3f}".format(ratio)+", Limits=({:5.3f}".format(idealRatio-ratioTolerance)+",{:5.3f})".format(idealRatio+ratioTolerance)


    if displayFlag:
        cv2.imshow('Camera', frame)
        cv2.imshow(  'Mask', mask)

    # calculate the frame rate
    end = time.time()
    fps = 1/(end-start)

    # send target information to roboRIO
    table.putNumber('imageTimestamp', start)
    table.putNumber('targetCenterX', targetCenterX)
    table.putNumber('targetCenterY', targetCenterY)
    table.putNumber('targetWidth',   targetWidth)
    table.putNumber('targetHeight',  targetHeight)
    table.putNumber('targetArea',    targetArea)
    table.putNumber('fps', fps)

    if targetArea > 0:
        # print target center to screen
        print "FPS:{:3.1f}".format(fps) + ", X:{: 5.3f}".format(targetCenterX) + ", Y:{: 5.3f}".format(targetCenterY) + ", W:{:5.3f}".format(targetWidth) + ", H:{:5.3f}".format(targetHeight) + ", A:{:5.3f}".format(targetArea)
    else:
        print "FPS:{:3.1f}".format(fps) + ", Target not found";



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if displayFlag or displayErodeFlag:
    cv2.destroyAllWindows()

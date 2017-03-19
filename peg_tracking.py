import numpy as np
import cv2
import time
from networktables import NetworkTable
import subprocess # to run shell commands
import gtk
import sys

# Set global flags
displayFlag = False
erodeFlag = False
calculateAngles = True

#try:
#    gtk_init_check(null, null)
#except (NameError):
#    print("Disabling display...")
#    displayFlag = 0
#except:
#    print("Error:", sys.exc_info()[0])

cameraMatrix = np.loadtxt("/home/pi/git/Vision/camera_matrix.txt")
distortionCoefs = np.loadtxt("/home/pi/git/Vision/distortion_coefs.txt")


# Initialize NetworkTable server
NetworkTable.setIPAddress("10.96.86.2")
NetworkTable.setClientMode()
NetworkTable.initialize()
table = NetworkTable.getTable("SmartDashboard")

# Get image frame size
cap = cv2.VideoCapture(0)
_, frame = cap.read()
frameWidth = int(cap.get(3))
print frameWidth
frameHeight = int(cap.get(4))
print frameHeight

frameSize = [frameWidth, frameHeight]
FoV = [60, 60/frameWidth*frameHeight]

# Define a function to convert pixel locations to horizontal and vertical angles

def pixelToAngles(point, frameSize, FoV):
    HAtT = (point[0]+0.5 - frameSize[0]/2) * FoV[0]/frameSize[0]
    VAtT = (point[1]+0.5 - frameSize[1]/2) * FoV[1]/frameSize[1]
    return [HAtT, VAtT]

# Set webcam exposure controls
# For some reason, it doesn't work if we only run it before this script
# We apparently need to run it within this script
subprocess.call("/home/pi/git/Vision/config_webcam.sh", shell=True)

# Set image processing constants
idealRatio = 0.39  # 4/10.25
ratioTolerance = 0.200

firstErodeKernel = np.ones((5,5),np.uint8)
secondDilateKernal = np.ones((50,50),np.uint8)
thirdErodeKernel = np.ones((50,50),np.uint8)

lower_bound = np.array([60,   0, 200])
upper_bound = np.array([90, 255, 255])


while (True):
    start = time.time()

    # Set some default illegal values for info sent to roboRIO
    targetCenterX = -999
    targetCenterY = -999
    targetWidth = -999
    targetHeight = -999
    targetArea = -999

    # Capture image frame
    _, frame = cap.read()

    if erodeFlag:
        # Erode & dilate to remove noise
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

    # Convert RGB image to HSV for filtering
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Filter on HSV
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find contours of what's left
    image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Process contours to find center of target
    if len(contours) > 1:
        # Get area of all contours
        areaArray = []
        for c in contours:
            area = cv2.contourArea(c)
            areaArray.append(area)

        # Sort contours by area (the true target should be largest, the rest is noise we can ignore
        sortedData = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

        # Goal consists of the 2 largest contours
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

        # Calculate area of positive and negative space in target
        insideArea = cv2.contourArea(sortedData[0][1]) + cv2.contourArea(sortedData[1][1])
        outline = cv2.convexHull(goal)
        outsideArea = cv2.contourArea(outline)

        # Filter once more by ratio of area in target
        ratio = insideArea/(outsideArea + 0.01)
        if (ratio >= idealRatio - ratioTolerance) and (ratio <= idealRatio + ratioTolerance):
            # We have found the target.
            
            TLscore = np.empty([len(outline)])
            TRscore = np.empty([len(outline)])
            
            # Calculate location of center, normalized to the size of the image frame
            M = cv2.moments(outline)
            x,y,outsideWidth,outsideHeight = cv2.boundingRect(outline)
            targetCenterX = int(M["m10"] / M["m00"])
            targetCenterY = int(M["m01"] / M["m00"])
            targetCenterX = float(targetCenterX-frameWidth/2)/(frameWidth/2)        # normalize
            targetCenterY = float(targetCenterY-frameHeight/2)/(frameHeight/2)      # normalize
            targetArea = float(outsideArea)/(frameHeight*frameWidth)                # normalize
            targetWidth = float(outsideWidth)/frameWidth                # normalize
            targetHeight = float(outsideHeight)/frameHeight                         # normalize
            
            # Calculate the horizontal and vertical angles to the target points
            if calculateAngles:
                for j in range(0, len(outline)):
                    TLscore[j] = outline[j][0][1] - outline[j][0][0]
                    TRscore[j] = outline[j][0][1] + outline[j][0][0]
                
                TLindex = np.argmax(TLscore)
                BRindex = np.argmin(TLscore)
                TRindex = np.argmax(TRscore)
                BLindex = np.argmin(TRscore)

                TL = [outline[TLindex][0][0], outline[TLindex][0][1]]
                TR = [outline[TRindex][0][0], outline[TRindex][0][1]]
                BL = [outline[BLindex][0][0], outline[BLindex][0][1]]
                BR = [outline[BRindex][0][0], outline[BRindex][0][1]]
                CR = [cX, cY]

                TLAngles = pixelToAngles(TL, frameSize, FoV)
                TRAngles = pixelToAngles(TR, frameSize, FoV)
                BLAngles = pixelToAngles(BL, frameSize, FoV)
                BRAngles = pixelToAngles(BR, frameSize, FoV)
                CRAngles = pixelToAngles(CR, frameSize, FoV)
            
            # Add lines around target display
            if displayFlag:
                for j in range(0, len(outline)):
                    cv2.line(frame, (outline[j - 1][0][0], outline[j - 1][0][1]), (outline[j][0][0], outline[j][0][1]), (0, 255, 0), 2)

    else:
        print "Ratio={:5.3f}".format(ratio)+", Limits=({:5.3f}".format(idealRatio-ratioTolerance)+",{:5.3f})".format(idealRatio+ratioTolerance)


    if displayFlag:
        cv2.imshow('Camera', frame)
        cv2.imshow(  'Mask', mask)

    # Calculate the frame rate
    end = time.time()
    fps = 1/(end-start)

    # Send target information to roboRIO
    table.putNumber('Camera/imageTimestamp', start)
    table.putNumber('Camera/targetCenterX', targetCenterX)
    table.putNumber('Camera/targetCenterY', targetCenterY)
    table.putNumber('Camera/targetWidth',   targetWidth)
    table.putNumber('Camera/targetHeight',  targetHeight)
    table.putNumber('Camera/targetArea',    targetArea)
    table.putNumber('Camera/fps', fps)
    if calculateAngles:
        table.putNumberArray('CR', CRAngles)
        table.putNumberArray('TL', TLAngles)
        table.putNumberArray('TR', TRAngles)
        table.putNumberArray('BL', BLAngles)
        table.putNumberArray('BR', BRAngles)

    if targetArea > 0:
        # Print target center to screen
        print "FPS:{:3.1f}".format(fps) + ", X:{: 5.3f}".format(targetCenterX) + ", Y:{: 5.3f}".format(targetCenterY) + ", W:{:5.3f}".format(targetWidth) + ", H:{:5.3f}".format(targetHeight) + ", A:{:5.3f}".format(targetArea)
    else:
        print "FPS:{:3.1f}".format(fps) + ", Target not found";



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if displayFlag:
    cv2.destroyAllWindows()

import numpy as np
import cv2
from collections import deque
from networktables import NetworkTable

table = NetworkTable.getTable("GoalCenters")

idealRatio = 0.314  # 88/280
# idealRatio = 0.5
ratioTolerance = 0.075

lower_bound = np.array([0, 0, 217])
upper_bound = np.array([179, 255, 255])

#rightCorners = deque(maxlen = 2)
#leftCorners = deque(maxlen = 2)

#slope = 0.0000

cap = cv2.VideoCapture(0)

_, frame = cap.read()

frameWidth = cap.get(3)
print frameWidth
frameHeight = cap.get(4)
print frameHeight

print _

while (1):
    _, frame = cap.read()

    if _ == True:

        img = frame

        goal = np.zeros((frameHeight, frameWidth, 3), np.uint8)

        ### FINDING AND OUTLINING THE GOAL###

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:

            maxGoalSize = 0

            for i in range(0, len(contours)):
                hull = cv2.convexHull(contours[i])

                insideArea = cv2.contourArea(contours[i])
                hullArea = cv2.contourArea(hull)
                ratio = insideArea / (hullArea + 0.1)

                if ratio > idealRatio - ratioTolerance and ratio < idealRatio + ratioTolerance:
                    if hullArea > maxGoalSize:
                        maxGoalSize = hullArea
                        maxGoalLocation = i

            for i in range(0, len(contours)):
                hull = cv2.convexHull(contours[i])

                insideArea = cv2.contourArea(contours[i])
                hullArea = cv2.contourArea(hull)
                ratio = insideArea / (hullArea + 0.1)

                if ratio > idealRatio - ratioTolerance and ratio < idealRatio + ratioTolerance:
                    if i == maxGoalLocation:
                        for j in range(0, len(hull)):
                            cv2.line(img, (hull[j - 1][0][0], hull[j - 1][0][1]), (hull[j][0][0], hull[j][0][1]), (0, 255, 0), 2)
                            cv2.line(goal, (hull[j - 1][0][0], hull[j - 1][0][1]), (hull[j][0][0], hull[j][0][1]), (0, 255, 0), 2)
                        x, y, w, h = cv2.boundingRect(hull)
                        cv2.rectangle(goal, (x,y), (x+w, y+h), (255, 0, 0), 2)
                        goalCenterX = x+(w/2)
                        goalCenterY = y+h/2
                        goalCenter = (goalCenterX, goalCenterY)
                        cv2.circle(goal, goalCenter, 3, (0, 0, 255), -1)
                        table.putNumber('goalCenterX', goalCenterX)
                        table.putNumber('goalCenterY', goalCenterY)
                        print goalCenter
                    else:
                        for j in range(0, len(hull)):
                            cv2.line(img, (hull[j - 1][0][0], hull[j - 1][0][1]), (hull[j][0][0], hull[j][0][1]), (0, 0, 255), 2)

        else:
            table.putNumber('goalCenterX', -1)
            table.putNumber('goalCenterY', -1)

    #    cont = cv2.drawContours(frame, contours, -1, (0, 255, 0))

        ###FINDING AND CIRCLING THE CORNERS###

    #    gray = cv2.cvtColor(goal, cv2.COLOR_BGR2GRAY)
        #gray = cv2.blur(gray, (4, 4))

    #    corners = cv2.goodFeaturesToTrack(gray, 4, 0.01, 10)
    #    if corners is not None:
    #        corners = np.int0(corners)
    #        numcorn = len(corners[:, 0, 0])
    #
    #        for i in corners:
    #            x,y = i.ravel()
    #            cv2.circle(goal, (x,y), 3, (0, 0, 255), -1)

            ###IDENTIFYING EACH CORNER
    #        if numcorn == 4:
    #            rightestPoint = 0
    #            leftestPoint = frameWidth
    #            for i in range(0, 4):
    #                if corners[i, 0, 0] > rightestPoint:
    #                    rightestPoint = corners[i, 0, 0]
    #                if corners[i, 0, 0] < leftestPoint:
    #                    leftestPoint = corners[i, 0, 0]
    #            midPoint = (rightestPoint - leftestPoint)/2 + leftestPoint
    #
    #            for i in range(0, 4):
    #                if corners[i, 0, 0] > midPoint:
    #                    rightCorners.appendleft((float(corners[i, 0, 0]), float(corners[i, 0, 1])))
    #                if corners[i, 0, 0] < midPoint:
    #                    leftCorners.appendleft((float(corners[i, 0, 0]), float(corners[i, 0, 1])))
    #
    #            if rightCorners[0][1] > rightCorners[1][1]:
    #                topRightCorner = rightCorners[1]
    #                bottomRightCorner = rightCorners[0]
    #            else:
    #                topRightCorner = rightCorners[0]
    #                bottomRightCorner = rightCorners[1]
    #
    #            if leftCorners[0][1] > leftCorners[1][1]:
    #                topLeftCorner = leftCorners[1]
    #                bottomLeftCorner = leftCorners[0]
    #            else:
    #                topLeftCorner = leftCorners[0]
    #                bottomLeftCorner = leftCorners[1]
    #
    #            ###CALCULATIONS OF ROBOT POSITION###
    #            yval = (bottomLeftCorner[1] - bottomRightCorner[1])
    #            xval = (bottomRightCorner[0] - bottomLeftCorner[0])
    #            slope = yval/xval
    #            print yval
    #            print xval
    #            print slope


        #cv2.imshow('Final Product', img)
        #cv2.imshow('Mask', mask)
        cv2.imshow('Goal', goal)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

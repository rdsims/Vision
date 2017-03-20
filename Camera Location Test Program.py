import numpy
import random
import math

# The whole point of this program is to generate a random location and orientation for a camera on a robot, generate
# some fake data, and see how the algorithm does.

# These are the identified points on the target in relation to the center.
targetPoints = [[-5.125,  2.5, 0],
                [ 5.125,  2.5, 0],
                [-5.125, -2.5, 0],
                [ 5.125, -2.5, 0]]

# Height of the center of the target above the floor
targetHeight = 13.25

# Max boundaries for camera
maxRobotWidth = 18
maxRobotLength = 18
maxRobotHeight = 24


def genCamLocation():
    x = float(random.randrange(-maxRobotWidth, maxRobotWidth + 1))
    y = float(random.randrange(0, maxRobotHeight + 1))
    z = float(random.randrange(-maxRobotLength, maxRobotLength + 1))
    camHoriz = math.radians(float(random.randrange(-180, 180 + 1)))
    camVert = math.radians(float(random.randrange(-90, 90 + 1)))
    camTilt = math.radians(float(random.randrange(-180, 180 + 1)))
    return x, y, z, camHoriz, camVert, camTilt


def genRobotLocation():
    x = float(random.randrange(-48, 48 + 1))
    z = float(random.randrange(0, 96 + 1))
    heading = math.radians(float(random.randrange(-180, 180 + 1)))
    return x, z, heading

#def calcTargetAngles(camLoc, robotLoc, targetPoints):

for i in range(100):
    x, y, z, camHoriz, camVert, camTilt = genCamLocation()
    print genCamLocation()
    print genRobotLocation()



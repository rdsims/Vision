import numpy
import cv2
from collections import deque
import time
import math
import scipy.optimize as optimize


HAtTCR = math.radians(33.63201926)
VAtTCR = math.radians(12.24244416)
HAtTTL = math.radians(22.48132194)
VAtTTL = math.radians(16.7347041)
HAtTTR = math.radians(45.71546071)
VAtTTR = math.radians(18.02833644)
HAtTBL = math.radians(22.48132194)
VAtTBL = math.radians(6.17684119)
HAtTBR = math.radians(45.71546071)
VAtTBR = math.radians(6.68171465)


def targetError(c):
    return math.sqrt(((41*math.cos(c[3])*math.cos(c[1]))/8 + (5*math.cos(c[1])*math.sin(c[3]))/2 + (41*pow(math.cos(c[2]), 2)*math.cos(c[1]))/4 - (41*pow(math.cos(c[2]), 2)*pow(math.cos(c[1]), 3))/4 - (5*math.cos(c[3])*math.sin(c[2])*math.sin(c[1]))/2 + (41*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))/8 - c[0]*math.cos(VAtTCR)*math.sin(HAtTCR) - (41*math.cos(c[3])*pow(math.cos(c[2]), 2)*math.cos(c[1]))/4 - 5*pow(math.cos(c[2]), 2)*math.sin(c[2])*math.sin(c[1]) + (41*math.cos(c[3])*pow(math.cos(c[2]), 2)*pow(math.cos(c[1]), 3))/4 + 5*math.cos(c[3])*pow(math.cos(c[2]), 2)*math.sin(c[2])*math.sin(c[1]) - (math.cos(VAtTBL)*math.sin(HAtTBL)*(math.sin(VAtTBL)*(5*math.cos(c[2]) - (5*math.cos(c[3])*math.cos(c[2]))/2 + (41*math.cos(c[2])*math.sin(c[3]))/8 - 5*pow(math.cos(c[2]), 3) + 5*math.cos(c[3])*pow(math.cos(c[2]), 3) - 5*math.cos(c[2])*pow(math.cos(c[1]), 2) - c[0]*math.sin(VAtTCR) + 5*pow(math.cos(c[2]), 3)*pow(math.cos(c[1]), 2) + 5*math.cos(c[3])*math.cos(c[2])*pow(math.cos(c[1]), 2) - (41*math.cos(c[2])*pow(math.cos(c[1]), 2)*math.sin(c[3]))/4 - 5*math.cos(c[3])*pow(math.cos(c[2]), 3)*math.cos(c[1])**2 - (41*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + (41*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + 5*math.cos(c[2])*math.cos(c[1])*math.sin(c[3])*math.sin(c[2])*math.sin(c[1])) + math.cos(VAtTBL)*math.sin(HAtTBL)*((41*math.cos(c[3])*math.cos(c[1]))/8 + (5*math.cos(c[1])*math.sin(c[3]))/2 + (41*math.cos(c[2])**2*math.cos(c[1]))/4 - (41*math.cos(c[2])**2*math.cos(c[1])**3)/4 - (5*math.cos(c[3])*math.sin(c[2])*math.sin(c[1]))/2 + (41*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))/8 - c[0]*math.cos(VAtTCR)*math.sin(HAtTCR) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1]))/4 - 5*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3)/4 + 5*math.cos(c[3])*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + 5*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1])) + math.cos(HAtTBL)*math.cos(VAtTBL)*((41*math.cos(c[3])*math.sin(c[1]))/8 + (5*math.sin(c[3])*math.sin(c[1]))/2 - (41*math.cos(c[1])*math.sin(c[3])*math.sin(c[2]))/8 - c[0]*math.cos(HAtTCR)*math.cos(VAtTCR) - 5*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[2])**2*math.sin(c[3])*math.sin(c[1]) + 5*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) + (41*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4 + (5*math.cos(c[3])*math.cos(c[1])*math.sin(c[2]))/2 + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4)))/(math.sin(VAtTBL)**2 + math.cos(HAtTBL)**2*math.cos(VAtTBL)**2 + math.cos(VAtTBL)**2*math.sin(HAtTBL)**2) + 5*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]))**2 + ((41*math.cos(c[1])*math.sin(c[3])*math.sin(c[2]))/8 - (5*math.sin(c[3])*math.sin(c[1]))/2 - (41*math.cos(c[3])*math.sin(c[1]))/8 + c[0]*math.cos(HAtTCR)*math.cos(VAtTCR) + 5*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) + 5*math.cos(c[2])**2*math.sin(c[3])*math.sin(c[1]) - 5*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) - (41*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4 - (5*math.cos(c[3])*math.cos(c[1])*math.sin(c[2]))/2 - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) + (math.cos(HAtTBL)*math.cos(VAtTBL)*(math.sin(VAtTBL)*(5*math.cos(c[2]) - (5*math.cos(c[3])*math.cos(c[2]))/2 + (41*math.cos(c[2])*math.sin(c[3]))/8 - 5*math.cos(c[2])**3 + 5*math.cos(c[3])*math.cos(c[2])**3 - 5*math.cos(c[2])*math.cos(c[1])**2 - c[0]*math.sin(VAtTCR) + 5*math.cos(c[2])**3*math.cos(c[1])**2 + 5*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])**2 - (41*math.cos(c[2])*math.cos(c[1])**2*math.sin(c[3]))/4 - 5*math.cos(c[3])*math.cos(c[2])**3*math.cos(c[1])**2 - (41*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + (41*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + 5*math.cos(c[2])*math.cos(c[1])*math.sin(c[3])*math.sin(c[2])*math.sin(c[1])) + math.cos(VAtTBL)*math.sin(HAtTBL)*((41*math.cos(c[3])*math.cos(c[1]))/8 + (5*math.cos(c[1])*math.sin(c[3]))/2 + (41*math.cos(c[2])**2*math.cos(c[1]))/4 - (41*math.cos(c[2])**2*math.cos(c[1])**3)/4 - (5*math.cos(c[3])*math.sin(c[2])*math.sin(c[1]))/2 + (41*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))/8 - c[0]*math.cos(VAtTCR)*math.sin(HAtTCR) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1]))/4 - 5*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3)/4 + 5*math.cos(c[3])*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + 5*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1])) + math.cos(HAtTBL)*math.cos(VAtTBL)*((41*math.cos(c[3])*math.sin(c[1]))/8 + (5*math.sin(c[3])*math.sin(c[1]))/2 - (41*math.cos(c[1])*math.sin(c[3])*math.sin(c[2]))/8 - c[0]*math.cos(HAtTCR)*math.cos(VAtTCR) - 5*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[2])**2*math.sin(c[3])*math.sin(c[1]) + 5*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) + (41*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4 + (5*math.cos(c[3])*math.cos(c[1])*math.sin(c[2]))/2 + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4)))/(math.sin(VAtTBL)**2 + math.cos(HAtTBL)**2*math.cos(VAtTBL)**2 + math.cos(VAtTBL)**2*math.sin(HAtTBL)**2) + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4)**2 + ((5*math.cos(c[3])*math.cos(c[2]))/2 - 5*math.cos(c[2]) - (41*math.cos(c[2])*math.sin(c[3]))/8 + 5*math.cos(c[2])**3 - 5*math.cos(c[3])*math.cos(c[2])**3 + 5*math.cos(c[2])*math.cos(c[1])**2 + c[0]*math.sin(VAtTCR) - 5*math.cos(c[2])**3*math.cos(c[1])**2 - 5*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])**2 + (41*math.cos(c[2])*math.cos(c[1])**2*math.sin(c[3]))/4 + 5*math.cos(c[3])*math.cos(c[2])**3*math.cos(c[1])**2 + (math.sin(VAtTBL)*(math.sin(VAtTBL)*(5*math.cos(c[2]) - (5*math.cos(c[3])*math.cos(c[2]))/2 + (41*math.cos(c[2])*math.sin(c[3]))/8 - 5*math.cos(c[2])**3 + 5*math.cos(c[3])*math.cos(c[2])**3 - 5*math.cos(c[2])*math.cos(c[1])**2 - c[0]*math.sin(VAtTCR) + 5*math.cos(c[2])**3*math.cos(c[1])**2 + 5*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])**2 - (41*math.cos(c[2])*math.cos(c[1])**2*math.sin(c[3]))/4 - 5*math.cos(c[3])*math.cos(c[2])**3*math.cos(c[1])**2 - (41*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + (41*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + 5*math.cos(c[2])*math.cos(c[1])*math.sin(c[3])*math.sin(c[2])*math.sin(c[1])) + math.cos(VAtTBL)*math.sin(HAtTBL)*((41*math.cos(c[3])*math.cos(c[1]))/8 + (5*math.cos(c[1])*math.sin(c[3]))/2 + (41*math.cos(c[2])**2*math.cos(c[1]))/4 - (41*math.cos(c[2])**2*math.cos(c[1])**3)/4 - (5*math.cos(c[3])*math.sin(c[2])*math.sin(c[1]))/2 + (41*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))/8 - c[0]*math.cos(VAtTCR)*math.sin(HAtTCR) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1]))/4 - 5*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3)/4 + 5*math.cos(c[3])*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + 5*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1])) + math.cos(HAtTBL)*math.cos(VAtTBL)*((41*math.cos(c[3])*math.sin(c[1]))/8 + (5*math.sin(c[3])*math.sin(c[1]))/2 - (41*math.cos(c[1])*math.sin(c[3])*math.sin(c[2]))/8 - c[0]*math.cos(HAtTCR)*math.cos(VAtTCR) - 5*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[2])**2*math.sin(c[3])*math.sin(c[1]) + 5*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) + (41*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4 + (5*math.cos(c[3])*math.cos(c[1])*math.sin(c[2]))/2 + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4)))/(math.sin(VAtTBL)**2 + math.cos(HAtTBL)**2*math.cos(VAtTBL)**2 + math.cos(VAtTBL)**2*math.sin(HAtTBL)**2) + (41*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 - (41*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 - 5*math.cos(c[2])*math.cos(c[1])*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))**2)**(1/2) + (((41*math.cos(c[3])*math.cos(c[1]))/8 - (5*math.cos(c[1])*math.sin(c[3]))/2 + (41*math.cos(c[2])**2*math.cos(c[1]))/4 - (41*math.cos(c[2])**2*math.cos(c[1])**3)/4 + (5*math.cos(c[3])*math.sin(c[2])*math.sin(c[1]))/2 + (41*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))/8 + c[0]*math.cos(VAtTCR)*math.sin(HAtTCR) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1]))/4 + 5*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3)/4 - 5*math.cos(c[3])*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) - (math.cos(VAtTBR)*math.sin(HAtTBR)*(math.cos(VAtTBR)*math.sin(HAtTBR)*((41*math.cos(c[3])*math.cos(c[1]))/8 - (5*math.cos(c[1])*math.sin(c[3]))/2 + (41*math.cos(c[2])**2*math.cos(c[1]))/4 - (41*math.cos(c[2])**2*math.cos(c[1])**3)/4 + (5*math.cos(c[3])*math.sin(c[2])*math.sin(c[1]))/2 + (41*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))/8 + c[0]*math.cos(VAtTCR)*math.sin(HAtTCR) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1]))/4 + 5*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3)/4 - 5*math.cos(c[3])*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) - 5*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]) + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1])) - math.sin(VAtTBR)*(5*math.cos(c[2]) - (5*math.cos(c[3])*math.cos(c[2]))/2 - (41*math.cos(c[2])*math.sin(c[3]))/8 - 5*math.cos(c[2])**3 + 5*math.cos(c[3])*math.cos(c[2])**3 - 5*math.cos(c[2])*math.cos(c[1])**2 - c[0]*math.sin(VAtTCR) + 5*math.cos(c[2])**3*math.cos(c[1])**2 + 5*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])**2 + (41*math.cos(c[2])*math.cos(c[1])**2*math.sin(c[3]))/4 - 5*math.cos(c[3])*math.cos(c[2])**3*math.cos(c[1])**2 + (41*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 - (41*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + 5*math.cos(c[2])*math.cos(c[1])*math.sin(c[3])*math.sin(c[2])*math.sin(c[1])) + math.cos(HAtTBR)*math.cos(VAtTBR)*((41*math.cos(c[3])*math.sin(c[1]))/8 - (5*math.sin(c[3])*math.sin(c[1]))/2 - (41*math.cos(c[1])*math.sin(c[3])*math.sin(c[2]))/8 + c[0]*math.cos(HAtTCR)*math.cos(VAtTCR) + 5*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) + 5*math.cos(c[2])**2*math.sin(c[3])*math.sin(c[1]) - 5*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) + (41*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4 - (5*math.cos(c[3])*math.cos(c[1])*math.sin(c[2]))/2 - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4)))/(math.sin(VAtTBR)**2 + math.cos(HAtTBR)**2*math.cos(VAtTBR)**2 + math.cos(VAtTBR)**2*math.sin(HAtTBR)**2) - 5*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]) + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]))**2 + ((5*math.sin(c[3])*math.sin(c[1]))/2 - (41*math.cos(c[3])*math.sin(c[1]))/8 + (41*math.cos(c[1])*math.sin(c[3])*math.sin(c[2]))/8 - c[0]*math.cos(HAtTCR)*math.cos(VAtTCR) - 5*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[2])**2*math.sin(c[3])*math.sin(c[1]) + 5*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) - (41*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4 + (5*math.cos(c[3])*math.cos(c[1])*math.sin(c[2]))/2 + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) + (math.cos(HAtTBR)*math.cos(VAtTBR)*(math.cos(VAtTBR)*math.sin(HAtTBR)*((41*math.cos(c[3])*math.cos(c[1]))/8 - (5*math.cos(c[1])*math.sin(c[3]))/2 + (41*math.cos(c[2])**2*math.cos(c[1]))/4 - (41*math.cos(c[2])**2*math.cos(c[1])**3)/4 + (5*math.cos(c[3])*math.sin(c[2])*math.sin(c[1]))/2 + (41*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))/8 + c[0]*math.cos(VAtTCR)*math.sin(HAtTCR) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1]))/4 + 5*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3)/4 - 5*math.cos(c[3])*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) - 5*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]) + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1])) - math.sin(VAtTBR)*(5*math.cos(c[2]) - (5*math.cos(c[3])*math.cos(c[2]))/2 - (41*math.cos(c[2])*math.sin(c[3]))/8 - 5*math.cos(c[2])**3 + 5*math.cos(c[3])*math.cos(c[2])**3 - 5*math.cos(c[2])*math.cos(c[1])**2 - c[0]*math.sin(VAtTCR) + 5*math.cos(c[2])**3*math.cos(c[1])**2 + 5*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])**2 + (41*math.cos(c[2])*math.cos(c[1])**2*math.sin(c[3]))/4 - 5*math.cos(c[3])*math.cos(c[2])**3*math.cos(c[1])**2 + (41*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 - (41*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + 5*math.cos(c[2])*math.cos(c[1])*math.sin(c[3])*math.sin(c[2])*math.sin(c[1])) + math.cos(HAtTBR)*math.cos(VAtTBR)*((41*math.cos(c[3])*math.sin(c[1]))/8 - (5*math.sin(c[3])*math.sin(c[1]))/2 - (41*math.cos(c[1])*math.sin(c[3])*math.sin(c[2]))/8 + c[0]*math.cos(HAtTCR)*math.cos(VAtTCR) + 5*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) + 5*math.cos(c[2])**2*math.sin(c[3])*math.sin(c[1]) - 5*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) + (41*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4 - (5*math.cos(c[3])*math.cos(c[1])*math.sin(c[2]))/2 - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4)))/(math.sin(VAtTBR)**2 + math.cos(HAtTBR)**2*math.cos(VAtTBR)**2 + math.cos(VAtTBR)**2*math.sin(HAtTBR)**2) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4)**2 + (5*math.cos(c[2]) - (5*math.cos(c[3])*math.cos(c[2]))/2 - (41*math.cos(c[2])*math.sin(c[3]))/8 - 5*math.cos(c[2])**3 + 5*math.cos(c[3])*math.cos(c[2])**3 - 5*math.cos(c[2])*math.cos(c[1])**2 - c[0]*math.sin(VAtTCR) + 5*math.cos(c[2])**3*math.cos(c[1])**2 + 5*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])**2 + (41*math.cos(c[2])*math.cos(c[1])**2*math.sin(c[3]))/4 - 5*math.cos(c[3])*math.cos(c[2])**3*math.cos(c[1])**2 + (math.sin(VAtTBR)*(math.cos(VAtTBR)*math.sin(HAtTBR)*((41*math.cos(c[3])*math.cos(c[1]))/8 - (5*math.cos(c[1])*math.sin(c[3]))/2 + (41*math.cos(c[2])**2*math.cos(c[1]))/4 - (41*math.cos(c[2])**2*math.cos(c[1])**3)/4 + (5*math.cos(c[3])*math.sin(c[2])*math.sin(c[1]))/2 + (41*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))/8 + c[0]*math.cos(VAtTCR)*math.sin(HAtTCR) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1]))/4 + 5*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3)/4 - 5*math.cos(c[3])*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) - 5*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]) + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1])) - math.sin(VAtTBR)*(5*math.cos(c[2]) - (5*math.cos(c[3])*math.cos(c[2]))/2 - (41*math.cos(c[2])*math.sin(c[3]))/8 - 5*math.cos(c[2])**3 + 5*math.cos(c[3])*math.cos(c[2])**3 - 5*math.cos(c[2])*math.cos(c[1])**2 - c[0]*math.sin(VAtTCR) + 5*math.cos(c[2])**3*math.cos(c[1])**2 + 5*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])**2 + (41*math.cos(c[2])*math.cos(c[1])**2*math.sin(c[3]))/4 - 5*math.cos(c[3])*math.cos(c[2])**3*math.cos(c[1])**2 + (41*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 - (41*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + 5*math.cos(c[2])*math.cos(c[1])*math.sin(c[3])*math.sin(c[2])*math.sin(c[1])) + math.cos(HAtTBR)*math.cos(VAtTBR)*((41*math.cos(c[3])*math.sin(c[1]))/8 - (5*math.sin(c[3])*math.sin(c[1]))/2 - (41*math.cos(c[1])*math.sin(c[3])*math.sin(c[2]))/8 + c[0]*math.cos(HAtTCR)*math.cos(VAtTCR) + 5*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) + 5*math.cos(c[2])**2*math.sin(c[3])*math.sin(c[1]) - 5*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) + (41*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4 - (5*math.cos(c[3])*math.cos(c[1])*math.sin(c[2]))/2 - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4)))/(math.sin(VAtTBR)**2 + math.cos(HAtTBR)**2*math.cos(VAtTBR)**2 + math.cos(VAtTBR)**2*math.sin(HAtTBR)**2) + (41*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 - (41*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + 5*math.cos(c[2])*math.cos(c[1])*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))**2)**(1/2) + (((41*math.cos(c[3])*math.cos(c[1]))/8 - (5*math.cos(c[1])*math.sin(c[3]))/2 + (41*math.cos(c[2])**2*math.cos(c[1]))/4 - (41*math.cos(c[2])**2*math.cos(c[1])**3)/4 + (5*math.cos(c[3])*math.sin(c[2])*math.sin(c[1]))/2 + (41*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))/8 - c[0]*math.cos(VAtTCR)*math.sin(HAtTCR) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1]))/4 + 5*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3)/4 - 5*math.cos(c[3])*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + (math.cos(VAtTTL)*math.sin(HAtTTL)*(math.sin(VAtTTL)*(5*math.cos(c[2]) - (5*math.cos(c[3])*math.cos(c[2]))/2 - (41*math.cos(c[2])*math.sin(c[3]))/8 - 5*math.cos(c[2])**3 + 5*math.cos(c[3])*math.cos(c[2])**3 - 5*math.cos(c[2])*math.cos(c[1])**2 + c[0]*math.sin(VAtTCR) + 5*math.cos(c[2])**3*math.cos(c[1])**2 + 5*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])**2 + (41*math.cos(c[2])*math.cos(c[1])**2*math.sin(c[3]))/4 - 5*math.cos(c[3])*math.cos(c[2])**3*math.cos(c[1])**2 + (41*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 - (41*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + 5*math.cos(c[2])*math.cos(c[1])*math.sin(c[3])*math.sin(c[2])*math.sin(c[1])) - math.cos(VAtTTL)*math.sin(HAtTTL)*((41*math.cos(c[3])*math.cos(c[1]))/8 - (5*math.cos(c[1])*math.sin(c[3]))/2 + (41*math.cos(c[2])**2*math.cos(c[1]))/4 - (41*math.cos(c[2])**2*math.cos(c[1])**3)/4 + (5*math.cos(c[3])*math.sin(c[2])*math.sin(c[1]))/2 + (41*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))/8 - c[0]*math.cos(VAtTCR)*math.sin(HAtTCR) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1]))/4 + 5*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3)/4 - 5*math.cos(c[3])*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) - 5*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]) + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1])) + math.cos(HAtTTL)*math.cos(VAtTTL)*((5*math.sin(c[3])*math.sin(c[1]))/2 - (41*math.cos(c[3])*math.sin(c[1]))/8 + (41*math.cos(c[1])*math.sin(c[3])*math.sin(c[2]))/8 + c[0]*math.cos(HAtTCR)*math.cos(VAtTCR) - 5*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[2])**2*math.sin(c[3])*math.sin(c[1]) + 5*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) - (41*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4 + (5*math.cos(c[3])*math.cos(c[1])*math.sin(c[2]))/2 + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4)))/(math.sin(VAtTTL)**2 + math.cos(HAtTTL)**2*math.cos(VAtTTL)**2 + math.cos(VAtTTL)**2*math.sin(HAtTTL)**2) - 5*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]) + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]))**2 + ((5*math.sin(c[3])*math.sin(c[1]))/2 - (41*math.cos(c[3])*math.sin(c[1]))/8 + (41*math.cos(c[1])*math.sin(c[3])*math.sin(c[2]))/8 + c[0]*math.cos(HAtTCR)*math.cos(VAtTCR) - 5*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[2])**2*math.sin(c[3])*math.sin(c[1]) + 5*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) - (41*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4 + (5*math.cos(c[3])*math.cos(c[1])*math.sin(c[2]))/2 + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - (math.cos(HAtTTL)*math.cos(VAtTTL)*(math.sin(VAtTTL)*(5*math.cos(c[2]) - (5*math.cos(c[3])*math.cos(c[2]))/2 - (41*math.cos(c[2])*math.sin(c[3]))/8 - 5*math.cos(c[2])**3 + 5*math.cos(c[3])*math.cos(c[2])**3 - 5*math.cos(c[2])*math.cos(c[1])**2 + c[0]*math.sin(VAtTCR) + 5*math.cos(c[2])**3*math.cos(c[1])**2 + 5*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])**2 + (41*math.cos(c[2])*math.cos(c[1])**2*math.sin(c[3]))/4 - 5*math.cos(c[3])*math.cos(c[2])**3*math.cos(c[1])**2 + (41*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 - (41*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + 5*math.cos(c[2])*math.cos(c[1])*math.sin(c[3])*math.sin(c[2])*math.sin(c[1])) - math.cos(VAtTTL)*math.sin(HAtTTL)*((41*math.cos(c[3])*math.cos(c[1]))/8 - (5*math.cos(c[1])*math.sin(c[3]))/2 + (41*math.cos(c[2])**2*math.cos(c[1]))/4 - (41*math.cos(c[2])**2*math.cos(c[1])**3)/4 + (5*math.cos(c[3])*math.sin(c[2])*math.sin(c[1]))/2 + (41*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))/8 - c[0]*math.cos(VAtTCR)*math.sin(HAtTCR) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1]))/4 + 5*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3)/4 - 5*math.cos(c[3])*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) - 5*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]) + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1])) + math.cos(HAtTTL)*math.cos(VAtTTL)*((5*math.sin(c[3])*math.sin(c[1]))/2 - (41*math.cos(c[3])*math.sin(c[1]))/8 + (41*math.cos(c[1])*math.sin(c[3])*math.sin(c[2]))/8 + c[0]*math.cos(HAtTCR)*math.cos(VAtTCR) - 5*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[2])**2*math.sin(c[3])*math.sin(c[1]) + 5*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) - (41*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4 + (5*math.cos(c[3])*math.cos(c[1])*math.sin(c[2]))/2 + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4)))/(math.sin(VAtTTL)**2 + math.cos(HAtTTL)**2*math.cos(VAtTTL)**2 + math.cos(VAtTTL)**2*math.sin(HAtTTL)**2) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4)**2 + (5*math.cos(c[2]) - (5*math.cos(c[3])*math.cos(c[2]))/2 - (41*math.cos(c[2])*math.sin(c[3]))/8 - 5*math.cos(c[2])**3 + 5*math.cos(c[3])*math.cos(c[2])**3 - 5*math.cos(c[2])*math.cos(c[1])**2 + c[0]*math.sin(VAtTCR) + 5*math.cos(c[2])**3*math.cos(c[1])**2 + 5*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])**2 + (41*math.cos(c[2])*math.cos(c[1])**2*math.sin(c[3]))/4 - 5*math.cos(c[3])*math.cos(c[2])**3*math.cos(c[1])**2 - (math.sin(VAtTTL)*(math.sin(VAtTTL)*(5*math.cos(c[2]) - (5*math.cos(c[3])*math.cos(c[2]))/2 - (41*math.cos(c[2])*math.sin(c[3]))/8 - 5*math.cos(c[2])**3 + 5*math.cos(c[3])*math.cos(c[2])**3 - 5*math.cos(c[2])*math.cos(c[1])**2 + c[0]*math.sin(VAtTCR) + 5*math.cos(c[2])**3*math.cos(c[1])**2 + 5*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])**2 + (41*math.cos(c[2])*math.cos(c[1])**2*math.sin(c[3]))/4 - 5*math.cos(c[3])*math.cos(c[2])**3*math.cos(c[1])**2 + (41*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 - (41*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + 5*math.cos(c[2])*math.cos(c[1])*math.sin(c[3])*math.sin(c[2])*math.sin(c[1])) - math.cos(VAtTTL)*math.sin(HAtTTL)*((41*math.cos(c[3])*math.cos(c[1]))/8 - (5*math.cos(c[1])*math.sin(c[3]))/2 + (41*math.cos(c[2])**2*math.cos(c[1]))/4 - (41*math.cos(c[2])**2*math.cos(c[1])**3)/4 + (5*math.cos(c[3])*math.sin(c[2])*math.sin(c[1]))/2 + (41*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))/8 - c[0]*math.cos(VAtTCR)*math.sin(HAtTCR) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1]))/4 + 5*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3)/4 - 5*math.cos(c[3])*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) - 5*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]) + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1])) + math.cos(HAtTTL)*math.cos(VAtTTL)*((5*math.sin(c[3])*math.sin(c[1]))/2 - (41*math.cos(c[3])*math.sin(c[1]))/8 + (41*math.cos(c[1])*math.sin(c[3])*math.sin(c[2]))/8 + c[0]*math.cos(HAtTCR)*math.cos(VAtTCR) - 5*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[2])**2*math.sin(c[3])*math.sin(c[1]) + 5*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) - (41*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4 + (5*math.cos(c[3])*math.cos(c[1])*math.sin(c[2]))/2 + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4)))/(math.sin(VAtTTL)**2 + math.cos(HAtTTL)**2*math.cos(VAtTTL)**2 + math.cos(VAtTTL)**2*math.sin(HAtTTL)**2) + (41*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 - (41*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + 5*math.cos(c[2])*math.cos(c[1])*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))**2)**(1/2) + (((41*math.cos(c[3])*math.cos(c[1]))/8 + (5*math.cos(c[1])*math.sin(c[3]))/2 + (41*math.cos(c[2])**2*math.cos(c[1]))/4 - (41*math.cos(c[2])**2*math.cos(c[1])**3)/4 - (5*math.cos(c[3])*math.sin(c[2])*math.sin(c[1]))/2 + (41*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))/8 + c[0]*math.cos(VAtTCR)*math.sin(HAtTCR) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1]))/4 - 5*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3)/4 + 5*math.cos(c[3])*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) - (math.cos(VAtTTR)*math.sin(HAtTTR)*(math.sin(VAtTTR)*(5*math.cos(c[2]) - (5*math.cos(c[3])*math.cos(c[2]))/2 + (41*math.cos(c[2])*math.sin(c[3]))/8 - 5*math.cos(c[2])**3 + 5*math.cos(c[3])*math.cos(c[2])**3 - 5*math.cos(c[2])*math.cos(c[1])**2 + c[0]*math.sin(VAtTCR) + 5*math.cos(c[2])**3*math.cos(c[1])**2 + 5*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])**2 - (41*math.cos(c[2])*math.cos(c[1])**2*math.sin(c[3]))/4 - 5*math.cos(c[3])*math.cos(c[2])**3*math.cos(c[1])**2 - (41*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + (41*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + 5*math.cos(c[2])*math.cos(c[1])*math.sin(c[3])*math.sin(c[2])*math.sin(c[1])) + math.cos(VAtTTR)*math.sin(HAtTTR)*((41*math.cos(c[3])*math.cos(c[1]))/8 + (5*math.cos(c[1])*math.sin(c[3]))/2 + (41*math.cos(c[2])**2*math.cos(c[1]))/4 - (41*math.cos(c[2])**2*math.cos(c[1])**3)/4 - (5*math.cos(c[3])*math.sin(c[2])*math.sin(c[1]))/2 + (41*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))/8 + c[0]*math.cos(VAtTCR)*math.sin(HAtTCR) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1]))/4 - 5*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3)/4 + 5*math.cos(c[3])*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + 5*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1])) + math.cos(HAtTTR)*math.cos(VAtTTR)*((41*math.cos(c[3])*math.sin(c[1]))/8 + (5*math.sin(c[3])*math.sin(c[1]))/2 - (41*math.cos(c[1])*math.sin(c[3])*math.sin(c[2]))/8 + c[0]*math.cos(HAtTCR)*math.cos(VAtTCR) - 5*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[2])**2*math.sin(c[3])*math.sin(c[1]) + 5*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) + (41*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4 + (5*math.cos(c[3])*math.cos(c[1])*math.sin(c[2]))/2 + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4)))/(math.sin(VAtTTR)**2 + math.cos(HAtTTR)**2*math.cos(VAtTTR)**2 + math.cos(VAtTTR)**2*math.sin(HAtTTR)**2) + 5*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]))**2 + ((41*math.cos(c[3])*math.sin(c[1]))/8 + (5*math.sin(c[3])*math.sin(c[1]))/2 - (41*math.cos(c[1])*math.sin(c[3])*math.sin(c[2]))/8 + c[0]*math.cos(HAtTCR)*math.cos(VAtTCR) - 5*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[2])**2*math.sin(c[3])*math.sin(c[1]) + 5*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) + (41*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4 + (5*math.cos(c[3])*math.cos(c[1])*math.sin(c[2]))/2 + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - (math.cos(HAtTTR)*math.cos(VAtTTR)*(math.sin(VAtTTR)*(5*math.cos(c[2]) - (5*math.cos(c[3])*math.cos(c[2]))/2 + (41*math.cos(c[2])*math.sin(c[3]))/8 - 5*math.cos(c[2])**3 + 5*math.cos(c[3])*math.cos(c[2])**3 - 5*math.cos(c[2])*math.cos(c[1])**2 + c[0]*math.sin(VAtTCR) + 5*math.cos(c[2])**3*math.cos(c[1])**2 + 5*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])**2 - (41*math.cos(c[2])*math.cos(c[1])**2*math.sin(c[3]))/4 - 5*math.cos(c[3])*math.cos(c[2])**3*math.cos(c[1])**2 - (41*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + (41*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + 5*math.cos(c[2])*math.cos(c[1])*math.sin(c[3])*math.sin(c[2])*math.sin(c[1])) + math.cos(VAtTTR)*math.sin(HAtTTR)*((41*math.cos(c[3])*math.cos(c[1]))/8 + (5*math.cos(c[1])*math.sin(c[3]))/2 + (41*math.cos(c[2])**2*math.cos(c[1]))/4 - (41*math.cos(c[2])**2*math.cos(c[1])**3)/4 - (5*math.cos(c[3])*math.sin(c[2])*math.sin(c[1]))/2 + (41*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))/8 + c[0]*math.cos(VAtTCR)*math.sin(HAtTCR) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1]))/4 - 5*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3)/4 + 5*math.cos(c[3])*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + 5*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1])) + math.cos(HAtTTR)*math.cos(VAtTTR)*((41*math.cos(c[3])*math.sin(c[1]))/8 + (5*math.sin(c[3])*math.sin(c[1]))/2 - (41*math.cos(c[1])*math.sin(c[3])*math.sin(c[2]))/8 + c[0]*math.cos(HAtTCR)*math.cos(VAtTCR) - 5*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[2])**2*math.sin(c[3])*math.sin(c[1]) + 5*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) + (41*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4 + (5*math.cos(c[3])*math.cos(c[1])*math.sin(c[2]))/2 + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4)))/(math.sin(VAtTTR)**2 + math.cos(HAtTTR)**2*math.cos(VAtTTR)**2 + math.cos(VAtTTR)**2*math.sin(HAtTTR)**2) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4)**2 + (5*math.cos(c[2]) - (5*math.cos(c[3])*math.cos(c[2]))/2 + (41*math.cos(c[2])*math.sin(c[3]))/8 - 5*math.cos(c[2])**3 + 5*math.cos(c[3])*math.cos(c[2])**3 - 5*math.cos(c[2])*math.cos(c[1])**2 + c[0]*math.sin(VAtTCR) + 5*math.cos(c[2])**3*math.cos(c[1])**2 + 5*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])**2 - (41*math.cos(c[2])*math.cos(c[1])**2*math.sin(c[3]))/4 - 5*math.cos(c[3])*math.cos(c[2])**3*math.cos(c[1])**2 - (math.sin(VAtTTR)*(math.sin(VAtTTR)*(5*math.cos(c[2]) - (5*math.cos(c[3])*math.cos(c[2]))/2 + (41*math.cos(c[2])*math.sin(c[3]))/8 - 5*math.cos(c[2])**3 + 5*math.cos(c[3])*math.cos(c[2])**3 - 5*math.cos(c[2])*math.cos(c[1])**2 + c[0]*math.sin(VAtTCR) + 5*math.cos(c[2])**3*math.cos(c[1])**2 + 5*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])**2 - (41*math.cos(c[2])*math.cos(c[1])**2*math.sin(c[3]))/4 - 5*math.cos(c[3])*math.cos(c[2])**3*math.cos(c[1])**2 - (41*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + (41*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + 5*math.cos(c[2])*math.cos(c[1])*math.sin(c[3])*math.sin(c[2])*math.sin(c[1])) + math.cos(VAtTTR)*math.sin(HAtTTR)*((41*math.cos(c[3])*math.cos(c[1]))/8 + (5*math.cos(c[1])*math.sin(c[3]))/2 + (41*math.cos(c[2])**2*math.cos(c[1]))/4 - (41*math.cos(c[2])**2*math.cos(c[1])**3)/4 - (5*math.cos(c[3])*math.sin(c[2])*math.sin(c[1]))/2 + (41*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))/8 + c[0]*math.cos(VAtTCR)*math.sin(HAtTCR) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1]))/4 - 5*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3)/4 + 5*math.cos(c[3])*math.cos(c[2])**2*math.sin(c[2])*math.sin(c[1]) + 5*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1]) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[2])*math.sin(c[1])) + math.cos(HAtTTR)*math.cos(VAtTTR)*((41*math.cos(c[3])*math.sin(c[1]))/8 + (5*math.sin(c[3])*math.sin(c[1]))/2 - (41*math.cos(c[1])*math.sin(c[3])*math.sin(c[2]))/8 + c[0]*math.cos(HAtTCR)*math.cos(VAtTCR) - 5*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[2])**2*math.sin(c[3])*math.sin(c[1]) + 5*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) + (41*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4 + (5*math.cos(c[3])*math.cos(c[1])*math.sin(c[2]))/2 + 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])*math.sin(c[2]) - 5*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**3*math.sin(c[2]) - (41*math.cos(c[3])*math.cos(c[2])**2*math.cos(c[1])**2*math.sin(c[1]))/4)))/(math.sin(VAtTTR)**2 + math.cos(HAtTTR)**2*math.cos(VAtTTR)**2 + math.cos(VAtTTR)**2*math.sin(HAtTTR)**2) - (41*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + (41*math.cos(c[3])*math.cos(c[2])*math.cos(c[1])*math.sin(c[2])*math.sin(c[1]))/4 + 5*math.cos(c[2])*math.cos(c[1])*math.sin(c[3])*math.sin(c[2])*math.sin(c[1]))**2)


def cameraPosError(c):
    return ((math.cos(c[2])*(c[0] + math.cos(origRobotHeading)*(origRobotX - nextRobotX) - math.sin(origRobotHeading)*(origRobotZ - nextRobotZ) - c[0]*math.cos(origRobotHeading - nextRobotHeading) + c[1]*math.sin(origRobotHeading - nextRobotHeading)) - math.sin(c[2])*(c[1] + math.cos(origRobotHeading)*(origRobotZ - nextRobotZ) + math.sin(origRobotHeading)*(origRobotX - nextRobotX) - c[1]*math.cos(origRobotHeading - nextRobotHeading) - c[0]*math.sin(origRobotHeading - nextRobotHeading)) - math.cos(origCameraHoriz)*(origCameraX - nextCameraX) + math.sin(origCameraHoriz)*(origCameraZ - nextCameraZ))**2 + (math.cos(c[2])*(c[1] + math.cos(origRobotHeading)*(origRobotZ - nextRobotZ) + math.sin(origRobotHeading)*(origRobotX - nextRobotX) - c[1]*math.cos(origRobotHeading - nextRobotHeading) - c[0]*math.sin(origRobotHeading - nextRobotHeading)) + math.sin(c[2])*(c[0] + math.cos(origRobotHeading)*(origRobotX - nextRobotX) - math.sin(origRobotHeading)*(origRobotZ - nextRobotZ) - c[0]*math.cos(origRobotHeading - nextRobotHeading) + c[1]*math.sin(origRobotHeading - nextRobotHeading)) - math.cos(origCameraHoriz)*(origCameraZ - nextCameraZ) - math.sin(origCameraHoriz)*(origCameraX - nextCameraX))**2)**(1/2)


def rotateAboutAxis(a, b, c, u, v, w, x, y, z, theta):
    # rotateAboutAxis rotates a point (x,y,z) in 3D space an angle theta about a line with unit directional vector
    # (u,v,w) that passes through point (a,b,c)

    a = float(a)
    b = float(b)
    c = float(c)
    u = float(u)
    v = float(v)
    w = float(w)
    x = float(x)
    y = float(y)
    z = float(z)
    theta = float(theta)

    x1 = (a * (pow(v, 2) + pow(w, 2)) - u * (b * v + c * w - u * x - v * y - w * z)) * (1 - math.cos(theta)) + x * math.cos(theta) + (-c * v + b * w - w * y + v * z) * math.sin(theta);
    y1 = (b * (pow(u, 2) + pow(w, 2)) - v * (a * u + c * w - u * x - v * y - w * z)) * (1 - math.cos(theta)) + y * math.cos(theta) + (c * u - a * w + w * x - u * z) * math.sin(theta);
    z1 = (c * (pow(u, 2) + pow(v, 2)) - w * (a * u + b * v - u * x - v * y - w * z)) * (1 - math.cos(theta)) + z * math.cos(theta) + (-b * u + a * v - v * x + u * y) * math.sin(theta);

    return x1, y1, z1


def locateTarget(pointAngles):
    # locateTarget takes the horizontal and vertical angles to different points on a known target and calculates the
    # x, y, and z coordinates of the camera in relation to the target, as well as the horizontal and vertical heading
    # of the camera, and the tilt along that axis to boot.

    # Unpacking the input array into individual variables
    # The first letter either represents "horizontal" or "vertical"
    # The last two letters are decoded as follows:
    # CR = "center"
    # TL = "top left"
    # TR = "top right"
    # BL = "bottom left"
    # BR = "bottom right"
    # So the whole variable is decoded: *horizontal or vertical* angle to target *point on target*

    HAtTCR = pointAngles[0]
    VAtTCR = pointAngles[1]
    HAtTTL = pointAngles[2]
    VAtTTL = pointAngles[3]
    HAtTTR = pointAngles[4]
    VAtTTR = pointAngles[5]
    HAtTBL = pointAngles[6]
    VAtTBL = pointAngles[7]
    HAtTBR = pointAngles[8]
    VAtTBR = pointAngles[9]

    # Next we define the target in 3D space so that the center is (0, 0, 0) and the z-axis is pointing outward.
    TLx0 = float(-10.25 / 2.0)
    TLy0 = float(5.0 / 2.0)
    TLz0 = float(0.0)
    TRx0 = float(10.25 / 2.0)
    TRy0 = float(5.0 / 2.0)
    TRz0 = float(0.0)
    BLx0 = float(-10.25 / 2.0)
    BLy0 = float(-5.0 / 2.0)
    BLz0 = float(0.0)
    BRx0 = float(10.25 / 2.0)
    BRy0 = float(-5.0 / 2.0)
    BRz0 = float(0.0)

    # Now we define our search area in terms of the distance from the camera to the target, the rotation about the y
    # vertical) axis in relation to the camera, the x (horizontal sideways) axis, and finally the tilt about its
    # viewing axis. Angle measurements are in degrees and distances are in inches.

    distCRMin = 20
    distCRMax = 30
    yRotMin = -90
    yRotMax = 90
    xRotMin = -45
    xRotMax = 45
    tiltMin = -10
    tiltMax = 10

    # Define the step size
    distCRStep = 5.0
    yRotStep = 5.0
    xRotStep = 5.0
    tiltStep = 5.0

    # Converting all the angles to radians because python only uses radians in trigonometry, also making them integers
    distCRMin = int(round(distCRMin * 1000000))
    distCRMax = int(round(distCRMax * 1000000))
    yRotMin = int(round(math.radians(yRotMin) * 1000000))
    yRotMax = int(round(math.radians(yRotMax) * 1000000))
    xRotMin = int(round(math.radians(xRotMin) * 1000000))
    xRotMax = int(round(math.radians(xRotMax) * 1000000))
    tiltMin = int(round(math.radians(tiltMin) * 1000000))
    tiltMax = int(round(math.radians(tiltMax) * 1000000))

    distCRStep = int(round(distCRStep * 1000000))
    yRotStep = int(round(math.radians(yRotStep) * 1000000))
    xRotStep = int(round(math.radians(xRotStep) * 1000000))
    tiltStep = int(round(math.radians(tiltStep) * 1000000))

    # Just defining some things relating to error and stuff

    minError = 100000

    bestDistCR = 0
    bestYRot = 0
    bestXRot = 0
    bestTilt = 0

    # Next, we run through a series of for loops to minimize the error function by brute force, because we can and
    # we're not worried about time.

    for distCR in range(distCRMin, distCRMax + distCRStep, distCRStep):
        distCR = float(distCR) / 1000000.0

        # Lastly, we put it out in space, along with the center point
        CRx = math.cos(VAtTCR) * math.sin(HAtTCR) * distCR
        CRy = math.sin(VAtTCR) * distCR
        CRz = math.cos(VAtTCR) * math.cos(HAtTCR) * distCR

        for yRot in range(yRotMin, yRotMax + yRotStep, yRotStep):
            yRot = float(yRot) / 1000000.0

            # First we rotate about the y-axis
            TLx1, TLy1, TLz1 = rotateAboutAxis(0, 0, 0, 0, 1, 0, TLx0, TLy0, TLz0, yRot)
            TRx1, TRy1, TRz1 = rotateAboutAxis(0, 0, 0, 0, 1, 0, TRx0, TRy0, TRz0, yRot)
            BLx1, BLy1, BLz1 = rotateAboutAxis(0, 0, 0, 0, 1, 0, BLx0, BLy0, BLz0, yRot)
            BRx1, BRy1, BRz1 = rotateAboutAxis(0, 0, 0, 0, 1, 0, BRx0, BRy0, BRz0, yRot)

            for xRot in range(xRotMin, xRotMax + xRotStep, xRotStep):
                xRot = float(xRot) / 1000000.0

                # Now about the target's x-axis
                TLx2, TLy2, TLz2 = rotateAboutAxis(0, 0, 0, -math.cos(yRot), 0, math.sin(yRot), TLx1, TLy1, TLz1, xRot)
                TRx2, TRy2, TRz2 = rotateAboutAxis(0, 0, 0, -math.cos(yRot), 0, math.sin(yRot), TRx1, TRy1, TRz1, xRot)
                BLx2, BLy2, BLz2 = rotateAboutAxis(0, 0, 0, -math.cos(yRot), 0, math.sin(yRot), BLx1, BLy1, BLz1, xRot)
                BRx2, BRy2, BRz2 = rotateAboutAxis(0, 0, 0, -math.cos(yRot), 0, math.sin(yRot), BRx1, BRy1, BRz1, xRot)

                for tilt in range(tiltMin, tiltMax + tiltStep, tiltStep):
                    # Dividing by 1000000 beacuse that's how things work

                    tilt = float(tilt) / 1000000.0

                    print "DistCR: " + str(distCR) + " yRot: " + str(math.degrees(yRot)) + " xRot: " + str(math.degrees(xRot)) + " tilt: " + str(math.degrees(tilt))

                    # Next is the tilt about the target's normal vector
                    TLx3, TLy3, TLz3 = rotateAboutAxis(0, 0, 0, math.sin(yRot)*math.cos(xRot), -math.sin(xRot), -math.cos(yRot)*math.cos(xRot), TLx2, TLy2, TLz2, tilt)
                    TRx3, TRy3, TRz3 = rotateAboutAxis(0, 0, 0, math.sin(yRot)*math.cos(xRot), -math.sin(xRot), -math.cos(yRot)*math.cos(xRot), TRx2, TRy2, TRz2, tilt)
                    BLx3, BLy3, BLz3 = rotateAboutAxis(0, 0, 0, math.sin(yRot)*math.cos(xRot), -math.sin(xRot), -math.cos(yRot)*math.cos(xRot), BLx2, BLy2, BLz2, tilt)
                    BRx3, BRy3, BRz3 = rotateAboutAxis(0, 0, 0, math.sin(yRot)*math.cos(xRot), -math.sin(xRot), -math.cos(yRot)*math.cos(xRot), BRx2, BRy2, BRz2, tilt)

                    #print "TL: " + str(TLx3) + ", " + str(TLy3) + ", " + str(TLz3)
                    #print "TR: " + str(TRx3) + ", " + str(TRy3) + ", " + str(TRz3)
                    #print "BL: " + str(BLx3) + ", " + str(BLy3) + ", " + str(BLz3)
                    #print "BR: " + str(BRx3) + ", " + str(BRy3) + ", " + str(BRz3)

                    TLx = TLx3 + CRx
                    TLy = TLy3 + CRy
                    TLz = TLz3 - CRz
                    TRx = TRx3 + CRx
                    TRy = TRy3 + CRy
                    TRz = TRz3 - CRz
                    BLx = BLx3 + CRx
                    BLy = BLy3 + CRy
                    BLz = BLz3 - CRz
                    BRx = BRx3 + CRx
                    BRy = BRy3 + CRy
                    BRz = BRz3 - CRz

                    #print "TL: " + str(TLx) + ", " + str(TLy) + ", " + str(TLz)
                    #print "TR: " + str(TRx) + ", " + str(TRy) + ", " + str(TRz)
                    #print "BL: " + str(BLx) + ", " + str(BLy) + ", " + str(BLz)
                    #print "BR: " + str(BRx) + ", " + str(BRy) + ", " + str(BRz)

                    # Now it's time to calculate the error of our guess: the distance from the corners to their
                    # respective "lines" coming out of the camera by cross product.

                    TLCross = numpy.cross([math.cos(VAtTTL) * math.sin(HAtTTL), math.sin(VAtTTL), math.cos(VAtTTL) * math.cos(HAtTTL)], [TLx, TLy, -TLz])
                    errorTL = math.sqrt(pow(TLCross[0], 2) + pow(TLCross[1], 2) + pow(TLCross[2], 2))

                    TRCross = numpy.cross([math.cos(VAtTTR) * math.sin(HAtTTR), math.sin(VAtTTR), math.cos(VAtTTR) * math.cos(HAtTTR)], [TRx, TRy, -TRz])
                    errorTR = math.sqrt(pow(TRCross[0], 2) + pow(TRCross[1], 2) + pow(TRCross[2], 2))

                    BLCross = numpy.cross([math.cos(VAtTBL) * math.sin(HAtTBL), math.sin(VAtTBL), math.cos(VAtTBL) * math.cos(HAtTBL)], [BLx, BLy, -BLz])
                    errorBL = math.sqrt(pow(BLCross[0], 2) + pow(BLCross[1], 2) + pow(BLCross[2], 2))

                    BRCross = numpy.cross([math.cos(VAtTBR) * math.sin(HAtTBR), math.sin(VAtTBR), math.cos(VAtTBR) * math.cos(HAtTBR)], [BRx, BRy, -BRz])
                    errorBR = math.sqrt(pow(BRCross[0], 2) + pow(BRCross[1], 2) + pow(BRCross[2], 2))

                    error = errorTL + errorTR + errorBL + errorBR

                    # If the calculated error is less than the previous best guess's error, then we store it and the guess

                    #print error

                    if error < minError:
                        minError = error
                        bestDistCR = distCR
                        bestYRot = yRot
                        bestXRot = xRot
                        bestTilt = tilt
                        #print minError

    result = optimize.minimize(targetError, [bestDistCR, bestYRot, bestXRot, bestTilt])
    distCR = result.x[0]
    yRot   = result.x[1]
    xRot   = result.x[2]
    tilt   = result.x[3]
    print result
    print [bestDistCR, bestYRot, bestXRot, bestTilt]
    return distCR, yRot, xRot, tilt


def locateCamera(goalOrientation, goalAngles, heightOfTarget):
    # locateCamera takes the distance to the center of the target, the y-rotation, the x-rotation, and the tilt and
    # calculates the camera position relative to the goal.

    # Unpacking the input arrays

    distCR = goalOrientation[0]
    yRot = goalOrientation[1]
    xRot = goalOrientation[2]
    tilt = goalOrientation[3]

    HAtTCR = goalAngles[0]
    VAtTCR = goalAngles[1]

    # Calculating where the center of the target is in relation to the camera

    CRx = math.cos(VAtTCR) * math.sin(HAtTCR) * distCR
    CRy = math.sin(VAtTCR) * distCR
    CRz = math.cos(VAtTCR) * math.cos(HAtTCR) * distCR

    # Setting up a coordinate system for the target

    CRRightx, CRRighty, CRRightz = rotateAboutAxis(0, 0, 0, 0, 1, 0, 1, 0, 0, yRot)
    CRUpx,    CRUpy,    CRUpz    = rotateAboutAxis(0, 0, 0, 0, 1, 0, 0, 1, 0, yRot)
    CRRightx, CRRighty, CRRightz = rotateAboutAxis(0, 0, 0, -math.cos(yRot), 0, math.sin(yRot), CRRightx, CRRighty, CRRightz, xRot)
    CRUpx,    CRUpy,    CRUpz    = rotateAboutAxis(0, 0, 0, -math.cos(yRot), 0, math.sin(yRot), CRUpx, CRUpy, CRUpz, xRot)
    CRRightx, CRRighty, CRRightz = rotateAboutAxis(0, 0, 0, math.sin(yRot)*math.cos(xRot), -math.sin(xRot), -math.cos(yRot)*math.cos(xRot), CRRightx, CRRighty, CRRightz, tilt)
    CRUpx,    CRUpy,    CRUpz    = rotateAboutAxis(0, 0, 0, math.sin(yRot)*math.cos(xRot), -math.sin(xRot), -math.cos(yRot)*math.cos(xRot), CRUpx, CRUpy, CRUpz, tilt)

    ## Calculating the 3D coordinates of the camera

    # Initializing the 3D coordinates of the camera

    camerax = 0
    cameray = 0
    cameraz = 0
    cameraPointx = 0
    cameraPointy = 0
    cameraPointz = -1
    cameraRightx = 1
    cameraRighty = 0
    cameraRightz = 0

    camerax = camerax - CRx
    cameray = cameray - CRy
    cameraz = cameraz + CRz
    cameraPointx = camerax + cameraPointx
    cameraPointy = cameray + cameraPointy
    cameraPointz = cameraz + cameraPointz
    cameraRightx = camerax + cameraRightx
    cameraRighty = cameray + cameraRighty
    cameraRightz = cameraz + cameraRightz

    CRx = 0
    CRy = 0
    CRz = 0

    # Rotating the camera about the y and x axis, then about the z axis.

    targetNormx = (CRRighty*CRUpz - CRUpy*CRRightz)
    targetNormy = -(CRRightx*CRUpz - CRUpx*CRRightz)
    targetNormz = (CRRightx*CRUpy - CRUpx*CRRighty)

    targCross = numpy.cross([targetNormx, targetNormy, targetNormz], [0, 0, 1])
    targCrossLen = math.sqrt(pow(targCross[1], 2) + pow(targCross[2], 2) + pow(targCross[2], 2))

    camerax, cameray, cameraz = rotateAboutAxis(0, 0, 0, targCross[0] / targCrossLen, targCross[1] / targCrossLen, targCross[2] / targCrossLen, camerax, cameray, cameraz, math.asin(targCrossLen))
    cameraPointx, cameraPointy, cameraPointz = rotateAboutAxis(0, 0, 0, targCross[0] / targCrossLen, targCross[1] / targCrossLen, targCross[2] / targCrossLen, cameraPointx, cameraPointy, cameraPointz, math.asin(targCrossLen))
    cameraRightx, cameraRighty, cameraRightz = rotateAboutAxis(0, 0, 0, targCross[0] / targCrossLen, targCross[1] / targCrossLen, targCross[2] / targCrossLen, cameraRightx, cameraRighty, cameraRightz, math.asin(targCrossLen))
    CRUpx, CRUpy, CRUpz = rotateAboutAxis(0, 0, 0, targCross[0] / targCrossLen, targCross[1] / targCrossLen, targCross[2] / targCrossLen, CRUpx, CRUpy, CRUpz, math.asin(targCrossLen))

    targCross = numpy.cross([CRUpx, CRUpy, CRUpz], [0, 1, 0])
    targCrossLen = math.sqrt(pow(targCross[1], 2) + pow(targCross[2], 2) + pow(targCross[2], 2))

    camerax, cameray, cameraz = rotateAboutAxis(0, 0, 0, targCross[0] / targCrossLen, targCross[1] / targCrossLen, targCross[2] / targCrossLen, camerax, cameray, cameraz, math.asin(targCrossLen))
    cameraPointx, cameraPointy, cameraPointz = rotateAboutAxis(0, 0, 0, targCross[0] / targCrossLen, targCross[1] / targCrossLen, targCross[2] / targCrossLen, cameraPointx, cameraPointy, cameraPointz, math.asin(targCrossLen))
    cameraRightx, cameraRighty, cameraRightz = rotateAboutAxis(0, 0, 0, targCross[0] / targCrossLen, targCross[1] / targCrossLen, targCross[2] / targCrossLen, cameraRightx, cameraRighty, cameraRightz, math.asin(targCrossLen))
    CRUpx, CRUpy, CRUpz = rotateAboutAxis(0, 0, 0, targCross[0] / targCrossLen, targCross[1] / targCrossLen, targCross[2] / targCrossLen, CRUpx, CRUpy, CRUpz, math.asin(targCrossLen))

    cameraPointx = cameraPointx - camerax
    cameraPointy = cameraPointy - cameray
    cameraPointz = cameraPointz - cameraz

    cameraRightx = cameraRightx - camerax
    cameraRighty = cameraRighty - cameray
    cameraRightz = cameraRightz - cameraz

    # Calculating the position and heading of the camera in relation to the target (off the ground)

    cameraX = camerax

    cameraY = cameray + heightOfTarget

    cameraZ = cameraz

    cameraHorizontal = -math.atan(cameraPointx / cameraPointz)

    cameraVertical = math.atan(cameraPointy / (math.sqrt(pow(cameraPointx, 2) + pow(cameraPointz, 2))))

    crossThingy = numpy.cross([cameraPointx, cameraPointy, cameraPointz], [0, 1, 0])
    otherCrossThingy = numpy.cross([cameraRightx, cameraRighty, cameraRightz], [crossThingy[0], crossThingy[1], crossThingy[2]])
    cameraTilt = math.asin(math.sqrt(pow(otherCrossThingy[0], 2) + pow(otherCrossThingy[1], 2) + pow(otherCrossThingy[2], 2)))

    return cameraX, cameraY, cameraZ, cameraHorizontal, cameraVertical, cameraTilt


def calculateCameraPositionOnRobot(origRobotPos, origCameraPos, nextRobotPos, nextCameraPos):
    # calculateCameraPositionOnRobot calculates the X, Y, Z, horizontal heading, vertical heading, and tilt of a camera
    # on the robot relative to the center of rotation, the floor, and the front of the robot. It utilizes the X, Y, Z,
    # horizontal heading, vertical heading, and tilt of the camera in two positions, along with the X, Z, and horizontal
    # heading of the robot (acquired through gyro and wheel encoders) at the same time the camera position was calculated.

    cameraPosOnRobotX = 0
    cameraPosOnRobotY = 0
    cameraPosOnRobotZ = 0
    cameraPosOnRobotHoriz = 0
    cameraPosOnRobotVert = 0
    cameraPosOnRobotTilt = 0

    # Unpacking the input arrays

    global origRobotX
    global origRobotZ
    global origRobotHeading

    global nextRobotX
    global nextRobotZ
    global nextRobotHeading

    origRobotX = float(origRobotPos[0])
    origRobotZ = float(origRobotPos[1])
    origRobotHeading = float(origRobotPos[2])

    nextRobotX = float(nextRobotPos[0])
    nextRobotZ = float(nextRobotPos[1])
    nextRobotHeading = float(nextRobotPos[2])

    global origCameraX
    global origCameraY
    global origCameraZ
    global origCameraHoriz
    global origCameraVert
    global origCameraTilt

    global nextCameraX
    global nextCameraY
    global nextCameraZ
    global nextCameraHoriz
    global nextCameraVert
    global nextCameraTilt

    origCameraX = float(origCameraPos[0])
    origCameraY = float(origCameraPos[1])
    origCameraZ = float(origCameraPos[2])
    origCameraHoriz = float(origCameraPos[3])
    origCameraVert = float(origCameraPos[4])
    origCameraTilt = float(origCameraPos[5])

    nextCameraX = float(nextCameraPos[0])
    nextCameraY = float(nextCameraPos[1])
    nextCameraZ = float(nextCameraPos[2])
    nextCameraHoriz = float(nextCameraPos[3])
    nextCameraVert = float(nextCameraPos[4])
    nextCameraTilt = float(nextCameraPos[5])

    cameraPosOnRobotY = (origCameraY + nextCameraY) / 2
    cameraPosOnRobotVert = (origCameraVert + nextCameraVert) / 2
    cameraPosOnRobotTilt = (origCameraTilt + nextCameraTilt) / 2

    xLeft = -18
    xStep = 1
    xRight = 18
    zFront = 18
    zStep = 1
    zBack = -18
    headingLeft = math.radians(-90)
    headingStep = math.radians(1)
    headingRight = math.radians(90)

    xLeft = int(xLeft * 1000000)
    xStep = int(xStep * 1000000)
    xRight = int(xRight * 1000000)
    zFront = int(zFront * 1000000)
    zStep = int(zStep * 1000000)
    zBack = int(zBack * 1000000)
    headingLeft = int(headingLeft * 1000000)
    headingStep = int(headingStep * 1000000)
    headingRight = int(headingRight * 1000000)

    minError = float('inf')

    for xPos in range(xLeft, xRight + xStep, xStep):
        xPos = float(xPos) / 1000000
        for zPos in range(zBack, zFront + zStep, zStep):
            zPos = float(zPos) / 1000000
            for relCameraHeading in range(headingLeft, headingRight + headingStep, headingStep):
                relCameraHeading = float(relCameraHeading) / 1000000

                # Get first robot pose to be(0, 0, 0) and second robot pose to be in the same relative position.

                firstRobotPoseX = 0
                firstRobotPoseZ = 0
                firstRobotPoseHeading = 0

                secondRobotPoseX = nextRobotX - origRobotX
                secondRobotPoseZ = nextRobotZ - origRobotZ

                secondRobotPoseX, _, secondRobotPoseZ = rotateAboutAxis(0, 0, 0, 0, 1, 0, secondRobotPoseX, 0, secondRobotPoseZ, origRobotHeading)

                secondRobotPoseHeading = nextRobotHeading - origRobotHeading

                #print str(secondRobotPoseX) + ", " + str(secondRobotPoseZ) + ", " + str(math.degrees(secondRobotPoseHeading))

                # Apply the guess to each robot pose to find tentative camera positions

                firstCameraPoseGuessHeading = relCameraHeading
                firstCameraPoseGuessX = xPos
                firstCameraPoseGuessZ = -zPos

                secondCameraPoseGuessHeading = secondRobotPoseHeading + relCameraHeading
                secondCameraPoseGuessX = secondRobotPoseX + (xPos * math.cos(secondRobotPoseHeading) + zPos * math.sin(secondRobotPoseHeading))
                secondCameraPoseGuessZ = secondRobotPoseZ - (-xPos * math.sin(secondRobotPoseHeading) + zPos * math.cos(secondRobotPoseHeading))

                #print str(firstCameraPoseGuessX) + ", " + str(firstCameraPoseGuessZ) + ", " + str(math.degrees(firstCameraPoseGuessHeading))
                #print str(secondCameraPoseGuessX) + ", " + str(secondCameraPoseGuessZ) + ", " + str(math.degrees(secondCameraPoseGuessHeading))

                # Get first tentative camera pose to be (0, 0, 0) and second tentative camera pose to be in the same relative position

                secondCameraPoseGuessX = secondCameraPoseGuessX - firstCameraPoseGuessX
                secondCameraPoseGuessZ = secondCameraPoseGuessZ - firstCameraPoseGuessZ
                secondCameraPoseGuessX, _, secondCameraPoseGuessZ = rotateAboutAxis(0, 0, 0, 0, 1, 0, secondCameraPoseGuessX, 0, secondCameraPoseGuessZ, firstCameraPoseGuessHeading)
                secondCameraPoseGuessHeading = secondCameraPoseGuessHeading - firstCameraPoseGuessHeading

                # Get first actual camera pose to be (0, 0, 0) and second actual camera pose to be in the same relative position

                secondCameraPoseX = nextCameraX - origCameraX
                secondCameraPoseZ = nextCameraZ - origCameraZ
                secondCameraPoseHeading = nextCameraHoriz - origCameraHoriz
                secondCameraPoseX, _, secondCameraPoseZ = rotateAboutAxis(0, 0, 0, 0, 1, 0, secondCameraPoseX, 0, secondCameraPoseZ, origCameraHoriz)

                # Calculate the distance from the second tentative camera position to the second actual camera position

                error = math.sqrt(pow((secondCameraPoseGuessX - secondCameraPoseX), 2) + pow((secondCameraPoseGuessZ - secondCameraPoseZ), 2))

                if error < minError:
                    minError = error
                    cameraPosOnRobotX = xPos
                    cameraPosOnRobotZ = zPos
                    cameraPosOnRobotHoriz = relCameraHeading

    result = optimize.minimize(cameraPosError, [cameraPosOnRobotX, cameraPosOnRobotZ, cameraPosOnRobotHoriz])
    cameraPosOnRobotX = result.x[0]
    cameraPosOnRobotZ = result.x[1]
    cameraPosOnRobotHoriz = result.x[2]
    print result
    print [cameraPosOnRobotX, cameraPosOnRobotZ, cameraPosOnRobotHoriz]

    return cameraPosOnRobotX, cameraPosOnRobotY, cameraPosOnRobotZ, cameraPosOnRobotHoriz, cameraPosOnRobotVert, cameraPosOnRobotTilt

# Need an array (rawRobotCamArray) in the following form:
# Position 1: [[HAtTCR, VAtTCR, HAtTTL, VAtTTL, HAtTTR, VAtTTR, HAtTBL, VAtTBL, HAtTBR, VAtTBR, robotPosX, robotPosZ, robotHeading],
# Position 2: [[HAtTCR, VAtTCR, HAtTTL, VAtTTL, HAtTTR, VAtTTR, HAtTBL, VAtTBL, HAtTBR, VAtTBR, robotPosX, robotPosZ, robotHeading],
# Position 3: [[HAtTCR, VAtTCR, HAtTTL, VAtTTL, HAtTTR, VAtTTR, HAtTBL, VAtTBL, HAtTBR, VAtTBR, robotPosX, robotPosZ, robotHeading],
# ...
# Position n: [[HAtTCR, VAtTCR, HAtTTL, VAtTTL, HAtTTR, VAtTTR, HAtTBL, VAtTBL, HAtTBR, VAtTBR, robotPosX, robotPosZ, robotHeading]]

rawRobotCamArray = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

#print rawRobotCamArray[0][12]

robotCamPosArray = numpy.empty([len(rawRobotCamArray), 9])

camPosArray = numpy.empty([len(rawRobotCamArray)-1, 6])

for position in range(0, len(rawRobotCamArray)):
    HAtTCR = rawRobotCamArray[position][0]
    VAtTCR = rawRobotCamArray[position][1]
    HAtTTL = rawRobotCamArray[position][2]
    VAtTTL = rawRobotCamArray[position][3]
    HAtTTR = rawRobotCamArray[position][4]
    VAtTTR = rawRobotCamArray[position][5]
    HAtTBL = rawRobotCamArray[position][6]
    VAtTBL = rawRobotCamArray[position][7]
    HAtTBR = rawRobotCamArray[position][8]
    VAtTBR = rawRobotCamArray[position][9]



    # Calculate the position of the camera for each position and put it into a new array

    distCR, yRot, xRot, tilt = locateTarget([HAtTCR, VAtTCR, HAtTTL, VAtTTL, HAtTTR, VAtTTR, HAtTBL, VAtTBL, HAtTBR, VAtTBR])
    cameraX, cameraY, cameraZ, cameraHorizontal, cameraVertical, cameraTilt = locateCamera([distCR, yRot, xRot, tilt], [HAtTCR, VAtTCR], 13.25)

    robotCamPosArray[position][0] = cameraX
    robotCamPosArray[position][1] = cameraY
    robotCamPosArray[position][2] = cameraZ
    robotCamPosArray[position][3] = cameraHorizontal
    robotCamPosArray[position][4] = cameraVertical
    robotCamPosArray[position][5] = cameraTilt
    robotCamPosArray[position][6] = rawRobotCamArray[position][10]
    robotCamPosArray[position][7] = rawRobotCamArray[position][11]
    robotCamPosArray[position][8] = rawRobotCamArray[position][12]

for position in range(0, len(robotCamPosArray)-1):
    firstCameraX = robotCamPosArray[position][0]
    firstCameraY = robotCamPosArray[position][1]
    firstCameraZ = robotCamPosArray[position][2]
    firstCameraHorizontal = robotCamPosArray[position][3]
    firstCameraVertical = robotCamPosArray[position][4]
    firstCameraTilt = robotCamPosArray[position][5]
    firstRobotPosX = robotCamPosArray[position][6]
    firstRobotPosZ = robotCamPosArray[position][7]
    firstRobotHeading = robotCamPosArray[position][8]

    secondCameraX = robotCamPosArray[position+1][0]
    secondCameraY = robotCamPosArray[position+1][1]
    secondCameraZ = robotCamPosArray[position+1][2]
    secondCameraHorizontal = robotCamPosArray[position+1][3]
    secondCameraVertical = robotCamPosArray[position+1][4]
    secondCameraTilt = robotCamPosArray[position+1][5]
    secondRobotPosX = robotCamPosArray[position+1][6]
    secondRobotPosZ = robotCamPosArray[position+1][7]
    secondRobotHeading = robotCamPosArray[position+1][8]

    # Calculating the camera's position on the robot based on the two positions' data

    cameraPosOnRobotX, cameraPosOnRobotY, cameraPosOnRobotZ, cameraPosOnRobotHoriz, cameraPosOnRobotVert, cameraPosOnRobotTilt = calculateCameraPositionOnRobot(
        [firstRobotPosX,  firstRobotPosZ,  firstRobotHeading],  [firstCameraX,  firstCameraY,  firstCameraZ,  firstCameraHorizontal,  firstCameraVertical,  firstCameraTilt],
        [secondRobotPosX, secondRobotPosZ, secondRobotHeading], [secondCameraX, secondCameraY, secondCameraZ, secondCameraHorizontal, secondCameraVertical, secondCameraTilt])

    camPosArray[position][0] = cameraPosOnRobotX
    camPosArray[position][1] = cameraPosOnRobotY
    camPosArray[position][2] = cameraPosOnRobotZ
    camPosArray[position][3] = cameraPosOnRobotHoriz
    camPosArray[position][4] = cameraPosOnRobotVert
    camPosArray[position][5] = cameraPosOnRobotTilt

print camPosArray
print camPosArray[0]

cameraPosOnRobotX     = numpy.mean(camPosArray[:, 0])
cameraPosOnRobotY     = numpy.mean(camPosArray[:, 1])
cameraPosOnRobotZ     = numpy.mean(camPosArray[:, 2])
cameraPosOnRobotHoriz = numpy.mean(camPosArray[:, 3])
cameraPosOnRobotVert  = numpy.mean(camPosArray[:, 4])
cameraPosOnRobotTilt  = numpy.mean(camPosArray[:, 5])

print str(cameraPosOnRobotX) + ", " + str(cameraPosOnRobotY) + ", " + str(cameraPosOnRobotZ) + ", " + str(math.degrees(cameraPosOnRobotHoriz)) + ", " + str(math.degrees(cameraPosOnRobotVert)) + ", " + str(math.degrees(cameraPosOnRobotTilt))


import sys
import time
from networktables import NetworkTable

import logging
logging.basicConfig(level=logging.DEBUG)

# if len(sys.argv) != 2:
#     print("Error: specify an IP to connect to!")
#     exit(0)
#
# ip = sys.argv[1]
#
#NetworkTable.setIPAddress(ip)
NetworkTable.setIPAddress("10.96.86.31")
NetworkTable.setClientMode()
NetworkTable.initialize()

table = NetworkTable.getTable("Vision")

while True:
    try:
        print('Center:', table.getNumber('targetCenterX'), table.getNumber('targetCenterY'))
    except KeyError:
        print('Answer not found')
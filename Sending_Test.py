import time
from networktables import NetworkTable

table = NetworkTable.getTable("GoalCenters")

while True:
    table.putNumber('answer', 42)
    time.sleep(1)

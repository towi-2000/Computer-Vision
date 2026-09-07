from gpiozero import DistanceSensor
import time

sensor = DistanceSensor(echo=23, trigger=24, max_distance=2)
while True:
    print(sensor.distance)
    time.sleep(1)
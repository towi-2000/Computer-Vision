#----------------------------
# imports 
#----------------------------
import cv2 as cv, numpy as np, time, RPi.GPIO as GPIO
from smbus2 import SMBus, i2c_msg
from collections import deque
from gpiozero import DistanceSensor

#----------------------------
# Variables declaration
#----------------------------

sleeptime = 1
aruco_type = cv.aruco.DICT_4X4_50

#----------------------------
# class declarations
#----------------------------

class AdaptiveGain:
    def __init__(self, start=0.1, min_gain=0.01, max_gain=1.0,
                 grow=1.05, shrink=0.9, window=20, deadband=0.02, tune_interval=0.5, growth_error_cap=1.0):
        self.value = start
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.grow = grow
        self.shrink = shrink
        self.deadband = deadband
        self.errors = deque(maxlen=window)
        self.window = window
        self.tune_interval = tune_interval
        self.last_tune = time.time()
        self.growth_error_cap = growth_error_cap
    
    def update(self, error):
        self.errors.append(error)
        if len(self.errors) < self.window:
            return self.value

        e = list(self.errors)
        crossings = sum(1 for a, b in zip(e, e[1:]) if(
            abs(a) > self.deadband
            and abs(b) > self.deadband
            and a * b < 0
        ))
        
        if crossings >= self.window // 3:
            self.value *= self.shrink
        elif np.mean(np.abs(e)) > self.deadband:
            mean_error = np.mean(np.abs(e))
            self.value *= (1 + mean_error * 0.05)
        else:
            self.value *= 0.999
        
        self.value = float(np.clip(self.value, self.min_gain, self.max_gain))
        
        return self.value

    def record(self, error):
        self.errors.append(error)
    
    def maybe_tune(self):
        if time.time() - self.last_tune < self.tune_interval:
            return self.value
        
        e = list(self.errors)
        crossings = sum(1 for a,b in zip(e, e[1:]) if abs(a) > self.deadband and abs(b) > self.deadband and a * b < 0)
        
        if crossings >= self.window // 3:
            self.value *= self.shrink
        elif np.mean(np.abs(e)) > self.deadband:
            capped_error = min(np.mean(np.abs(e)), self.growth_error_cap)
            self.value *= (1 + capped_error * 0.05)
        else:
            self.value *= 0.999
        
        self.value = float(np.clip(self.value, self.min_gain, self.max_gain))
        self.last_tune = time.time()
        return self.value

class Data:
    def __init__(self):
        self.cx: int = 0
        self.cy: int = 0
        self.speed_max: int = 255
        self.speed_min: int = 40
        self.speed_l: int = 0
        self.speed_r: int = 0
        self.dist: float = 0
        self.dist_max: float = 20.0
        self.timeout_factor: float = 0.1
        self.measurement_time: float = 0.1 #states the time, when to measure
        self.last_measurement: float = time.time()
        self.trigger: int = 24
        self.echo: int = 23
        self.screen_mid: int | None = None
        self.distances = deque(maxlen=10)
        self.lost_frames: int = 0
        self.i2caddress: int = 0x70
    
    #applies the speed limits
    def applyLimits(self, speed:float):
        if speed > 0:
            speed = max(speed, self.speed_min)

        elif speed < 0:
            speed = min(speed, -self.speed_min)
        
        return speed

#----------------------------
# function declarations
#----------------------------

#sends speed to motors
def drive(pwm0: int, pwm1:int, pwm2:int, pwm3:int):
    #print("drive")
    add = state.i2caddress
    i2c.write_byte_data(add, 0x02, pwm0)
    i2c.write_byte_data(add, 0x03, pwm1)
    i2c.write_byte_data(add, 0x04, pwm2)
    i2c.write_byte_data(add, 0x05, pwm3)

#translates speed to PWM
def sendSpeed(speed_l:int, speed_r:int):
    if speed_l >= 0:
        pwm3 = speed_l  #links vorwärts
        pwm2 = 0  #links rückwärts
    else:
        pwm3 = 0
        pwm2 = abs(speed_l)
    
    if speed_r >= 0:
        pwm1 = speed_r #rechts vprwärts
        pwm0 = 0 #rechs rückwärts
    else:
        pwm1 = 0
        pwm0 = abs(speed_r)
    
    pwm0 = max(0, min(255, pwm0))
    pwm1 = max(0, min(255, pwm1))
    pwm2 = max(0, min(255, pwm2))
    pwm3 = max(0, min(255, pwm3))
    
    drive(pwm0, pwm1, pwm2, pwm3)

#finds a working camera
def find_camera(max_index = 10):
    for i in range(max_index):
        cam = cv.VideoCapture(i)

        if cam.isOpened():
            return i
        cam.release()

#measures distance to target
# def measure_distance(trigger:int, echo:int, timeout=0.03):
#     GPIO.output(trigger, True)
#     time.sleep(0.00001)
#     GPIO.output(trigger, False)

#     start_wait = time.monotonic()

#     while GPIO.input(echo) == 0:
#         if time.monotonic() - start_wait > timeout:
#             return None

#     pulse_start = time.monotonic_ns()
#     start = time.monotonic()

#     while GPIO.input(echo) == 1:
#         if time.monotonic() - start > timeout:
#             return None

#     pulse_end = time.monotonic_ns()
#     pulse_time = (pulse_end - pulse_start) / 1e9
#     #print(f"pulse_time={pulse_time}")
#     if pulse_time > 0.008:
#         return None
    
#     distance = pulse_time * 34300 / 2
    
#     if distance < 2:
#         return None
#     if distance > 400:
#         return None
#     return distance
def measure_distance():
    print("Distanzmessung")
    return sensor.distance * 100

#----------------------------
# initializations
#----------------------------

#image procession
cam_index = find_camera()
if cam_index is None: raise RuntimeError("Keine Kamera gefunden")
cam = cv.VideoCapture(cam_index)

#aruco marker
aruco_dict = cv.aruco.getPredefinedDictionary(aruco_type)
marker = cv.aruco.generateImageMarker(aruco_dict, 0, 1000)
aruco_dict = cv.aruco.getPredefinedDictionary(aruco_type)
detector = cv.aruco.ArucoDetector(aruco_dict)

state = Data()
#state.dist = 10
turn_gain = AdaptiveGain(start=0.3)
dist_gain = AdaptiveGain(start=0.5)

#ultrasonic sensor init
sensor = DistanceSensor(echo=23, trigger=24)
GPIO.setwarnings(False)
GPIO.cleanup()

GPIO.setmode(GPIO.BCM)
GPIO.setup(state.echo, GPIO.IN)
GPIO.setup(state.trigger, GPIO.OUT)
time.sleep(0.5)

#motor i2c
i2c = SMBus(1) #SDA: 3, SCL: 5
i2c.write_byte_data(0x70, 0x00, 0x01)
i2c.write_byte_data(0x70, 0xE8, 0xAA)

#----------------------------
# while-loop
#----------------------------
while cam.isOpened():
    ret, frame = cam.read()
    if not ret: continue
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    #measure distance
    if time.time() - state.last_measurement > state.measurement_time:
        distance = measure_distance()
        print(f"Distanz: {distance}")
        if distance is not None: 
            state.distances.append(distance)
            state.dist = np.median(state.distances)
        state.last_measurement = time.time()

    #set camera and screen data
    if state.screen_mid is None:
        cam_height, cam_width, cam_channels = frame.shape
        state.screen_mid = cam_width // 2

    if ret:
        #Aruco Marker in Frame finden
        corners, ids, rejected = detector.detectMarkers(frame_gray)
        if ids is None:
            state.lost_frames += 1
        
            if state.lost_frames < 10:
                sendSpeed(state.speed_l, state.speed_r)
            else:
                sendSpeed(0,0)

            continue
            
        state.lost_frames = 0
        
        cv.aruco.drawDetectedMarkers(frame, corners, ids)
        
        pts = corners[0][0]
        state.cx = int(np.mean(pts[:,0]))
        state.cy = int(np.mean(pts[:,1]))

        #print(f"Mittelpunkt: ({state.cx}, {state.cy})")
        # print(f"Distanz: {state.dist}")

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
        #Motorsteuerung (P-Regler: output = error * factor)
        if state.screen_mid is not None:
            #calculate errors
            turn_error = (state.cx - state.screen_mid) / state.screen_mid
            dist_error = np.clip(
                (state.dist - state.dist_max) / state.dist_max,
                -1.0,
                1.0
            )
            
            turn_gain.record(turn_error)
            dist_gain.record(dist_error)
            
            #turning
            curr_turn_gain = turn_gain.maybe_tune()
            turn = turn_error * curr_turn_gain * state.speed_max
            
            #distance
            curr_dist_gain = dist_gain.maybe_tune()
            dist = dist_error * curr_dist_gain * state.speed_max
            
            dist *= max(0.3, 1.0 - abs(turn_error))
            
            speed_r = int(state.applyLimits(dist - turn))
            speed_l = int(state.applyLimits(dist + turn))
            state.speed_r = speed_r
            state.speed_l = speed_l

            sendSpeed(speed_l, speed_r)
#end program
drive(0,0,0,0)
cv.destroyAllWindows()
cam.release()
GPIO.cleanup()
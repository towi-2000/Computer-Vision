#----------------------------
# imports 
#----------------------------
import cv2 as cv, numpy as np, time, RPi.GPIO as GPIO, serial, json
from collections import deque
# from gpiozero import OutputDevice, InputDevice

#----------------------------
# Variables declaration
#----------------------------

sleeptime = 1
aruco_type = cv.aruco.DICT_4X4_50

#----------------------------
# function declarations
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
        return self.value

class Data:
    def __init__(self):
        self.cx: int | None = None
        self.cy: int | None = None
        self.speed_max: int = 255
        self.speed_min: int = 40
        self.dist: float | None = None
        self.dist_max: float = 20.0
        self.timeout_factor: float = 0.1
        self.measurement_time: float = 0.1 #states the time, when to measure
        self.last_measurement: float = time.time()
        self.trigger: int = 24
        self.echo: int = 23
        self.screen_mid: int | None = None
        self.distances = deque(maxlen=5)
    
    #applies the speed limits
    def applyLimits(self, speed:float):
        if abs(speed) < self.speed_min:
            return 0

        if abs(speed) > self.speed_max:
            return np.sign(speed) * self.speed_max
        
        return speed

#----------------------------
# function declarations
#----------------------------

#sends speed values to robot
def sendSpeed(speed_l:int, speed_r:int):
    uart.write((json.dumps({
        "left":speed_l,
        "right":speed_r,
    }) + "\n").encode("utf-8"))

#finds a working camera
def find_camera(max_index = 10):
    for i in range(max_index):
        cam = cv.VideoCapture(i)

        if cam.isOpened():
            return i
        cam.release()

def measure_distance(timeout_factor:float, trigger:int, echo:int):
    start_time = 0.0
    stop_time = 0.0
    timeout = time.time() + timeout_factor
    
    # Send a 10 microseconds pulse to the trigger pin
    GPIO.output(trigger, True)
    time.sleep(0.00001)  # 10 microseconds
    GPIO.output(trigger, False)
    
    # Wait until the echo signal starts
    while GPIO.input(echo) == 0:
        start_time = time.time()
        if time.time() > timeout:
            return None

    # Wait until the echo signal ends
    while GPIO.input(echo) == 1:
        stop_time = time.time()
        if time.time() > timeout:
            return None

    # Calculate the duration of the echo signal
    elapsed_time = stop_time - start_time
    
    # Convert time to distance
    # Speed of sound in air (34300 cm/s) and round trip    
    return (elapsed_time * 34300) / 2

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
turn_gain = AdaptiveGain(start=0.3)
dist_gain = AdaptiveGain(start=0.5)

uart = serial.Serial("/dev/serial0", 115200, timeout=1) #TX 14, RX 15

#ultrasonic sensor init
GPIO.setmode(GPIO.BCM)
GPIO.setup(state.echo, GPIO.IN)
GPIO.setup(state.trigger, GPIO.OUT)
#----------------------------
# while-loop
#----------------------------
while cam.isOpened():
    ret, frame = cam.read()
    if not ret: continue
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    #measure distance
    if time.time() - state.last_measurement > state.measurement_time:
        distance = measure_distance(state.timeout_factor, state.trigger, state.echo)
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
            sendSpeed(0,0)
            turn_gain.errors.clear()
            dist_gain.errors.clear()
            continue
        
        cv.aruco.drawDetectedMarkers(frame, corners, ids)
        cv.imshow("img", frame)
        
        pts = corners[0][0]
        state.cx = int(np.mean(pts[:,0]))
        state.cy = int(np.mean(pts[:,1]))
    
        print(f"Mittelpunkt: ({state.cx}, {state.cy})")

        cv.imshow("camera", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
        #Motorsteuerung (P-Regler: output = error * factor)
        if state.cx is not None and state.screen_mid is not None and state.dist is not None:
            #Tuning von turn_factor und dist_factor
            
            #calculate errors
            turn_error = (state.cx - state.screen_mid) / state.screen_mid
            dist_error = (state.dist_max - state.dist) / state.dist_max
            
            turn_gain.record(turn_error)
            dist_gain.record(dist_error)
            
            #turning
            curr_turn_gain = turn_gain.maybe_tune()
            turn = turn_error * curr_turn_gain * state.speed_max
            
            #distance
            curr_dist_gain = dist_gain.maybe_tune()
            dist = dist_error * curr_dist_gain * state.speed_max
            
            if abs(turn_error) > 0.4:
                dist *= 0.3
            
            speed_r = int(state.applyLimits(dist + turn))
            speed_l = int(state.applyLimits(dist - turn))

            sendSpeed(speed_l, speed_r)

#end program
cv.destroyAllWindows()
cam.release()
GPIO.cleanup()
uart.close()
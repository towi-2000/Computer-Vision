#----------------------------
# imports 
#----------------------------
import cv2 as cv, numpy as np, time
# from gpiozero import OutputDevice, InputDevice

#----------------------------
# Variables declaration
#----------------------------
#ultrasonic sensor
trigger_pin = 24
echo_pin = 23

#drive speeds
max_speed, min_speed = 255, -255

sleeptime = 1
distance_to_target = 20     #distance to target in cm
aruco_type = cv.aruco.DICT_4X4_50

#----------------------------
# function declarations
#----------------------------
class Data():
    def __init__(self):
        self.cx: int | None = None
        self.cy: int | None = None
        
        self.base_speed: int = 128
        self.speed_l: int = self.base_speed
        self.speed_r: int = self.base_speed
        self.speed_max: int = 255
        
        self.dist: float | None = None
        self.dist_max: float = 20.0
        self.dist_factor: int = 5
        
        self.screen_mid: int | None = None
        self.error: float | None = None
        self.turn_factor: float = 0.3

#----------------------------
# function declarations
#----------------------------

#sends speed values to robot
def sendSpeed(speed_l:int, speed_r:int):pass

#finds a working camera
def find_camera(max_index = 10):
    for i in range(max_index):
        cam = cv.VideoCapture(i)

        if cam.isOpened():
            return i
        cam.release()

# def measure_distance():
#     # Send a 10 microseconds pulse to the trigger pin
#     trigger.on()
#     time.sleep(0.00001)  # 10 microseconds
#     trigger.off()
    
#     # Wait until the echo signal starts
#     while not echo.is_active:
#         start_time = time.time()

#     # Wait until the echo signal ends
#     while echo.is_active:
#         stop_time = time.time()

#     # Calculate the duration of the echo signal
#     elapsed_time = stop_time - start_time
    
#     # Convert time to distance
#     distance = (elapsed_time * 34300) / 2  # Speed of sound in air (34300 cm/s) and round trip
    
#     return distance

#----------------------------
# initializations
#----------------------------

#ultrasonic sensor pins
# trigger = OutputDevice(trigger_pin)
# echo = InputDevice(echo_pin)

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

#----------------------------
# while-loop
#----------------------------
while cam.isOpened():
    ret, frame = cam.read()
    if not ret: continue
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #set camera and screen data
    if state.screen_mid is None:
        cam_height, cam_width, cam_channels = frame.shape
        state.screen_mid = cam_width // 2

    if ret:
        #Aruco Marker in Frame finden
        corners, ids, rejected = detector.detectMarkers(frame_gray)
        if ids is None: continue
        
        cv.aruco.drawDetectedMarkers(frame, corners, ids)
        cv.imshow("img", frame)
        
        pts = corners[0][0]
        state.cx = int(np.mean(pts[:,0]))
        state.cy = int(np.mean(pts[:,1]))
    
        print(f"Mittelpunkt: ({state.cx}, {state.cy})")

        cv.imshow("camera", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
        #Motorsteuerung
        if state.cx is not None and state.screen_mid is not None:
            #Lenkung
            state.error = state.cx - state.screen_mid
            turn = state.error * state.turn_factor
            speed_l = int(state.base_speed + turn)
            speed_r = int(state.base_speed - turn)
            
            state.speed_r = int(np.clip(
                state.base_speed + state.turn_factor,
                -state.speed_max,
                state.speed_max
            ))
            state.speed_l = int(np.clip(
                state.base_speed - state.turn_factor,
                -state.speed_max,
                state.speed_max
            ))
            
        #Distanzmessung
        # TODO: Distanz messen
        if state.dist is not None and state.error is not None:
            distance_error = state.dist_max - state.dist
            forward = state.dist_factor * state.error
            forward = np.clip(forward, -state.speed_max, state.speed_max)
            

#end program
cv.destroyAllWindows()
cam.release()
#----------------------------
# imports 
#----------------------------
import cv2 as cv, numpy as np, time
from lib import *
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

#----------------------------
# function declarations
#----------------------------
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
#set init speeds
speed_l = 128
speed_r = 128

#ultrasonic sensor pins
# trigger = OutputDevice(trigger_pin)
# echo = InputDevice(echo_pin)

#image procession
cam = cv.VideoCapture(find_camera())
target_img = cv.imread("target.png")
target_markers, target_hierarchy = target_init(target_img)

#target coordinates
cx = None
cy = None

#camera data
cam_dimensions_written = False
cam_width = None
cam_height = None

#screen data
screen_left = None
screen_mid = None
screen_right = None

#----------------------------
# while-loop
#----------------------------
while cam.isOpened():
    ret, frame = cam.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #set camera and screen data
    if cam_dimensions_written == False:
        cam_height, cam_width, cam_channels = frame.shape
        screen_left = cam_width // 3
        screen_mid = cam_width // 2
        screen_right = screen_left * 2
        cam_dimensions_written = True

    if ret:
        #Bild binarisieren
        ret, imgf = cv.threshold(frame_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        #Konturen erkennen
        contours, hierarchy = cv.findContours(image=imgf, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

        #Konturen approximieren
        if len(contours) > 0:
            epsilon = 0.01 * cv.arcLength(contours[0], True)
            approx = cv.approxPolyDP(contours[0], epsilon, True)

        #Koordinaten der Marker finden
        markers = find_markers(hierarchy[0], contours)

        #Marker von Target dem Frame zuweisen
        

        cv.imshow("camera", frame_gray)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

#end program
cv.destroyAllWindows()
cam.release()

'''
Strategie:
1. Bild binarisieren (klarer schwarz-weiß Kontrast; vielleicht Otsu)
2. Konturhierarchien finden
3. nach Vierecken suchen
4. gefundene Marker zuordnen
5. Homographie berechnen
https://www.instructables.com/Object-Tracking-With-Opencv-and-Python-With-Just-5/
'''
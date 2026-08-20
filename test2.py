import cv2 as cv

aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
marker = cv.aruco.generateImageMarker(aruco_dict, 0, 1000)
cv.imwrite("aruco_0.png", marker)
#--------------------------------------
# Library for computer vision functions
#--------------------------------------

#--------------------------------------
# imports
#--------------------------------------
import cv2 as cv, numpy as np

#--------------------------------------
# variable declarations
#--------------------------------------
sift = cv.xfeatures2d.SIFT_create()
sift_standard = cv.SIFT_create()
orb = cv.ORB_create()
bf_matcher = cv.BFMatcher()
backSub = cv.createBackgroundSubtractorMOG2()
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

#--------------------------------------
# functions
#--------------------------------------

#calculate homography with sift
def sift_calculate_homography(frame_raw: np.ndarray, target_raw: np.ndarray):
    #source: https://www.geeksforgeeks.org/python/python-opencv-object-tracking-using-homography/
    result = []
    frame = cv.cvtColor(frame_raw, cv.COLOR_BGR2GRAY)
    kp_frame, desc_frame = sift.detectAndCompute(frame, None)
    target = cv.cvtColor(target_raw, cv.COLOR_BGR2GRAY)
    h, w = frame.shape
    kp_target, desc_target = sift.detectAndCompute(target, None)

    index_params = dict(algorithm = 0, trees = 5)
    search_params = dict()

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_target, desc_frame, k = 2)

    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            result.append(m)
    
    frame_pts = np.float32([kp_frame[m.trainIdx].pt for m in result]).reshape(-1, 1, 2)
    target_pts = np.float32([kp_target[m.trainIdx].pt for m in result]).reshape(-1, 1, 2)

    matrix, mask = cv.findHomography(frame_pts, target_pts, cv.RANSAC, 5.0)

    # matches_mask = mask.ravel().tolist()

    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, matrix)

    homography = cv.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

    return homography

#feature matching with orb
def orb_feature_matching(frame_raw: np.ndarray, target_raw: np.ndarray):
    frame = cv.cvtColor(frame_raw, cv.COLOR_BGR2GRAY)
    target = cv.cvtColor(target_raw, cv.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(frame, None)
    kp_target, des_target = orb.detectAndCompute(target, None)
    h, w = target.shape

    #ratio test
    matches = bf_matcher.knnMatch(des_frame, des_target, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 15: return None #at least 15 good matches

    #calculate homography
    target_pts = np.float32([kp_target[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    frame_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv.findHomography(target_pts, frame_pts, cv.RANSAC, 5.0)
    if H is None: return None

    #transform target-corners to frame
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    return cv.perspectiveTransform(corners, H)

#detects objects in a plain image
def simple_template_matcher(frame_raw: np.ndarray, target_raw: np.ndarray, treshold = 0.6):
    frame = cv.cvtColor(frame_raw, cv.COLOR_BGR2GRAY)
    target = cv.cvtColor(target_raw, cv.COLOR_BGR2GRAY)

    res = cv.matchTemplate(frame, target, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= treshold)
    coords = list(zip(*loc[::-1]))
    return coords

#map Target markers to Frame markers
def map_markers(target_markers:list, frame_markers:list):
    upper = 0.75
    lower = 0.7
    if len(frame_markers) != 4:
        return None
    
    #sort target list and frame list
    #(cx, cy, M['m00'], angle)
    target_list = sorted(target_markers, key=lambda tup:tup[2])
    target_dict = [{"cx": t[0], "cy": t[1], "area": t[2], "angle":None} for t in target_list]
    target_list = None
    frame_list = sorted(frame_markers, key=lambda tup:tup[2])

    #compare square areas
    ratio = target_dict[0]["area"] / target_dict[1]["area"]
    if ratio < lower or ratio > upper:
        return None

    ratio = target_dict[0]["area"] / target_dict[3]["area"]
    if ratio < lower or ratio > upper:
        return None
    
    #calculate center of target
    x = sum(p[0] for p in target_list) / 4
    y = sum(p[1] for p in target_list) / 4

    #calculate relative angles
    angles = calculate_angle(target_dict, (x, y))

#calculates the angles from given coordinates
def calculate_angle(pt:list, cnt:tuple):
    grades = [
        np.arctan2(cnt[1] - pt[0]["cy"], cnt[0] - pt[0]["cx"]),
        np.arctan2(cnt[1] - pt[1]["cy"], cnt[0] - pt[1]["cx"]),
        np.arctan2(cnt[1] - pt[2]["cy"], cnt[0] - pt[2]["cx"]),
        np.arctan2(cnt[1] - pt[3]["cy"], cnt[0] - pt[3]["cx"])
    ]

    pt[1]["angle"] = (grades[1] - grades[0]) % (2 * np.pi)
    pt[2]["angle"] = (grades[2] - grades[0]) % (2 * np.pi)
    pt[3]["angle"] = (grades[3] - grades[0]) % (2 * np.pi)
    sorted_rest = sorted(pt[1:], key=lambda m:m["angle"])
    result = [pt[0]] + sorted_rest
    return result

#detects the contours
def get_contours(img:np.ndarray, min_contour_area = 500):
    fg_mask = backSub.apply(img)
    contours, hierarchy = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    frame_ct = cv.drawContours(img, contours, -1, (0, 255, 0), 2)
    large_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_contour_area]
    return (contours, hierarchy, frame_ct, large_contours)

#finds a working camera
def find_camera(max_index = 10):
    for i in range(0,max_index):
        cam = cv.VideoCapture(i)

        if cam.isOpened():
            return i
        cam.release()

#find center of target in frame
def find_target_coordinates(frame_raw: np.ndarray, target_raw: np.ndarray):pass

#initialize Target
def target_init(target_raw: np.ndarray):
    target = cv.cvtColor(target_raw, cv.COLOR_BGR2GRAY)
    #Bild binarisieren
    ret, imgf = cv.threshold(target, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    #Konturen erkennen
    contours, hierarchy = cv.findContours(image=imgf, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

    #Koordinaten der Marker finden
    markers = find_markers(hierarchy[0], contours)

    return markers, hierarchy

#finds a specific contour in hierachy
def find_markers(input:list, contours:list):
    markers = []

    for i, element in enumerate(input):
        if element[3] != -1 or element[2] == -1 or input[element[2]][2] == -1:
            continue
        
        peri = cv.arcLength(contours[i], True)
        approx = cv.approxPolyDP(contours[i], 0.03 * peri, True)
        if len(approx) != 4 or not cv.isContourConvex(approx):
            continue

        M = cv.moments(contours[i])
        if M['m00'] == 0:
            continue

        cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
        markers.append((cx, cy, M['m00']))
    
    return markers
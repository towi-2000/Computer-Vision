#in der Konsole öffnen (nicht über VSCode)
import cv2 as cv, numpy as np

#camera calibration
# cv.calibrateCamera
# cv.getOptimalNewCameraMatrix

#create matcher
matcher = cv.BFMatcher()
print("Matcher generiert")

#create orb
# orb = cv.ORB_create(nfeatures=75)
orb = cv.ORB_create(nfeatures=100)
print("ORB generiert")

#load and convert target image
# target_img = cv.imread("target.png")
target_img = cv.cvtColor(cv.imread("target.png"), cv.COLOR_BGR2GRAY)
if target_img is None:
    print("Bild konnte nicht geladen werden")
    exit()
print("Bild erfolgreich geladen")
targetKeypoints, targetDescriptors = orb.detectAndCompute(target_img, None)

#open VideoCapture
cam = cv.VideoCapture(1)
print("VideoCapture geladen")

if not cam.isOpened():
    print("Kamera konnte nicht geöffnet werden")
    exit()
print("Kamera offen")

print("Cam found. Backend: ",cam.getBackendName())

#center point coordinates
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

while cam.isOpened():
    #read camera image
    ret, frame = cam.read()
    good_matches = []
    H = 0
    target_found = False

    if cam_dimensions_written == False:
        cam_height, cam_width, cam_channels = frame.shape
        screen_left = cam_width // 3
        screen_mid = cam_width // 2
        screen_right = screen_left * 2
        cam_dimensions_written = True

    if ret:
        #convert camera image
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frameKeypoints, frameDescriptors = orb.detectAndCompute(frame_gray, None)
        matches = matcher.knnMatch(targetDescriptors, frameDescriptors, k = 2)

        # for m, n in matches:
        #     if m.distance < 0.6 * n.distance:
        #         good_matches.append(m)
        
        # query_pts = np.float32([targetKeypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        # train_pts = np.float32([frameKeypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        # matrix, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)
        # matches_mask = mask.ravel().tolist()
        # h, w = frame.shape
        # pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        # dst = cv.perspectiveTransform(pts, matrix)
        # homography = cv.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

        final_img = cv.drawMatches(target_img, targetKeypoints, frame_gray, frameKeypoints, good_matches, None)
        
        #find center of target
        # blur = cv.GaussianBlur(frame_gray, (5, 5), cv.BORDER_DEFAULT)
        # ret_blur, thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY_INV)
        # contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        # for i in contours:
        #     M = cv.moments(i)
        #     if M['m00'] != 0:
        #         cx = int(M['m10']/M['m00'])
        #         cy = int(M['m01']/M['m00'])
                
        #         #visualize center point
        #         cv.drawContours(final_img, [i], -1, (0, 255, 0), 2)
        #         cv.circle(final_img, (cx, cy), 7, (0, 0, 255), -1)
        #         cv.putText(final_img, "center", (cx - 20, cy - 20),
        #                 cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv.imshow("camera",final_img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

#end program
cv.destroyAllWindows()
cam.release()
print("Programm beendet")
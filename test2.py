import cv2 as cv, numpy as np, matplotlib.pyplot as plt

target_img = cv.cvtColor(cv.imread("target.png"), cv.COLOR_RGB2GRAY)
backSub = cv.createBackgroundSubtractorMOG2()

fg_mask = backSub.apply(target_img)
contours, hierarchy = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
frame_ct = cv.drawContours(target_img, contours, -1, (0, 255, 0), 2)
plt.imshow("frame",frame_ct)
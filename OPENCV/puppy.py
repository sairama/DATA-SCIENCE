import cv2
import os
os.chdir('J:\\[FreeTutorials.Eu] Udemy - Python for Computer Vision with OpenCV and Deep Learning\\1. Course Overview and Introduction\\Computer-Vision-with-Python\\DATA')
img = cv2.imread('00-puppy.jpg')

while True:
    cv2.imshow('Puppy',img)
    #if we wait 1 milli second and press escape key
    if cv2.waitKey(1) & 0xFF ==27:
        break;

cv2.destroyAllWindows()
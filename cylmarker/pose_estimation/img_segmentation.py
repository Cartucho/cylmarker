import cv2 as cv
import numpy as np

def marker_segmentation(im):
    # Segment a cylindrical white marker
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    cv.imshow('test', im_gray)
    cv.waitKey(0)

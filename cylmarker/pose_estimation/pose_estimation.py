import cv2 as cv
import numpy as np

def estimate_poses(img_paths):
    for im_path in img_paths:
        im = cv.imread(im_path, cv.IMREAD_COLOR)
        cv.imshow('test', im)
        cv.waitKey(0)

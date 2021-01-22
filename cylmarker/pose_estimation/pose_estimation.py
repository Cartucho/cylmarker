from cylmarker import load_data
from cylmarker.pose_estimation import img_segmentation

import cv2 as cv
import numpy as np

def estimate_poses(cam_calib_data, config_file_data):
    img_paths = load_data.load_img_paths(config_file_data)
    for im_path in img_paths:
        im = cv.imread(im_path, cv.IMREAD_COLOR)
        # Segment the marker
        im_marker = img_segmentation.marker_segmentation(im, config_file_data)

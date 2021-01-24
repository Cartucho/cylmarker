from cylmarker import load_data
from cylmarker.pose_estimation import img_segmentation
from cylmarker.pose_estimation import connected_components

import cv2 as cv
import numpy as np

def estimate_poses(cam_calib_data, config_file_data):
    img_paths = load_data.load_img_paths(config_file_data)
    for im_path in img_paths:
        im = cv.imread(im_path, cv.IMREAD_COLOR)
        # TODO: Check if image was successfully opened
        # Segment the marker
        mask_marker_bg, mask_marker_fg = img_segmentation.marker_segmentation(im, config_file_data)
        # Find connected components
        ## TODO: find min num of connected components
        connected_components.find_connected_components(im, mask_marker_fg)

from cylmarker import load_data, keypoints
from cylmarker.pose_estimation import img_segmentation

import cv2 as cv
import numpy as np


def check_image(im, im_path):
    if im is None:
        print('Error opening the image {}'.format(im_path))
        exit()


def estimate_poses(cam_calib_data, config_file_data, data_pttrn, data_marker):
    # Load data needed to estimate pose
    min_detected_lines = config_file_data['min_detected_lines']
    max_ang_diff = config_file_data['max_angle_diff']
    ## Load pattern data
    sequence_length = len(data_pttrn['sequence_0']['code']) # TODO: hardcoded
    ## Load camera matrix and distortion coefficients
    # TODO:
    # Go through each image and estimate pose
    img_paths = load_data.load_img_paths(config_file_data)
    for im_path in img_paths:
        im = cv.imread(im_path, cv.IMREAD_COLOR)
        check_image(im, im_path) # check if image was sucessfully read
        # Segment the marker
        mask_marker_bg, mask_marker_fg = img_segmentation.marker_segmentation(im, config_file_data)
        # Find keypoints
        pttrn = keypoints.find_keypoints(mask_marker_fg, min_detected_lines, max_ang_diff, sequence_length, data_pttrn, data_marker)
        # Estimate pose
        if pttrn is not None:
            pnts_3d_object, pnts_2d_image = pttrn.get_data_for_pnp_solver()
            # TODO: Validate solution

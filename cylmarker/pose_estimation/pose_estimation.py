from cylmarker import load_data
from cylmarker.pose_estimation import img_segmentation
from cylmarker.pose_estimation import connected_components

import cv2 as cv
import numpy as np


def check_image(im, im_path):
    if im is None:
        print('Error opening the image {}'.format(im_path))
        exit()


def estimate_poses(cam_calib_data, config_file_data, data_pttrn, data_marker):
    # Load data needed to estimate pose
    sequence_length = len(data_pttrn['sequence_0'])
    min_n_conn_comp = config_file_data['min_detected_lines'] * sequence_length
    # Go through each image and estimate pose
    img_paths = load_data.load_img_paths(config_file_data)
    for im_path in img_paths:
        im = cv.imread(im_path, cv.IMREAD_COLOR)
        check_image(im, im_path) # check if image was sucessfully read
        # Segment the marker
        mask_marker_bg, mask_marker_fg = img_segmentation.marker_segmentation(im, config_file_data)
        # Find features
        ## Find connected components
        conn_comp_list = connected_components.find_conn_comp(im, mask_marker_fg, min_n_conn_comp)
        print(len(conn_comp_list))
        # TODO: Find sequences
        # TODO: Estimate pose
        # TODO: Validate solution

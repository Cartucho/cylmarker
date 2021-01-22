import cv2 as cv
import numpy as np


def get_hsv_lower_and_upper(h, h_margin, s_min, s_max, v_min, v_max):
    if h > 180 or h < 0: # TODO: check if it is inclusive or not
        print("Error: invalid h value")
        exit()
    if h_margin > 90:
        print("Error: please choose a smaller h_margin")
        exit()
    if s_min < 0 or s_max > 255:
        print("Error: s should be between 0 and 255")
        exit()
    if v_min < 0 or v_max > 255:
        print("Error: v should be between 0 and 255")
        exit()
    h_min = h - h_margin
    h_max = h + h_margin
    lower = [[h_min, s_min, v_min]]
    upper = [[h_max, s_max, v_max]]
    # Deal with the discontinuity of the `h` channel
    #  important for example when detecting the red color.
    if h_min <= 0:
        lower = [[0, s_min, v_min], [180 + h_min, s_min, v_min]]
        upper = [[h_max, s_max, v_max], [180, s_max, v_max]]
    elif h_max >= 180:
        lower = [[0, s_min, v_min], [h_min, s_min, v_min]]
        upper = [[0 + (h_max - 180), s_max, v_max], [180, s_max, v_max]]
    return np.array(lower, np.uint8), np.array(upper, np.uint8)


def marker_segmentation(im, config_file_data):
    print(config_file_data)
    # Segment the marker assuming that it has a unique colour
    im_hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)

    #cv.imshow('test', im_hsv)
    #cv.waitKey(0)
    h = config_file_data['marker_bg_hue']
    h_margin = config_file_data['marker_bg_hue_margin']
    s_min = config_file_data['marker_bg_s_min']
    s_max = config_file_data['marker_bg_s_max']
    v_min = config_file_data['marker_bg_v_min']
    v_max = config_file_data['marker_bg_v_max']
    lower, upper = get_hsv_lower_and_upper(h, h_margin, s_min, s_max, v_min, v_max)
    marker_mask = None
    for low, up in zip(lower, upper):
        marker_mask_tmp = cv.inRange(im_hsv, low, up)
        if marker_mask is None:
            marker_mask = marker_mask_tmp
        else:
            marker_mask = cv.bitwise_or(marker_blobs_red, marker_mask_tmp)
    #cv.imshow('test', marker_mask)
    #cv.waitKey(0)

    # Get largest contour (to avoid returning noise)


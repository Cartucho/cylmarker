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


def get_hsv_mask(im_hsv, lower, upper):
    mask_colour = None
    for low, up in zip(lower, upper):
        mask_colour_tmp = cv.inRange(im_hsv, low, up)
        if mask_colour is None:
            mask_colour = mask_colour_tmp
        else:
            mask_colour = cv.bitwise_or(marker_blobs_red, mask_colour_tmp)
    return mask_colour


def get_marker_background(im_hsv, config_file_data):
    # Load background HSV data
    h = config_file_data['marker_bg_hue']
    h_margin = config_file_data['marker_bg_hue_margin']
    s_min = config_file_data['marker_bg_s_min']
    s_max = config_file_data['marker_bg_s_max']
    v_min = config_file_data['marker_bg_v_min']
    v_max = config_file_data['marker_bg_v_max']

    # Get lower and upper values
    lower, upper = get_hsv_lower_and_upper(h, h_margin, s_min, s_max, v_min, v_max)
    mask_bg_colour = get_hsv_mask(im_hsv, lower, upper)

    # Erode mask
    # TODO: these values are hardcoded, maybe I should put them in the config file
    kernel = np.ones((5, 5), np.uint8)
    mask_bg_colour_eroded = cv.erode(mask_bg_colour, kernel, iterations = 2)


    ## Get largest contour (to avoid returning noise as a potential marker)
    contours, _hierarchy = cv.findContours(mask_bg_colour_eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    c = max(contours, key = cv.contourArea)
    mask_marker_bg = np.zeros(mask_bg_colour.shape, np.uint8)
    cv.drawContours(mask_marker_bg, [c], -1, 255, -1)

    #cv.imshow('test', mask_bg_colour) # TODO: remove
    #cv.imshow('test_fg', mask_bg_colour_eroded) # TODO: remove
    #cv.imshow('test_fg_2', mask_marker_bg) # TODO: remove
    #cv.waitKey(0) # TODO: remove

    return mask_marker_bg


def get_marker_foreground(im_hsv, config_file_data):
    h = config_file_data['marker_bg_hue']
    h_margin = config_file_data['marker_bg_hue_margin']
    # Load foreground HSV data
    s_min = config_file_data['marker_fg_s_min']
    s_max = config_file_data['marker_fg_s_max']
    v_min = config_file_data['marker_fg_v_min']
    v_max = config_file_data['marker_fg_v_max']

    # Get lower and upper values
    lower, upper = get_hsv_lower_and_upper(h, h_margin, s_min, s_max, v_min, v_max)
    mask_fg_colour = get_hsv_mask(im_hsv, lower, upper)
    #cv.imshow('test_fg', mask_fg_colour) # TODO: remove
    #cv.waitKey(0)

    return mask_fg_colour


def marker_segmentation(im, config_file_data):
    #print(config_file_data)
    # Segment the marker assuming that it has a unique colour
    im_hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    #cv.imshow('test', im_hsv) # TODO: remove
    #cv.waitKey(0)
    mask_marker_bg = get_marker_background(im_hsv, config_file_data)
    #marker_bg = cv.bitwise_and(im, im, mask=mask_marker_bg)
    marker_bg_hsv = cv.bitwise_and(im_hsv, im_hsv, mask=mask_marker_bg)
    mask_marker_fg = get_marker_foreground(marker_bg_hsv, config_file_data)

    return mask_marker_bg, mask_marker_fg


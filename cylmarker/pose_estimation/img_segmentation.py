import cv2 as cv
import numpy as np
import matplotlib

from matplotlib import pyplot as plt # For showing the histograms of the segmented marker


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
    if h_min < 0:
        lower = [[0, s_min, v_min], [180 + h_min, s_min, v_min]]
        upper = [[h_max, s_max, v_max], [180, s_max, v_max]]
    elif h_max > 180:
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
            mask_colour = cv.bitwise_or(mask_colour, mask_colour_tmp)
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

    # Erode mask (remove noise, not sure if needed)
    #kernel = np.ones((3, 3), np.uint8)
    #mask_bg_colour = cv.erode(mask_bg_colour, kernel, iterations = 1)

    ## Get largest contour (to avoid returning noise as a potential marker)
    contours, _hierarchy = cv.findContours(mask_bg_colour, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) == 0: # No marker detected
        return None, None
    c = max(contours, key = cv.contourArea)
    mask_marker_bg = np.zeros(mask_bg_colour.shape, np.uint8)
    cv.drawContours(mask_marker_bg, [c], -1, 255, -1)
    marker_area = cv.contourArea(c)

    # Erode mask (given that we already have the biggest green contour)
    kernel = np.ones((3, 3), np.uint8)
    mask_marker_bg = cv.erode(mask_marker_bg, kernel, iterations = 3)

    return mask_marker_bg, marker_area


def get_marker_foreground(im_hsv, mask_marker_bg, marker_area, config_file_data):
    min_cntr_area_prcntg = config_file_data['min_cntr_area_prcntg']
    min_cntr_area = (min_cntr_area_prcntg / 100.) * marker_area
    #print(min_cntr_area)

    # We will distinguish the foreground and the background using the V channel
    #  the intuition is that the darker parts of the marker should correspond to the keypoints
    th = cv.adaptiveThreshold(im_hsv[:,:,2], 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                 cv.THRESH_BINARY_INV, 47, 2)
    mask_fg_colour = cv.bitwise_and(th, th, mask=mask_marker_bg)
    #cv.imshow('test', th) # TODO: there seems to be alway a big contour that I could remove
    #cv.waitKey(0)

    # Filter (remove the ones that are too small)
    contours, _hierarchy = cv.findContours(mask_fg_colour, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cntr in contours:
        if cv.contourArea(cntr) < min_cntr_area:
            mask_fg_colour = cv.drawContours(mask_fg_colour, [cntr], -1, (0, 0, 0), -1)


    """
    # First, we get the darkest region of the HSV
    h = 90
    h_margin = 90 # h = 90 and h_margin = 90 will cover all the H's
    s_min = 0
    s_max = 255 # cover all the possible S's too
    v_min = 0
    v_max = int(0.15*255) # TODO: should be coming from the config file

    ## Get lower and upper values
    lower, upper = get_hsv_lower_and_upper(h, h_margin, s_min, s_max, v_min, v_max)
    mask_fg_colour = get_hsv_mask(im_hsv, lower, upper)
    # Intersect mask with background mask
    mask_fg_colour = cv.bitwise_and(mask_fg_colour, mask_fg_colour, mask=mask_marker_bg)

    # Then we get the "black-greenish" regions
    h = config_file_data['marker_bg_hue']
    h_margin = config_file_data['marker_bg_hue_margin']
    s_min = 0 #25 #config_file_data['marker_fg_s_min'] # try with 0 too
    s_max = 255 #config_file_data['marker_fg_s_max']
    v_min = v_max
    v_max = v_max + int(0.25*255)
    if v_max > 255:
        v_max = 255

    lower, upper = get_hsv_lower_and_upper(h, h_margin, s_min, s_max, v_min, v_max)
    mask_fg_colour_2 = get_hsv_mask(im_hsv, lower, upper)
    #cv.imshow('1', mask_fg_colour)
    #cv.imshow('2', mask_fg_colour_2)
    # Intersect mask with background mask
    mask_fg_colour = cv.bitwise_or(mask_fg_colour, mask_fg_colour_2, mask=mask_marker_bg)
    """

    return mask_fg_colour


def show_hsv_image(im_hsv):
    im_hsv_copy = cv.cvtColor(im_hsv, cv.COLOR_BGR2RGB)
    cv.imshow('image HSV | Red: H, Green:S, Blue:V', im_hsv_copy)
    cv.waitKey(0)


def show_marker_histogram(im_hsv, mask_marker_bg):
    color = ('r','g','b')
    label = ('H', 'S', 'V')
    plt.clf()

    # HSV
    plot_just_v = False
    for i, (col, lab) in enumerate(zip(color, label)):
        if plot_just_v:
            if i != 2:
                continue
        histr = cv.calcHist([im_hsv], [i], mask_marker_bg, [256], [0,256])
        plt.plot(histr.copy(), color = col, label = lab)
        plt.xlim([0,256])
    plt.legend(loc="upper right")

    #plt.show()
    # Convert plot to numpy image so that I can show with OpenCV
    fig = plt.gcf()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data_bgr = cv.cvtColor(data, cv.COLOR_RGB2BGR)
    cv.imshow("Marker histograms", data_bgr)
    cv.waitKey(0)


def show_marker_histogram_gray(im, mask_marker_bg):
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    cv.imshow('test', im_gray)
    cv.waitKey(0)


def show_features(im, mask_marker_fg):
    marker_fg = cv.bitwise_and(im, im, mask=mask_marker_fg)
    marker_fg[marker_fg!=0] = 255
    cv.imshow('features', marker_fg)
    cv.waitKey(0)


def marker_segmentation(im, config_file_data):
    #print(config_file_data)
    # Segment the marker assuming that it has a unique colour
    im_hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    #show_hsv_image(im_hsv)
    mask_marker_bg, marker_area = get_marker_background(im_hsv, config_file_data)
    if mask_marker_bg is None:
        return None, None
    marker_bg = cv.bitwise_and(im, im, mask=mask_marker_bg)
    #cv.imshow('marker_bg', marker_bg) # TODO: remove
    marker_bg_hsv = cv.bitwise_and(im_hsv, im_hsv, mask=mask_marker_bg)
    #show_marker_histogram(im_hsv, mask_marker_bg)
    #show_marker_histogram_gray(im, mask_marker_bg)
    mask_marker_fg = get_marker_foreground(marker_bg_hsv, mask_marker_bg, marker_area, config_file_data)
    #show_features(im, mask_marker_fg)
    return mask_marker_bg, mask_marker_fg


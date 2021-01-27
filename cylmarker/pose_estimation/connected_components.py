import cv2 as cv


class ConnectedComponent:
    def __init__(self, contour, centre_u, centre_v):
        self.contour = contour
        self.centre_u = centre_u
        self.centre_v = centre_v
        self.area = cv.contourArea(contour)
        # Initially we do not know the label or the id
        self.label = -1
        self.id = -1


def draw_conn_comp(im, conn_comp, color):
    for cc in conn_comp:
        cv.drawContours(im, [cc], -1, color, 1)
    return im


def find_conn_comp(im, mask_marker_fg, min_n_conn_comp):
    cv.imshow('im', im)
    cv.imshow('im2', mask_marker_fg)
    cv.waitKey(0)
    conn_comp_list = []

    contours, _hierarchy = cv.findContours(mask_marker_fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    n_conn_comp_detected = len(contours)
    if n_conn_comp_detected < min_n_conn_comp:
        return conn_comp_list # return empty list

    """
    # Filter the contours TODO: probably not needed
    MIN_CONTOUR_AREA = 5
    contours = [cntr for cntr in contours if cv.contourArea(cntr) >= MIN_CONTOUR_AREA]
    """

    """
     DEBUG
      Draw conn_comp one by one on the original image
    """
    """
    color = (0, 0, 255) # BGR
    im_draw = draw_conn_comp(im, contours, color)
    cv.imshow('window', im_draw)
    cv.waitKey(0)
    """

    for ctr in contours:
        moments = cv.moments(ctr)
        if moments["m00"] == 0.0:
            # get geometrical centre instead
            centre, _size, _angl = cv.minAreaRect(ctr)
            centre_u, centre_v = centre
        else:
            # TODO: try using always geometrical centre instead of this, and compare performance
            centre_u = float(moments["m10"] / moments["m00"])
            centre_v = float(moments["m01"] / moments["m00"])
        cc = ConnectedComponent(ctr, centre_u, centre_v)
        conn_comp_list.append(cc)
    return conn_comp_list

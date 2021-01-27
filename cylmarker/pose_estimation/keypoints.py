import cv2 as cv


class Keypoint:

    def __init__(self, centre_u, centre_v):
        self.centre_u = centre_u
        self.centre_v = centre_v
        # Initially we do not know the label or the id
        self.label = -1
        self.id = -1

    def add_contour(self, contour):
        self.contour = contour
        self.area = cv.contourArea(contour)

def draw_contour(im, contours, color):
    for cntr in contours:
        cv.drawContours(im, [cntr], -1, color, 1)
    return im


def find_keypoints(im, mask_marker_fg, min_n_keypoints):
    cv.imshow('im', im)
    cv.imshow('im2', mask_marker_fg)
    cv.waitKey(0)
    keypoints_list = []

    # Using connected components as keypoints (later in the code they will be uniquely identified)
    contours, _hierarchy = cv.findContours(mask_marker_fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    n_keypoints_detected = len(contours)
    if n_keypoints_detected < min_n_keypoints:
        return keypoints_list # return empty list

    """
    # Filter the contours TODO: probably not needed
    MIN_CONTOUR_AREA = 5
    contours = [cntr for cntr in contours if cv.contourArea(cntr) >= MIN_CONTOUR_AREA]
    """

    """
     DEBUG
      Draw contours one by one on the original image
    """
    """
    color = (0, 0, 255) # BGR
    im_draw = draw_contour(im, contours, color)
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
        kpt = Keypoint(centre_u, centre_v)
        kpt.add_contour(ctr)
        keypoints_list.append(kpt)
    return keypoints_list

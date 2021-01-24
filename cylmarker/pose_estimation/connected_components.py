import cv2 as cv

class ConnectedComponent:
    def __init__(self, name):
        self.name = name


def find_connected_components(im, mask_marker_fg):
    cv.imshow('im', im)
    cv.imshow('im2', mask_marker_fg)
    cv.waitKey(0)

    contours, _hierarchy = cv.findContours(mask_marker_fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    """
    if n_contour_old < (MIN_DETECTED_LINES * n_blobs_per_pattern):
        n_fails += 1
        n_fails_segmentation += 1
        end = time.time()
        total_time += end - start
        continue
    """

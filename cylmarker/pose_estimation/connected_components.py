import cv2 as cv

class ConnectedComponent:
    def __init__(self, name):
        self.name = name


def find_connected_components(im, mask_marker_fg, config_file_data):
    print(config_file_data)
    cv.imshow('im', im)
    cv.imshow('im2', mask_marker_fg)
    cv.waitKey(0)

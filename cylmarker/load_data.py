import errno
import os
import glob

import yaml
from natsort import natsorted


# TODO: this should be loaded from the config file
FILE_NAME_CONFIG = 'config.yaml'
FILE_NAME_CAM_CALIB = 'camera_calibration.yaml'
FILE_NAME_PATTERN = 'pattern.yaml'
FILE_NAME_MARKER_IMG = 'marker.svg'
FILE_NAME_MARKER_DATA = 'marker.yaml'


def is_path_dir(string):
    if os.path.isdir(string):
        return string
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), string)


def is_path_file(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), string)


def load_yaml_data(path):
    if is_path_file(path):
        with open(path) as f_tmp:
            return yaml.load(f_tmp, Loader=yaml.FullLoader)


def get_path_config(data_dir):
    return os.path.join(data_dir, FILE_NAME_CONFIG)


def get_path_cam_calib(data_dir):
    return os.path.join(data_dir, FILE_NAME_CAM_CALIB)


def load_config_and_cam_calib_data(data_dir):
    # Load config file data
    config_file = get_path_config(data_dir)
    config_file_data = load_yaml_data(config_file)

    # Load camera intrinsic matrix and distortion coefficients
    cam_calib_file = get_path_cam_calib(data_dir)
    cam_calib_data = load_yaml_data(cam_calib_file)

    return config_file_data, cam_calib_data


def load_pttrn_and_marker_data(data_dir):
    # Load pattern data
    pttrn_file = get_path_pattern(data_dir)
    pttrn_file_data = load_yaml_data(pttrn_file)
    # Load marker data
    marker_file = get_path_marker_data(data_dir)
    marker_file_data = load_yaml_data(marker_file)
    return pttrn_file_data, marker_file_data


def load_img_paths(config_file_data):
    img_dir_path = config_file_data['img_dir_path']
    if is_path_dir(img_dir_path):
        img_format = config_file_data['img_format']
        img_paths = os.path.join(img_dir_path, '*{}'.format(img_format))
        img_paths_sorted = natsorted(glob.glob(img_paths))
        return img_paths_sorted


def get_path_marker_img(data_dir):
    return os.path.join(data_dir, FILE_NAME_MARKER_IMG)


def get_path_marker_data(data_dir):
    return os.path.join(data_dir, FILE_NAME_MARKER_DATA)


def get_path_pattern(data_dir):
    return os.path.join(data_dir, FILE_NAME_PATTERN)


def get_marker_diameter(config_file_data):
    cyl_diam = config_file_data['cyl_diameter_mm']
    paper_thickness = config_file_data['printing_paper_thickness_mm']
    return cyl_diam + 2.0 * paper_thickness


def get_marker_mm_per_pixel(marker_diameter, marker_width):
    return marker_diameter/marker_width


def get_marker_radius_and_mmperpixel(config_file_data, marker_width):
    diam = get_marker_diameter(config_file_data)
    radius = diam/2.
    mm_per_pixel = get_marker_mm_per_pixel(diam, marker_width)
    return radius, mm_per_pixel




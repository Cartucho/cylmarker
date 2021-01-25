import errno
import os
import glob

import yaml
from natsort import natsorted


FILE_NAME_PATTERN = 'pattern.yaml'
FILE_NAME_MARKER = 'marker.svg'

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



def load_cam_calib_data(path):
    if is_path_file(path):
        with open(path) as f_tmp:
            return yaml.load(f_tmp, Loader=yaml.FullLoader)


def load_yaml(data_dir):
    # Load camera intrinsic matrix and distortion coefficients
    cam_calib_file = os.path.join(data_dir, 'camera_calibration.yaml')
    cam_calib_data = load_cam_calib_data(cam_calib_file)

    # Load config file data
    config_file = os.path.join(data_dir, 'config.yaml')
    config_file_data = load_cam_calib_data(config_file)

    return cam_calib_data, config_file_data


def load_img_paths(config_file_data):
    img_dir_path = config_file_data['img_dir_path']
    if is_path_dir(img_dir_path):
        img_format = config_file_data['img_format']
        img_paths = os.path.join(img_dir_path, '*{}'.format(img_format))
        img_paths_sorted = natsorted(glob.glob(img_paths))
        return img_paths_sorted


def get_path_marker(data_dir):
    return os.path.join(data_dir, FILE_NAME_MARKER)


def get_path_pattern(data_dir):
    return os.path.join(data_dir, FILE_NAME_PATTERN)

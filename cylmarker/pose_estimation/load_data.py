import errno
import os
import yaml


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


def load_data(data_dir):
    # Load camera intrinsic matrix and distortion coefficients
    cam_calib_file = os.path.join(data_dir, 'camera_calibration.yaml')
    cam_calib_data = load_cam_calib_data(cam_calib_file)

    return cam_calib_data

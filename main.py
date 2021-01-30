from cylmarker import load_data, save_data
from cylmarker.pose_estimation import pose_estimation
from cylmarker.make_new_pattern_and_marker import create_new_pattern, create_new_marker
import argparse


TASK_CHOICES = {'c': 'camera_calibration', 'p': 'pose_estimation', 'm': 'make_new_pattern_and_marker'}


def get_args():
    parser = argparse.ArgumentParser(description='This program calculates the 6DoF pose of a cylindrical marker.')
    parser.add_argument('--path', type=load_data.is_path_dir, default='data', help= 'Get path to data dir.')
    parser.add_argument('--task', choices=TASK_CHOICES.keys(), default='p', help= 'Task to be executed.')
    return parser.parse_args()


def main():
    args = get_args()
    task = TASK_CHOICES.get(args.task)
    # Load data
    data_config, data_cam_calib = load_data.load_config_and_cam_calib_data(args.path)
    if task == 'pose_estimation':
        data_pttrn, data_marker = load_data.load_pttrn_and_marker_data(args.path)
        # Estimate pose for each image
        pose_estimation.estimate_poses(data_cam_calib, data_config, data_pttrn, data_marker)
    elif task == 'camera_calibration':
        pass
    elif task == 'make_new_pattern_and_marker':
        save_data.check_and_warn_if_files_will_be_replaced(args.path)
        # Make and save new pattern
        new_pttrn = create_new_pattern.get_new_pttrn(data_config)
        save_data.save_new_pttrn(args.path, new_pttrn)
        # Make and save new marker
        marker_img, u_v, x_y_z, kpts = create_new_marker.draw_marker(args.path, data_config, new_pttrn)
        save_data.save_new_marker(args.path, marker_img, u_v, x_y_z, kpts)

if __name__ == "__main__":
    main()


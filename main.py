import argparse

from cylmarker import load_data
from cylmarker.pose_estimation import pose_estimation
from cylmarker.make_new_pattern_and_marker import create_new_pattern, create_new_marker

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
    cam_calib_data, config_file_data = load_data.load_yaml(args.path)
    if task == 'pose_estimation':
        # Estimate pose for each image
        pose_estimation.estimate_poses(cam_calib_data, config_file_data)
    elif task == 'camera_calibration':
        pass
    elif task == 'make_new_pattern_and_marker':
        new_pttrn = create_new_pattern.get_new_pttrn(config_file_data)
        #create_new_pattern.save_new_pttrn(args.path, new_pttrn)

if __name__ == "__main__":
    main()


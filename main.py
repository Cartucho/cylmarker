import argparse

from cylmarker.pose_estimation import load_data


TASK_CHOICES = {'c': 'camera_calibration', 'p': 'pose_estimation'}


def get_args():
    parser = argparse.ArgumentParser(description='This program calculates the 6DoF pose of a cylindrical marker.')
    parser.add_argument('--path', type=load_data.is_path_dir, default='data', help= 'Get path to data dir.')
    parser.add_argument('--task', choices=TASK_CHOICES.keys(), default='p', help= 'Task to be executed.')
    return parser.parse_args()


def main():
    args = get_args()
    task = TASK_CHOICES.get(args.task)
    if task == 'pose_estimation':
        # Load data
        cam_calib_data, config_file_data = load_data.load_yaml(args.path)
        # Load image paths
        img_paths = load_data.load_img_paths(config_file_data)
        print(img_paths)

if __name__ == "__main__":
    main()


import argparse
import os

from cylmarker.pose_estimation import load_data


def main():
    # Get path to the data dir
    parser = argparse.ArgumentParser(description='This program calculates the 6DoF pose of a cylindrical marker.')
    parser.add_argument('--path', type=load_data.is_path_dir, default='data', help= 'Path to folder with (a) intrinsic camera parameters and (b) marker pattern data.')
    args = parser.parse_args()

    # Load data
    cam_data = load_data.load_data(args.path)
    print(cam_data)

if __name__ == "__main__":
    main()


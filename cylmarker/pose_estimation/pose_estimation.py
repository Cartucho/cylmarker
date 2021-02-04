from cylmarker import load_data, keypoints
from cylmarker.pose_estimation import img_segmentation, validate_solution

import cv2 as cv
import numpy as np


def check_image(im, im_path):
    if im is None:
        print('Error opening the image {}'.format(im_path))
        exit()


def show_sgmntd_bg_and_fg(im, mask_marker_bg, mask_marker_fg):
    im_copy = im.copy()
    alpha = 0.4
    # First we show the background part only
    mask_marker_bg = cv.subtract(mask_marker_bg, mask_marker_fg);
    mask_bg_blue = np.zeros_like(im_copy)
    mask_bg_blue[:,:,0] = mask_marker_bg
    im_copy = cv.addWeighted(im_copy, 1.0, mask_bg_blue, alpha, 0)
    # Then we show the foreground part
    alpha = 0.7
    mask_fg_red = np.zeros_like(im_copy)
    mask_fg_red[:,:,2] = mask_marker_fg
    im_copy = cv.addWeighted(im_copy, 1.0, mask_fg_red, alpha, 0)
    cv.imshow("Segmentation | Blue: background | Red: foreground", im_copy)
    cv.waitKey(0)


def show_contours_and_lines(im, pttrn):
    blue = (255, 0, 0)
    green = (0, 255, 0)
    for sqnc in pttrn.list_sqnc:
        if sqnc.sqnc_id != -1:
            # draw line between first and last kpt
            u_0, v_0 = sqnc.list_kpts[0].get_centre_uv()
            u_1, v_1 = sqnc.list_kpts[-1].get_centre_uv()
            im = cv.line(im, (int(u_0), int(v_0)), (int(u_1), int(v_1)), green, 1)
            # draw contours
            for kpt in sqnc.list_kpts:
                cntr = kpt.cntr
                cv.drawContours(im, [cntr], -1, blue, 1)
    cv.imshow("image", im)
    cv.waitKey(0)


def show_axis(im, rvecs, tvecs, cam_matrix, dist_coeff):
    #print(cam_matrix)
    #print(np.transpose(tvecs))
    axis = np.float32([[0, 0, 0], [3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)
    #print(axis)
    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, cam_matrix, dist_coeff)
    print(imgpts)
    frame_centre = tuple(imgpts[0].ravel())
    im = cv.line(im, frame_centre, tuple(imgpts[1].ravel()), (255,0,0), 5)
    im = cv.line(im, frame_centre, tuple(imgpts[2].ravel()), (0,255,0), 5)
    im = cv.line(im, frame_centre, tuple(imgpts[3].ravel()), (0,0,255), 5)
    cv.imshow("image", im)
    cv.waitKey(0)
    #exit()


def estimate_poses(cam_calib_data, config_file_data, data_pttrn, data_marker):
    # Load data needed to estimate pose
    min_detected_lines = config_file_data['min_detected_lines']
    max_ang_diff_group = config_file_data['max_angle_diff_group']
    max_ang_diff_label = config_file_data['max_angle_diff_label']
    ## Load pattern data
    sequence_length = len(data_pttrn['sequence_0']['code']) # TODO: hardcoded
    ## Load camera matrix and distortion coefficients
    cam_matrix = cam_calib_data['camera_matrix']['data']
    cam_matrix = np.reshape(cam_matrix, (3, 3))
    dist_coeff = cam_calib_data['dist_coeff']['data']
    dist_coeff = np.array(dist_coeff)
    # Go through each image and estimate pose
    img_paths = load_data.load_img_paths(config_file_data)
    for im_path in img_paths:
        im = cv.imread(im_path, cv.IMREAD_COLOR)
        check_image(im, im_path) # check if image was sucessfully read
        """ 1. Segment the marker """
        mask_marker_bg, mask_marker_fg = img_segmentation.marker_segmentation(im, config_file_data)
        # Draw segmented background and foreground
        show_sgmntd_bg_and_fg(im, mask_marker_bg, mask_marker_fg)
        """ 2. Find keypoints """
        pttrn = keypoints.find_keypoints(im, mask_marker_fg, min_detected_lines, max_ang_diff_group, max_ang_diff_label, sequence_length, data_pttrn, data_marker)
        # Estimate pose
        if pttrn is not None:
            # Draw contours and lines (for visualization)
            #show_contours_and_lines(im, pttrn)
            pnts_3d_object, pnts_2d_image = pttrn.get_data_for_pnp_solver()
            """ 3. Estimate pose using the PnPRansac """
            #(cv.SOLVEPNP_EPNP is faster than cv.ITERATIVE)
            retval, rvec_pred, tvec_pred, inliers = cv.solvePnPRansac(pnts_3d_object, pnts_2d_image, cam_matrix, dist_coeff, None, None, False, 1000, 3.0, 0.9999, None, cv.SOLVEPNP_EPNP)
            rmat_pred, _ = cv.Rodrigues(rvec_pred)
            transf_marker_to_cam = np.concatenate((rmat_pred, tvec_pred), axis = 1)
            # Draw axis
            #show_axis(im, rvec_pred, tvec_pred, cam_matrix, dist_coeff)
            """ 4. Validate solution """
            passed = validate_solution.validate_pose(pttrn, im, transf_marker_to_cam, cam_matrix)
            print(passed)

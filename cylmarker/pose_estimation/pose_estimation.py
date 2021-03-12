from cylmarker import load_data, keypoints
from cylmarker.pose_estimation import img_segmentation

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


def show_contours_and_lines_and_centroids(im, pttrn):
    blue = (255, 0, 0)
    yellow=(124,225,255)
    red = (0, 0, 255)
    for sqnc in pttrn.list_sqnc:
        if sqnc.sqnc_id != -1:
            """ draw contours """
            for kpt in sqnc.list_kpts:
                cntr = kpt.cntr
                cv.drawContours(im, [cntr], -1, blue, 1)
            """ draw line between first and last kpt """
            u_0, v_0 = sqnc.list_kpts[0].get_centre_uv()
            u_1, v_1 = sqnc.list_kpts[-1].get_centre_uv()
            im = cv.line(im, (int(u_0), int(v_0)), (int(u_1), int(v_1)), yellow, 2, cv.LINE_AA) # lines
            """ draw centroids """
            for kpt in sqnc.list_kpts:
                u, v = kpt.get_centre_uv()
                im = cv.circle(im, (int(round(u)), int(round(v))), radius=2, color=red, thickness=-1)
    cv.imshow("image", im)
    cv.waitKey(0)


def show_axis(im, rvecs, tvecs, cam_matrix, dist_coeff, length):
    #print(cam_matrix)
    #print(np.transpose(tvecs))
    axis = np.float32([[0, 0, 0], [length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
    #print(axis)
    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, cam_matrix, dist_coeff)
    #print(imgpts)
    frame_centre = tuple(imgpts[0].ravel())

    thickness = 4
    im = cv.line(im, frame_centre, tuple(imgpts[3].ravel()), (255,0,0), thickness, cv.LINE_AA)
    im = cv.line(im, frame_centre, tuple(imgpts[2].ravel()), (0,255,0), thickness, cv.LINE_AA)
    im = cv.line(im, frame_centre, tuple(imgpts[1].ravel()), (0,0,255), thickness, cv.LINE_AA)

    cv.imshow("image", im)
    cv.waitKey(0)


def get_transf_inv(transf):
    # ref: https://math.stackexchange.com/questions/152462/inverse-of-transformation-matrix
    r = transf[0:3, 0:3]
    t = transf[0:3, 3]
    r_inv = np.transpose(r) # Orthogonal matrix so the inverse is its transpose
    t_new = np.matmul(-r_inv, t).reshape(3, 1)
    transf_inv = np.concatenate((r_inv, t_new), axis = 1)
    return transf_inv


def save_pts_info(im_path, pnts_3d_object, pnts_2d_image):
    filename = '{}.pts3d.txt'.format(im_path)
    np.savetxt(filename, (np.squeeze(pnts_3d_object)), fmt="%s", delimiter=',')
    filename = '{}.pts2d.txt'.format(im_path)
    np.savetxt(filename, (np.squeeze(pnts_2d_image)), fmt="%s", delimiter=',')


def save_pose(im_path, mat):
    filename = '{}.txt'.format(im_path)
    np.savetxt(filename, (mat), fmt="%s", delimiter=',')


def estimate_poses(cam_calib_data, config_file_data, data_pttrn, data_marker):
    ## Load pattern data
    sqnc_max_ind = len(data_pttrn) - 1
    sequence_length = len(data_pttrn['sequence_0']['code']) # TODO: hardcoded
    ## Load camera matrix and distortion coefficients
    cam_matrix = cam_calib_data['camera_matrix']['data']
    cam_matrix = np.reshape(cam_matrix, (3, 3))
    dist_coeff = cam_calib_data['dist_coeff']['data']
    dist_coeff = np.array(dist_coeff)
    # Go through each image and estimate pose
    img_dir_path = config_file_data['img_dir_path']
    img_format = config_file_data['img_format']
    img_paths = load_data.load_img_paths(img_dir_path, img_format)
    for im_path in img_paths:
        im = cv.imread(im_path, cv.IMREAD_COLOR)
        check_image(im, im_path) # check if image was sucessfully read
        """ Step I - Undistort the input image """
        im = cv.undistort(im, cam_matrix, dist_coeff)
        dist_coeff = None # we don't need to undistort again
        """ Step II - Segment the marker and detect features """
        mask_marker_bg, mask_marker_fg = img_segmentation.marker_segmentation(im, config_file_data)
        if mask_marker_bg is None:
            continue
        # Draw segmented background and foreground
        #show_sgmntd_bg_and_fg(im, mask_marker_bg, mask_marker_fg)
        """ Step III - Identify features """
        pttrn = keypoints.find_keypoints(im, mask_marker_fg, config_file_data, sqnc_max_ind, sequence_length, data_pttrn, data_marker)
        # Estimate pose
        if pttrn is not None:
            # Draw contours and lines (for visualization)
            #show_contours_and_lines_and_centroids(im, pttrn)
            pnts_3d_object, pnts_2d_image = pttrn.get_data_for_pnp_solver()
            #save_pts_info(im_path, pnts_3d_object, pnts_2d_image)
            """ Step IV - Estimate the marker's pose """
            valid, rvec_pred, tvec_pred, inliers = cv.solvePnPRansac(pnts_3d_object, pnts_2d_image, cam_matrix, dist_coeff, None, None, False, 1000, 3.0, 0.9999, None, cv.SOLVEPNP_EPNP)
            if valid:
                # Draw axis
                show_axis(im, rvec_pred, tvec_pred, cam_matrix, dist_coeff, 6)
                # Save solution
                #rmat_pred, _ = cv.Rodrigues(rvec_pred)
                #transf = np.concatenate((rmat_pred, tvec_pred), axis = 1)
                #save_pose(im_path, transf)

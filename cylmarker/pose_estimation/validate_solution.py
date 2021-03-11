from cylmarker import keypoints

import cv2 as cv
import numpy as np

def validate_pose(pttrn, im, rvecs, tvecs, cam_matrix, dist_coeff):
    # TODO: Check if sequences are visible by the camera
    # Check Intersection over Union (IoU) of each sequence
    im_height, im_width = im.shape[:2]
    error_not_found = True

    iou_counter = 0
    iou_total = 0.0
    for sqnc in pttrn.list_sqnc:
        if sqnc.sqnc_id != -1 and error_not_found:
            """
                We will compare the detected and projected contours
                of each sequence, one-by-one.
                By going one-by-one we also check if a sequence
                was mis-identified.
            """
            for kpt in sqnc.list_kpts:
                drawing_detected = np.zeros((im_height, im_width), np.uint8)
                drawing_projected = np.zeros((im_height, im_width), np.uint8)
                # First, we draw the detected contour
                cntr = kpt.cntr
                drawing_detected = cv.drawContours(drawing_detected, [cntr], -1, 255, -1)
                #cv.imshow("contours detected", drawing_detected)
                # Then, we get the projected contour
                corners_3d = np.float32(kpt.xyz_corners).reshape(-1,3)
                imgpts, jac = cv.projectPoints(corners_3d, rvecs, tvecs, cam_matrix, dist_coeff)
                imgpts = np.asarray(imgpts, dtype=np.int32)
                drawing_projected = cv.fillPoly(drawing_projected, [imgpts], 255)
                #cv.imshow("contours projected", drawing_projected)
                #cv.waitKey(0)

                intersection = np.logical_and(drawing_projected, drawing_detected)
                union = np.logical_or(drawing_projected, drawing_detected)
                iou_score = np.sum(intersection) / np.sum(union)
                if iou_score < 0.2: # TODO: This value should come from the config file!
                    return False, None
                iou_counter += 1
                iou_total += iou_score
    iou_avg = iou_total / iou_counter
    #if iou_avg < 0.5: # TODO: get from config file
    #    return False, None
    return error_not_found, iou_avg

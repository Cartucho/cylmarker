from cylmarker import keypoints

import cv2 as cv
import numpy as np

def validate_pose(pttrn, im, transf_marker_to_cam, cam_matrix):
    # TODO: Check if sequences are visible by the camera
    # Check Intersection over Union (IoU) of each sequence
    im_height, im_width = im.shape[:2]
    passed = True
    for sqnc in pttrn.list_sqnc:
        if sqnc.sqnc_id != -1:
            """
                We will compare the detected and projected contours
                of each sequence, one-by-one.
                By going one-by-one we also check if a sequence
                was mis-identified.
            """
            drawing_detected = np.zeros((im_height, im_width), np.uint8)
            drawing_projected = np.zeros((im_height, im_width), np.uint8)
            # First, we draw the detected contours
            for kpt in sqnc.list_kpts:
                cntr = kpt.cntr
                cv.drawContours(drawing_detected, [cntr], -1, 255, -1)
            #cv.imshow("contours detected", drawing_detected)
            # Then, we get the projected contours
            for kpt in sqnc.list_kpts:
                # For each keypoint we will draw the contour
                pts = []
                for (x, y, z) in kpt.xyz_corners:
                    point_3d = np.array([x,
                                         y,
                                         z,
                                         1.0])
                    point_2d = np.matmul(cam_matrix, np.matmul(transf_marker_to_cam, point_3d))
                    point_2d /= point_2d[2]
                    u = round(int(point_2d[0]))
                    v = round(int(point_2d[1]))
                    pts.append([u, v])
                pts_array = np.asarray(pts, dtype=np.int32)
                cv.fillPoly(drawing_projected, [pts_array], 255)#, -1)
            #cv.imshow("contours projected", drawing_projected)
            #cv.waitKey(0)
            intersection = np.logical_and(drawing_projected, drawing_detected)
            union = np.logical_or(drawing_projected, drawing_detected)
            iou_score = np.sum(intersection) / np.sum(union)
            if iou_score < 0.15:
                passed = False
                break
    return passed

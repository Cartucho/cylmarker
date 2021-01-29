import math

import cv2 as cv
import numpy as np


class Keypoint:

    def __init__(self, centre_u, centre_v):
        self.centre_u = centre_u
        self.centre_v = centre_v
        # Initially we do not know the label or the id
        self.label = -1
        self.id = -1
        # Data for grouping in sequences
        self.used = False
        self.anchor_du = -1.
        self.anchor_dv = -1.

    def add_contour(self, contour):
        self.contour = contour
        self.area = cv.contourArea(contour)

    def get_centre(self):
        return self.centre_u, self.centre_v


class Sequence:

    def __init__(self, sqnc_kpts_list, sqnc_id):
        self.kpts_list = sqnc_kpts_list
        self.id = sqnc_id



def draw_contours(im, contours, color):
    for cntr in contours:
        cv.drawContours(im, [cntr], -1, color, 1)
    return im


def find_angles_with_other_keypoints(kpt_anchor, kpts_list, max_ang_diff):
    anchor_u, anchor_v = kpt_anchor.get_centre()
    angles = []
    angles_kpts = []
    for kpt in kpts_list: # TODO: could be done in parallel
        if kpt is not kpt_anchor:
            u, v = kpt.get_centre()
            """
              -----> (u)
              |    90*
              v 0* x 180*
             (v)  270*
            """
            du = (anchor_u - u)
            dv = (anchor_v - v)
            kpt.anchor_du = du
            kpt.anchor_dv = dv

            angle_rads = math.atan2(dv, du)
            angle = math.degrees(angle_rads % (2 * math.pi)) # get angle between [0* and 360*[
            """ example: both the kpts in 100* and 280* belong to the same line
                        , so we can remap the ones > 180*, by subtracting 180.

              -----> (u)                       -----> (u)
              |      90*a                      |      90*a
              |                                |
              |    0* x 180*                   |    0* x 0*
              |                                |
              v     b         a = 100*         v     b       a = 100*
             (v)     270*     b = 280*        (v)      90*   b = 100*
            """
            if angle >= 180.0:
                angle -= 180.0
            angles.append(angle)
            angles_kpts.append(kpt)

            """ The issue is that nearby points can still have very distinct values

              -----> (u)
              |      90*
              |   a       b
              |    0* x 0*     a =   1*
              |   d       c    b = 179*
              v                c =   1*
             (v)     90*       d = 179*

              With these values, we would not be able to match `a` with `b`,
              `a` with `d`, `b` with `c` and `c` with `d`.

              The discontinuity issue is always between a small angle (close to 0*)
              and a large angle (close to 180*). So we can solve it by simply splitting
              the small angles into two possible angles:
                        a_1 = 1* or a_2 = 181*
                        c_1 = 1* or c_2 = 181*

              And, then if we compare all the values, including the new ones _1/_2,
              we can group all the keypoints in the same sequence. Notice, that the
              found sequence can only have either `a_1` or `a_2` and not both.

              Adding an additional point is ok since we will sort the angles, and consequently
              only one of them will be assigned to the forming sequence of keypoints.
              In other words, since 1* and 181* is so different (> MAX_ANGLE_DIFF) when we assign
              a group only either _1 or _2 will be used:

              Example:
              Imagine this sorted group of angles, see how they end up in very distinct positions:
              0, 1 (= a), 2, 3, 20, 21, 40, 60, 179, 179, 180, 180, 180, 181 (= a+180)
                      ^                                                         ^
                      | a_1                                                     | a_2
            """
            if angle < max_ang_diff:
                angles.append(angle + 180.0)
                # additional angle but keypoint is the same
                angles_kpts.append(kpt)
    # Sort angles
    angles_argsort = np.argsort(angles)
    angles = np.array(angles)
    print(angles[angles_argsort])

    return angles, angles_kpts


def group_keypoint_in_sequences(kpts_list, max_ang_diff, sequence_length):
    # ref: https://www.cs.princeton.edu/courses/archive/spring03/cs226/assignments/lines.html
    n_keypoints = len(kpts_list)
    used_kpts_counter = 0
    for kpt_anchor in kpts_list:
        if (n_keypoints - used_kpts_counter) < sequence_length:
            # it won't be possible to find another sequence
            # since we would need at least n = sequence_length
            # keypoints that were not used yet
            break
        if not kpt_anchor.used:
            kpt_anchor.used = True
            used_kpts_counter += 1
            # Find angles that the `kpt_anchor` does with the other `kpts` that were not used yet
            find_angles_with_other_keypoints(kpt_anchor, kpts_list, max_ang_diff)
            exit()
    return None


def filter_contours_by_min_area(contours, min_contour_area):
    return [cntr for cntr in contours if cv.contourArea(cntr) >= min_contour_area]


def calculate_contour_centre(cntr):
    """
    # TODO: try using only these two lines, instead of the rest
    centre, _size, _angl = cv.minAreaRect(cntr)
    centre_u, centre_v = centre
    """
    moments = cv.moments(cntr)
    if moments["m00"] == 0.0:
        # get geometrical centre instead
        centre, _size, _angl = cv.minAreaRect(cntr)
        centre_u, centre_v = centre
    else:
        centre_u = float(moments["m10"] / moments["m00"])
        centre_v = float(moments["m01"] / moments["m00"])
    return centre_u, centre_v


def get_connected_components(mask_marker_fg, min_n_keypoints):
    cnnctd_cmp_list = []
    # Using connected components as keypoints (later in the code they will be uniquely identified)
    contours, _hierarchy = cv.findContours(mask_marker_fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    n_keypoints_detected = len(contours)
    if n_keypoints_detected < min_n_keypoints:
        return cnnctd_cmp_list # return empty list
    #filter_contours_by_min_area(contours, min_contour_area) # TODO: add min_contour_area to config file

    for cntr in contours:
        centre_u, centre_v = calculate_contour_centre(cntr)
        kpt = Keypoint(centre_u, centre_v)
        kpt.add_contour(cntr)
        cnnctd_cmp_list.append(kpt)

    return cnnctd_cmp_list


def find_keypoints(mask_marker_fg, min_n_keypoints, max_ang_diff, sequence_length):
    cnnctd_cmp_list = get_connected_components(mask_marker_fg, min_n_keypoints)
    if not cnnctd_cmp_list:
        return None # Not enough connected components detected
    # Group the connected components in sequences
    sequences = group_keypoint_in_sequences(cnnctd_cmp_list, max_ang_diff, sequence_length)
    # TODO: Identify keypoints

    return keypoints_list

import math
import copy
import parse

import cv2 as cv
import numpy as np
from skspatial.objects import Point, Line


class Keypoint:

    NAME_CNT_UV = 'centre_uv'
    NAME_CNT_XYZ = 'centre_xyz'
    NAME_CRN_XYZ = 'corner_xyz'

    def __init__(self, label=-1, kpt_id=-1):
        self.label = label
        self.kpt_id = kpt_id
        self.used = False # For grouping in sequences (keypoint detection)

    def set_label_and_id(self, label, kpt_id):
        self.label = label
        self.kpt_id = kpt_id

    def set_centre_uv(self, centre_u, centre_v):
        self.centre_u = centre_u
        self.centre_v = centre_v

    def get_centre_uv(self):
        return self.centre_u, self.centre_v

    def get_centre_info(self):
        return [self.centre_u, self.centre_v], self.xyz_centre

    def set_corner_pts_uv(self, corners_uv):
        self.corners_uv = corners_uv

    def calculate_xyz(self, radius, mm_per_pixel, marker_width, tmp_uv):
        u, v = tmp_uv
        x = v * mm_per_pixel
        alpha = math.radians((u/marker_width) * 360.0)
        y = radius * math.sin(alpha)
        z = radius * math.cos(alpha)
        return x, y, z

    def calculate_xyz_of_centre_and_corners(self, rad, mm_pixel, width):
        tmp_uv = [self.centre_u, self.centre_v]
        x, y, z = self.calculate_xyz(rad, mm_pixel, width, tmp_uv)
        self.xyz_centre = [x, y, z]
        self.xyz_corners = []
        for tmp_uv in self.corners_uv:
            x, y, z = self.calculate_xyz(rad, mm_pixel, width, tmp_uv)
            self.xyz_corners.append([x, y, z])

    def set_xyz_of_centre_and_corners(self, data_marker_kpt):
        self.xyz_centre = data_marker_kpt['centre_xyz']
        kpt_corners_data = data_marker_kpt[self.NAME_CRN_XYZ]
        self.xyz_corners = []
        for corner_id in kpt_corners_data:
            tmp_xyz = kpt_corners_data[corner_id]
            self.xyz_corners.append(tmp_xyz)

    def set_contour(self, cntr):
        self.cntr = cntr
        self.cntr_area = cv.contourArea(cntr)
        [du, dv, u, y] = cv.fitLine(cntr, cv.DIST_L2, 0, 0.01, 0.01)
        angle_rads = math.atan2(dv, du)
        self.cntr_angle_rads = angle_rads

    def calculate_elongation(self):
        """ ref: https://stackoverflow.com/questions/14854592/retrieve-elongation-feature-in-python-opencv-what-kind-of-moment-it-supposed-to """
        if self.cntr is not None:
            m = cv.moments(self.cntr)
            x = m['mu20'] + m['mu02']
            y = 4 * m['mu11']**2 + (m['mu20'] - m['mu02'])**2
            if (x - y**0.5) < 0.0001:
                self.elong = 0.0 # TODO: maybe I should set to None
            self.elong = (x + y**0.5) / (x - y**0.5)

    def set_anchor_du_dv(self, anchor_du, anchor_dv):
        self.anchor_du = anchor_du
        self.anchor_dv = anchor_dv


class Sequence:

    NAME_SQNC = 'sequence_{}'
    NAME_CODE = 'code'
    NAME_KPT_IDS = 'kpt_ids'

    def __init__(self, list_kpts, sqnc_id=-1):
        self.list_kpts = list_kpts
        if sqnc_id != -1:
            self.sqnc_id = self.NAME_SQNC.format(sqnc_id)
        else:
            self.sqnc_id = sqnc_id

    def get_sqnc_id_int(self):
        if self.sqnc_id == -1:
            return self.sqnc_id
        parsed = parse.parse(self.NAME_SQNC, self.sqnc_id)
        return int(parsed[0])

    def set_angl_median(self, angl_median):
        self.angl_median = angl_median

    def calculate_avrg_area(self):
        area_sum = 0.0
        for counter, kpt in enumerate(self.list_kpts):
            area_sum += kpt.cntr_area
        self.avrg_area = area_sum / counter

    def get_code(self):
        code = []
        for kpt in self.list_kpts:
            code.append(kpt.label)
        return code

    def get_kpt_ids(self):
        kpt_ids = []
        for kpt in self.list_kpts:
            kpt_ids.append(kpt.kpt_id)
        return kpt_ids

    def get_code_and_kpt_ids(self):
        return self.get_code(), self.get_kpt_ids()

    def identify_and_copy_kpt_data(self, kpt_ids, data_marker):
        for i, kpt in enumerate(self.list_kpts):
            kpt.kpt_id = kpt_ids[i]
            data_marker_kpt = data_marker[kpt.kpt_id]
            kpt.set_xyz_of_centre_and_corners(data_marker_kpt)

    def label_keypoints(self, max_ang_diff_label):
        """
        # Label by area
        self.calculate_avrg_area()
        for kpt in self.list_kpts:
            if kpt.cntr_area < self.avrg_area:
                kpt.label = 0
            else:
                kpt.label = 1
        """
        for kpt in self.list_kpts:
            if kpt.label == -1:
                kpt_angle_rads = kpt.cntr_angle_rads
                kpt_angle = math.degrees(kpt_angle_rads % (2 * math.pi)) # get angle between [0 and 360[
                if kpt_angle >= 180.0:
                    kpt_angle -= 180.0
                kpt.label = 0
                if (180.0 - self.angl_median) < max_ang_diff_label:
                    if kpt_angle < max_ang_diff_label:
                        kpt_angle += 180.0
                elif self.angl_median < max_ang_diff_label:
                    if kpt_angle > 180.0 - max_ang_diff_label:
                        kpt_angle -= 180.0
                # Make comparison
                if abs(kpt_angle - self.angl_median) < max_ang_diff_label:
                    kpt.label = 1


class Pattern:

    def __init__(self, list_sqnc):
        self.list_sqnc = list_sqnc

    def get_identified_sqnc_list(self):
        list_sqnc_identified = []
        for sqnc in self.list_sqnc:
            if sqnc.sqnc_id != -1:
                list_sqnc_identified.append(sqnc)
        return list_sqnc_identified

    def get_data_for_pnp_solver(self):
        pnts_3d_object = []
        pnts_2d_image = []
        for sqnc in self.list_sqnc:
            if sqnc.sqnc_id != -1:
                for kpt in sqnc.list_kpts:
                    uv_centre, xyz_centre = kpt.get_centre_info()
                    pnts_3d_object.append([[xyz_centre[0]], [xyz_centre[1]], [xyz_centre[2]]])
                    pnts_2d_image.append([[uv_centre[0]], [uv_centre[1]]])
        pnts_3d_object = np.asarray(pnts_3d_object, dtype=np.float)
        pnts_2d_image = np.asarray(pnts_2d_image, dtype=np.float)
        return pnts_3d_object, pnts_2d_image



def draw_contours(im, contours, color):
    for cntr in contours:
        cv.drawContours(im, [cntr], -1, color, 1)
    return im


def find_angles_with_other_keypoints(kpt_anchor, kpts_list, max_ang_diff):
    anchor_u, anchor_v = kpt_anchor.get_centre_uv()
    angles = []
    angles_kpts = []
    for kpt in kpts_list: # TODO: could be done in parallel
        if kpt is not kpt_anchor and not kpt.used:
            u, v = kpt.get_centre_uv()
            """
              -----> (u)
              |    90*
              v 0* x 180*
             (v)  270*
            """
            du = (anchor_u - u)
            dv = (anchor_v - v)
            kpt.set_anchor_du_dv(du, dv)

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
              In other words, since 1* and 181* is so different (> `max_ang_diff_group`) when
              we assign a group only either _1 or _2 will be used:

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
    # Convert to numpy arrays
    angles = np.array(angles)
    angles_kpts = np.array(angles_kpts)
    # Sort them
    angles_argsort = np.argsort(angles)
    return angles[angles_argsort], angles_kpts[angles_argsort]


def sort_kpts_by_dist_to_anchor(sqnc_kpts, angl_median, kpt_anchor):
    angl_median = 180.0 - angl_median
    angl_median_rad = math.radians(angl_median)

    dist_anchor = [] # store distances to anchor point
    for kpt in sqnc_kpts:
        u_new = - kpt.anchor_du * math.cos(angl_median_rad) + kpt.anchor_dv * math.sin(angl_median_rad)
        dist_anchor.append(u_new)

    # Add anchor point too
    sqnc_kpts = np.append(sqnc_kpts, kpt_anchor)
    dist_anchor.append(0.0) # Distance from anchor to anchor = 0.0

    dist_anchor_argsort = np.argsort(dist_anchor)
    return sqnc_kpts[dist_anchor_argsort]


def find_sequence(kpt_anchor, angles, angles_kpts, max_ang_diff, sequence_length):
    # Find a group of angles of size `sequence_length`.
    # In that group, the `max - min` must be smaller than max_ang_diff.
    """
        example (sequence_length = 4):

        iteration=1
        [ 3.5  4.3  5.2  6.9  7.8  8.3  8.5  9.8  9.9  9.9]
          ___  ___  ___  ___
           ^              ^
           |              |
         ind_head      ind_tail

        iteration=2
        [ 3.5  4.3  5.2  6.9  7.8  8.3  8.5  9.8  9.9  9.9]
               ___  ___  ___  ___
                ^              ^
                |              |
             ind_head      ind_tail

        iteration=N
        [ 3.5  4.3  5.2  6.9  7.8  8.3  9.5  9.8  9.9  9.9]
                                        ___  ___  ___  ___
                                         ^              ^
                                         |              |
                                   ind_head_max      ind_tail
    """
    n_angles = len(angles)
    ind_head_max = n_angles - sequence_length
    sqnc = None
    for ind_head, ang_head in enumerate(angles[0:ind_head_max + 1]): # + 1 since exclusive
        ind_tail = ind_head + sequence_length - 1
        if (angles[ind_tail] - angles[ind_head]) < max_ang_diff:
            if ind_head < ind_head_max:
                # If there are more angles we will check if the next angle also respects
                # the same condition, if so it means that there are more than n=`sequence_length`
                # keypoints detected in that line, so we will skip it
                while ind_tail < ind_head_max:
                    if (angles[ind_tail + 1] - angles[ind_head]) > max_ang_diff:
                        break # the while
                    ind_tail += 1
            # Sequence found!
            sqnc_kpts = angles_kpts[ind_head:ind_tail + 1] # + 1 since exclusive
            angl_median = np.median(angles[ind_head:ind_tail + 1]) # + 1 since exclusive
            sqnc_kpts = sort_kpts_by_dist_to_anchor(sqnc_kpts, angl_median, kpt_anchor)
            # Create sequence object
            sqnc = Sequence(sqnc_kpts)
            sqnc.set_angl_median(angl_median)
            break

    return sqnc


def show_centres(im, sqnc, color):
    for kpt in sqnc.list_kpts:
        u, v = kpt.get_centre_uv()
        im = cv.circle(im, (int(round(u)), int(round(v))), radius=1, color=color)
    cv.imshow("Centres", im)
    cv.waitKey(0)


def fit_line_and_adjust_keypoint_centres(im, sqnc):
    # ref: https://scikit-spatial.readthedocs.io/en/stable/gallery/projection/plot_point_line.html#sphx-glr-download-gallery-projection-plot-point-line-py
    im_copy = im.copy()
    red = (0, 0, 255)
    green = (0, 255, 0)
    #show_centres(im_copy, sqnc, red)
    points = []
    for kpt in sqnc.list_kpts:
        u, v = kpt.get_centre_uv()
        points.append([[u], [v]])
    points = np.asarray(points, dtype=np.float)
    #vx, vy, x0, y0 = cv.fitLine(points=points, distType=cv.DIST_WELSCH, param=0, reps=0.01, aeps=0.01)
    vx, vy, x0, y0 = cv.fitLine(points=points, distType=cv.DIST_FAIR, param=0, reps=0.01, aeps=0.01)
    #, where (vx, vy) is a normalized vector collinear to the line and (x0, y0) is a point on the line
    line = Line(point=[x0[0], y0[0]], direction=[vx[0], vy[0]])
    for kpt in sqnc.list_kpts:
        u, v = kpt.get_centre_uv()
        pt = Point([u, v])
        pt_proj = line.project_point(pt)
        kpt.set_centre_uv(pt_proj[0], pt_proj[1])
    #show_centres(im_copy, sqnc, green)
    return sqnc


def group_keypoint_in_sequences(im, sqnc_kpts, max_ang_diff, sequence_length, min_detected_sqnc):
    # ref: https://www.cs.princeton.edu/courses/archive/spring03/cs226/assignments/lines.html
    n_keypoints = len(sqnc_kpts)
    used_kpts_counter = 0
    sqnc_list = []
    for kpt_anchor in sqnc_kpts:
        if (n_keypoints - used_kpts_counter) < sequence_length:
            # it won't be possible to find another sequence
            # since we would need at least n = sequence_length
            # keypoints that were not used yet
            break
        if not kpt_anchor.used:
            # Find angles that the `kpt_anchor` does with the other `kpts` that were not used yet
            angles, angles_kpts = find_angles_with_other_keypoints(kpt_anchor, sqnc_kpts, max_ang_diff)
            sqnc = find_sequence(kpt_anchor, angles, angles_kpts, max_ang_diff, sequence_length - 1) # - 1 since we are not including the anchor
            if sqnc is not None:
                # Flag those keypoints as used
                for kpt in sqnc.list_kpts:
                    kpt.used = True
                    used_kpts_counter += 1
                if len(sqnc.list_kpts) == sequence_length:
                    #sqnc = fit_line_and_adjust_keypoint_centres(im, sqnc) # The improvement is almost neglectable
                    sqnc_list.append(sqnc)
    if len(sqnc_list) < min_detected_sqnc:
        return None
    pttrn = Pattern(sqnc_list)
    return pttrn


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
    # Using connected components as keypoints (later in the code they will be uniquely identified)
    contours, _hierarchy = cv.findContours(mask_marker_fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    n_keypoints_detected = len(contours)
    if n_keypoints_detected < min_n_keypoints:
        return None

    cnnctd_cmp_list = []
    for cntr in contours:
        centre_u, centre_v = calculate_contour_centre(cntr)
        kpt = Keypoint()
        kpt.set_centre_uv(centre_u, centre_v)
        kpt.set_contour(cntr)
        cnnctd_cmp_list.append(kpt)

    return cnnctd_cmp_list


def show_labels(im, sqnc):
    im_copy = im.copy()
    red = (0, 0, 255)
    for kpt in sqnc.list_kpts:
        tmp_u, tmp_v = kpt.get_centre_uv()
        tmp_label = kpt.label
        im_copy = cv.putText(im_copy,'{}'.format(tmp_label),
              (int(tmp_u), int(tmp_v)),
              cv.FONT_HERSHEY_SIMPLEX,
              0.2,
              red,
              1)
        im_copy = cv.circle(im_copy, (int(round(tmp_u)), int(round(tmp_v))), radius=1, color=red)
    cv.imshow("Code", im_copy)
    cv.waitKey(0)


def find_code_match(im, sqnc, data_pttrn, name_code, name_kpt_ids, used_ind):
    #show_labels(im, sqnc)
    sqnc_code = sqnc.get_code()
    for ind, name_sqnc in enumerate(data_pttrn):
        if ind not in used_ind:
            pttrn_sqnc = data_pttrn[name_sqnc]
            pttrn_code = pttrn_sqnc[name_code]
            if sqnc_code == pttrn_code: #the equality operator == compares a list element-wise
                used_ind.append(ind)
                sqnc.sqnc_id = name_sqnc
                return sqnc, pttrn_sqnc[name_kpt_ids], used_ind
            elif sqnc_code == pttrn_code[::-1]: # check reverted pattern too
                used_ind.append(ind)
                sqnc.sqnc_id = name_sqnc
                return sqnc, pttrn_sqnc[name_kpt_ids][::-1], used_ind
    return sqnc, None, used_ind


def identify_sequence_and_keypoints(im, pttrn, max_ang_diff_label, data_pttrn, sequence_length, min_detected_sqnc, data_marker):
    # Label keypoints as 0 or 1
    for sqnc in pttrn.list_sqnc:
        sqnc.label_keypoints(max_ang_diff_label)

    # Identify keypoints
    used_ind = [] # keep track of the sequences that were already found
    for sqnc in pttrn.list_sqnc:
        sqnc, kpt_ids, used_ind = find_code_match(im, sqnc, data_pttrn, sqnc.NAME_CODE, sqnc.NAME_KPT_IDS, used_ind)
        if kpt_ids is not None:
            sqnc.identify_and_copy_kpt_data(kpt_ids, data_marker[sqnc.sqnc_id])

    if len(used_ind) < min_detected_sqnc:
        return None
    return pttrn


def remove_outlier_sequences(pttrn, sqnc_max_ind, min_detected_sqnc, max_detected_sqnc):
    """
      Check if those sequences can be seen simultaneously by the camera.
      If not we will remove the ones that don't make part of the group.

      Imagine that you have the following pattern, with sqnc_max_ind = 9
      and max_detected_sqnc = 3
        0  1  2  3  4  5  6  7  8  9
       |_______|
          |_______|
                     ...               0  1
                            |_______|
                               |____| |_|
                                  |_| |____| -> E.g., `sequence_9` could be seen with `sequence_0` or `sequence_1`

      Out of all the ranges (|___|) we will use the one with more inliers,
      the other sequences will be considered outliers.
    """
    list_sqnc_identified = pttrn.get_identified_sqnc_list()
    n_sqnc_identified = len(list_sqnc_identified)
    counter_inliers_max = -1
    ranges_with_more_inliers = None
    for head in range(sqnc_max_ind + 1):
        ranges = []
        tail = head + max_detected_sqnc - 1
        if tail <= sqnc_max_ind:
            ranges.append([head, tail])
        else:
            tail -= (sqnc_max_ind + 1)
            ranges.append([head, sqnc_max_ind])
            ranges.append([0, tail])
        # count how many of the sequences are inside either of the ranges
        counter_inliers = 0
        for sqnc in list_sqnc_identified:
            sqnc_id = sqnc.get_sqnc_id_int()
            for (rng_min, rng_max) in ranges:
                if sqnc_id >= rng_min and sqnc_id <= rng_max:
                    counter_inliers += 1
                    break # the inner for loop
        # if all the detected sqnc are inside either of the ranges, then they are all inliers
        if n_sqnc_identified == counter_inliers:
            return pttrn
        # else, update
        if counter_inliers > counter_inliers_max:
            counter_inliers_max = counter_inliers
            ranges_with_more_inliers = ranges

    # The number of inlier sequences should be between min_detected_sqnc and max_detected_sqnc
    if counter_inliers_max < min_detected_sqnc or counter_inliers_max > max_detected_sqnc:
        return None

    # Change the sqnc_id to -1 on the outliers
    for sqnc in list_sqnc_identified:
        sqnc_id = sqnc.get_sqnc_id_int()
        outlier = True
        for (rng_min, rng_max) in ranges_with_more_inliers:
            if sqnc_id >= rng_min and sqnc_id <= rng_max:
                outlier = False
                break # the inner for loop
        if outlier:
            sqnc.sqnc_id = -1

    return pttrn


def find_keypoints(im, mask_marker_fg, config_file_data, sqnc_max_ind, sequence_length, data_pttrn, data_marker):
    # Load data needed to find sequences of keypoints
    min_detected_sqnc = config_file_data['min_detected_sqnc']
    max_detected_sqnc = config_file_data['max_detected_sqnc']
    max_ang_diff_group = config_file_data['max_angle_diff_group']
    max_ang_diff_label = config_file_data['max_angle_diff_label']

    min_n_keypoints = min_detected_sqnc * sequence_length
    cnnctd_cmp_list = get_connected_components(mask_marker_fg, min_n_keypoints)
    if cnnctd_cmp_list is None:
        return None # Not enough connected components detected
    # Group the connected components in sequences
    pttrn = group_keypoint_in_sequences(im, cnnctd_cmp_list, max_ang_diff_group, sequence_length, min_detected_sqnc)
    if pttrn is None:
        return None # Not enough lines detected
    # Identify keypoints
    pttrn = identify_sequence_and_keypoints(im, pttrn, max_ang_diff_label, data_pttrn, sequence_length, min_detected_sqnc, data_marker)
    if pttrn is None:
        return None # Not enough lines identified
    # Remove outlier sequences (set sqnc.sqnc_id = -1, if it is an outlier)
    pttrn = remove_outlier_sequences(pttrn, sqnc_max_ind, min_detected_sqnc, max_detected_sqnc)
    return pttrn

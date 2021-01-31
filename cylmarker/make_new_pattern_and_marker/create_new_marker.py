from cylmarker import load_data, keypoints
import os
import math

import svgwrite


def draw_dash_and_dot(new_marker, kpt, u, v, feature_size_u_half, feature_size_v_half, feature_color):
    """  a_b
         | |
         |x|   x = (u, v)
         |_|
         d c

         e_f
         |x|   x = (u, v)
         g-h
    """
    # Rectangle
    a = [u - feature_size_u_half, v - feature_size_v_half]
    b = [u + feature_size_u_half, v - feature_size_v_half]
    c = [u + feature_size_u_half, v + feature_size_v_half]
    d = [u - feature_size_u_half, v + feature_size_v_half]
    # Square
    e = [u - feature_size_u_half, v - feature_size_u_half]
    f = [u + feature_size_u_half, v - feature_size_u_half]
    g = [u + feature_size_u_half, v + feature_size_u_half]
    h = [u - feature_size_u_half, v + feature_size_u_half]
    if kpt.label == 1:
        """ Rectangle """
        points = [a, b, c, d, a]
    elif kpt.label == 0:
        """ Square """
        points = [e, f, g, h, e]
    # Draw
    new_marker.add(new_marker.polygon(points=points,
        stroke='none',
        fill=feature_color)
    )

    kpt.set_centre_uv(u, v)
    kpt.set_corner_pts_uv(points[:-1]) # Exclude last point, which repeats

    return new_marker, kpt


def draw_marker(data_dir, config_file_data, new_pttrn):
    """ marker
         sequence_0  sequence_1 ...
        .---------------------> (u)
        |     x        x
        |     x        x    ...
        |     x        x
        v     x        x
       (v)
    """
    marker_data = config_file_data['new_marker']
    # There will be one feature per every value `x` of a sequence.
    pttrn_size = config_file_data['new_pattern']['pattern_size']
    n_sequences = pttrn_size['n_sequences']
    n_keypoints = pttrn_size['sequence_length']
    # Get feature size and its margins
    feature_size = marker_data['feature_size_pixels']
    feature_size_u = feature_size['horizontal']
    feature_size_v = feature_size['vertical']
    feature_size_u_half = feature_size_u / 2.0
    feature_size_v_half = feature_size_v / 2.0
    feature_margin = marker_data['feature_margin_pixels']

    # Define colors
    black = svgwrite.rgb(0, 0, 0, '%')
    white = svgwrite.rgb(100, 100, 100, '%')
    green = svgwrite.rgb(0, 100, 0, '%')

    # Calculate marker width
    feature_margin_u = feature_margin['horizontal']
    marker_width = n_sequences * (feature_size_u + feature_margin_u)

    # Calculate marker height
    feature_margin_v = feature_margin['vertical']
    marker_margin_v = marker_data['marker_margin_pixels']['vertical']
    """
        ------
        |       ]margin_v
        |   x   ]feature_size_v
        |       ]feature_margin_v
        |   x   ]feature_size_v
        |       ]feature_margin_v
        |   x   ]feature_size_v
        |       ]margin_v
        ------
    """
    marker_margin_v_total = 2 * marker_margin_v # since we add the margin on top and bottom
    marker_height = marker_margin_v_total + n_keypoints * feature_size_v + (n_keypoints - 1) * feature_margin_v
    use_hex_dist = marker_data['use_hexagonal_distribution']
    if use_hex_dist:
        """   non hex.    vs.    hex
          | x  x  x  x  x | x     x    x |
          |               |    x     x   |
          | x  x  x  x  x | x     x    x |
          |               |    x     x   |

            on the hexagonal grid there is a vertical shift = feature_size_v_half + feature_margin_v / 2,
            on the even columns, so that they intercalate.
        """
        hexagonal_shift_v = feature_size_v_half + feature_margin_v / 2.
        marker_height += hexagonal_shift_v

    # Get path of where to save the marker
    file_marker = load_data.get_path_marker_img(data_dir)
    # Make marker
    new_marker = svgwrite.Drawing(file_marker, profile='tiny')
    # Paint background
    fill_color = white
    if marker_data['draw_green_bg']:
        fill_color = green
    new_marker.add(new_marker.rect((0, 0), (marker_width, marker_height), stroke='none', fill=fill_color))

    # Add borders, to the top and bottom
    if marker_data['add_border_top_and_bot']:
        thck = marker_data['border_thickness_pixels']
        top = thck/2.0
        bot = marker_height - thck/2.0
        new_marker.add(new_marker.line((0, top), (marker_width, top), stroke_width=thck, stroke=black))
        new_marker.add(new_marker.line((0, bot), (marker_width, bot), stroke_width=thck, stroke=black))

    # Get cylinder's radius and mm per pixel
    radius, mm_per_pixel = load_data.get_marker_radius_and_mmperpixel(config_file_data, marker_width)

    # Paint keypoints
    """
      To centre the pattern in the u direction I will add `feature_margin_u_half` to the first sequence.
      This way instead of getting this (see below), we get this (see below):
                                 |x    x    x    |        |  x    x    x  |
                                 |x    x    x    |        |  x    x    x  |
                                 |x    x    x    |        |  x    x    x  |
                 margin spaces:  (0)         (4)           (2)          (2)
    """
    feature_margin_u_half = feature_margin_u / 2.0
    init_u = feature_size_u_half + feature_margin_u_half
    delta_u = feature_size_u + feature_margin_u
    init_v = marker_margin_v + feature_size_v_half
    delta_v = feature_size_v + feature_margin_v

    # Draw marker keypoints according to the `new_pttrn`
    for i, sqnc in enumerate(new_pttrn.list_sqnc):
        u = init_u + i * delta_u
        shift_v = 0
        # in the hexagonal we want to align with the centre of the gap between keypoints
        if use_hex_dist and i % 2 == 1:
            shift_v = hexagonal_shift_v
        for j, kpt in enumerate(sqnc.list_kpts):
            v = init_v + shift_v +  j * delta_v
            new_marker, kpt = draw_dash_and_dot(new_marker, kpt, u, v, feature_size_u_half, feature_size_v_half, black)
            kpt.calculate_xyz_of_centre_and_corners(radius, mm_per_pixel, marker_width)

    return new_marker, new_pttrn





from cylmarker import load_data
import os

import svgwrite


def draw_corner(new_marker, u, v, corner_size_half, val_bool, corner_color):
    """
        a  b
         \/
         /\
        c  d
    """
    a = [u - corner_size_half, v - corner_size_half]
    b = [u + corner_size_half, v - corner_size_half]
    c = [u - corner_size_half, v + corner_size_half]
    d = [u + corner_size_half, v + corner_size_half]
    if val_bool:
        points = [a, b, c, d, a]
    else:
        points = [a, c, b, d, a]
    new_marker.add(new_marker.polygon(points=points,
        stroke='none',
        fill=corner_color)
    )
    return new_marker


def draw_marker(data_dir, config_file_data, new_pttrn):
    """ marker
            code_0   code_1 ...
        .---------------------> (u)
        |     x        x
        |     x        x    ...
        |     x        x
        v     x        x
       (v)
    """
    marker_data = config_file_data['new_marker']
    print(marker_data)
    # There will be one corner per every value `x` of a code.
    pttrn_size = config_file_data['new_pattern']['pattern_size']
    n_codes = pttrn_size['n_codes']
    n_corners = pttrn_size['code_length']
    # Get corner size and its margins
    corner_size = marker_data['corner_size_pixels']
    corner_size_half = corner_size / 2.0
    corner_margin = marker_data['corner_margin_pixels']

    # Define colors
    black = svgwrite.rgb(0, 0, 0, '%')
    white = svgwrite.rgb(100, 100, 100, '%')
    green = svgwrite.rgb(0, 100, 0, '%')

    # Calculate marker width
    corner_margin_u = corner_margin['horizontal']
    marker_width = n_codes * (corner_size + corner_margin_u)
    print(marker_width)

    # Calculate marker height
    corner_margin_v = corner_margin['vertical']
    marker_margin_v = marker_data['marker_margin_pixels']['vertical']
    """
        ------
        |       ]margin_v
        |   x   ]corner_size
        |       ]corner_margin_v
        |   x   ]corner_size
        |       ]corner_margin_v
        |   x   ]corner_size
        |       ]margin_v
        ------
    """
    marker_margin_v_total = 2 * marker_margin_v # since we add the margin on top and bottom
    marker_height = marker_margin_v_total + n_corners * corner_size + (n_corners - 1) * corner_margin_v
    use_hex_dist = marker_data['use_hexagonal_distribution']
    if use_hex_dist:
        """   non hex.    vs.    hex
          | x  x  x  x  x | x     x    x |
          |               |    x     x   |
          | x  x  x  x  x | x     x    x |
          |               |    x     x   |

            on the hexagonal grid there is a shift = corner_size_half + corner_margin_v / 2,
            on the even columns, so that they intercalate.
        """
        hexagonal_shift_v = corner_size_half + corner_margin_v / 2.
        marker_height += hexagonal_shift_v
    #marker_height = marker_margin_v_total + n_corners * corner_size + (n_corners - 1) * corner_margin_v + corner_margin_v / 2.0
    print(marker_height)

    # Get path of where to save the marker
    file_marker = load_data.get_path_marker(data_dir)
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

    # Paint corners
    """
      To centre the pattern in the u direction I will add `corner_margin_u_half` to the first code.
      This way instead of getting this (see below), we get this (see below):
                                 |x    x    x    |        |  x    x    x  |
                                 |x    x    x    |        |  x    x    x  |
                                 |x    x    x    |        |  x    x    x  |
    """
    corner_margin_u_half = corner_margin_u / 2.0
    init_u = corner_size_half + corner_margin_u_half
    delta_u = corner_size + corner_margin_u
    init_v = marker_margin_v + corner_size_half
    delta_v = corner_size + corner_margin_v
    for i, code in enumerate(new_pttrn):
        print(code)
        u = init_u + i * delta_u
        shift_v = 0
        # in the hexagonal we want to align with the centre of the gap between corners
        if use_hex_dist and i % 2 == 1:
            shift_v = hexagonal_shift_v
        for j, val_bool in enumerate(code):
            v = init_v + shift_v +  j * delta_v
            new_marker = draw_corner(new_marker, u, v, corner_size_half, val_bool, black)

    # Draw marker corners according to the `new_pttrn`
    return new_marker







def draw_marker(config_file_data, new_pttrn):
    """ marker
        .---------------------> (u)
        |     x        x
        |     x        x
        |     x        x
        v     x        x
       (v)  code_0   code_1 ...
    """
    marker_data = config_file_data['new_marker']
    print(marker_data)
    # There will be one corner per every value `x` of a code.
    n_codes = len(new_pttrn)
    n_corners = len(new_pttrn[0]) # same as code_length
    # Get corner size and its margins
    corner_size = marker_data['corner_size_pixels']
    corner_margin = marker_data['corner_margin_pixels']

    # Calculate marker width
    corner_margin_u = corner_margin['vertical']
    marker_width = n_codes * (corner_size + corner_margin_u)
    print(marker_width)

    # Calculate marker height
    marker_margin_v = marker_data['marker_margin_pixels']['vertical']
    marker_margin_v_total = 2 * marker_margin_v # since we add the margin on top and bottom
    corner_margin_v = corner_margin['vertical']
    n_cells = n_corners
    if marker_data['use_hexagonal_distribution']:
        """ non hex. vs. hex
            x x x    |  x   x
                     |    x
            x x x    |  x   x
                     |    x
        """
        n_cells += 1
    marker_height = marker_margin_v_total + n_cells * (corner_size + corner_margin_v)
    print(marker_height)

    # Get marker background
    # Draw marker corners according to the `new_pttrn`
    # return new_marker





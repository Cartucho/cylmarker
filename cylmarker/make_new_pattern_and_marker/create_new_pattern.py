



def arr_equal_in_list(my_arr, list_arrays):
    """ check if a numpy array already exists in list """
    # ref: https://stackoverflow.com/questions/23979146/check-if-numpy-array-is-in-list-of-numpy-arrays
    return next((True for elem in list_arrays if np.array_equal(elem, my_arr)), False)


#def get_list_of_random_possible_codes(, max_n_codes):
#max_patterns = 2**N_BLOB_ROWS


def get_new_pttrn(config_file_data):
    print('creating new pattern')

    new_pttrn = None
    return new_pttrn

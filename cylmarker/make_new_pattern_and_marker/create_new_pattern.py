import random

import itertools
import numpy as np
from tqdm import tqdm


def get_list_of_possible_codes(pattern_size, max_n_random_codes):
    # Get number of codes in the pattern (number of columns)
    n_codes = pattern_size['n_codes']
    # Get number of elements per code
    code_length = pattern_size['code_length']


    size_of_random_part_of_code = code_length - 2 # -2 since the `value_start` and `value_end` are fixed
    possible_codes = []
    """ For large codes it is better to randomly generate them.
        However, if the code is small enough we can store all
        the possible combinations in the `possible_codes`.

        Although the use may think of the code as `0`s and `1`s,
        we will be using booleans instead of ints to consume less memory.
    """
    print('\nFinding the possible codes...')
    n_possible_codes = 2 ** size_of_random_part_of_code
    if n_possible_codes < max_n_random_codes:
        possible_codes = []
        for code in tqdm(itertools.product([True, False], repeat=size_of_random_part_of_code), total=n_possible_codes):
            possible_codes.append(code)
    else:
        for i in tqdm(range(max_n_random_codes)):
            it = np.random.choice([True, False], size=size_of_random_part_of_code)
            possible_codes.append(it)
    return possible_codes


def get_tmp_pattern_and_strength(possible_codes, comb):
    tmp_pattern = []
    for ind in comb:
        tmp_pattern.append(np.asarray(possible_codes[ind]))
    strength = 0
    for comb in itertools.combinations(range(len(tmp_pattern)), 2):
        ind_1, ind_2 = comb
        strength += sum([k != l for k, l in zip(tmp_pattern[ind_1], tmp_pattern[ind_2])])
    return tmp_pattern, strength


def get_best_pattern_from_possible_codes(pttrn_size, possible_codes, n_iter_max):
    n_codes = pttrn_size['n_codes']
    """ Select the best pattern.
        We randomly choose `n_codes` out of all the `possible_codes`.
        Then we check its strength by comparing each code to all the other
        codes inside the same pattern. In other words, we want the codes to
        be as different as possible from each other, so we count the differences
        and keep updating the best pattern found so far.
     """

    # First, we need to decide if we go through all the possible combinations or not
    possible_codes_len = len(possible_codes)
    compare_all_codes = True

    # The next line counts the number of possible combinations, while this counter is smaller than `n_iter_max + 1`
    # ref: https://stackoverflow.com/questions/50652393/itertools-combinations-permutations-size
    # ref: https://stackoverflow.com/questions/23722473/limiting-the-number-of-combinations-permutations-in-python/50057642
    n_possible_combinations = sum(1 for ignore in itertools.islice(itertools.combinations(range(len(possible_codes)), n_codes), n_iter_max + 1))
    if n_possible_combinations > n_iter_max:
        compare_all_codes = False

    best_pattern = None
    best_pattern_strength = -1
    # TODO: repeated code in here, clean up needed
    print('\nFinding the best pattern...')
    if compare_all_codes:
        for comb in tqdm(itertools.combinations(range(possible_codes_len), n_codes), total=n_possible_combinations):
            tmp_pattern, strength = get_tmp_pattern_and_strength(possible_codes, comb)
            if strength > best_pattern_strength:
                best_pattern_strength = strength
                best_pattern = tmp_pattern
    else:
        for _i in tqdm(range(n_iter_max)):
            comb = random.sample(range(len(possible_codes)), n_codes)
            tmp_pattern, strength = get_tmp_pattern_and_strength(possible_codes, comb)
            if strength > best_pattern_strength:
                best_pattern_strength = strength
                best_pattern = tmp_pattern

    # TODO: The codes should be sorted to maximize the diff between neighbour codes.
    #       Use `min_detected_lines` to define the number of neighbours of each code.

    return best_pattern


def append_code_value_start_and_end(pttrn, code_val_start, code_val_end):
    pttrn_completed = []
    for code in pttrn:
        # prepend `code_val_start`
        code = np.insert(code, 0, bool(code_val_start), axis=0)
        # append `code_val_end`
        code = np.append(code, bool(code_val_end))
        # save it
        pttrn_completed.append(code)
    return pttrn_completed


def get_new_pttrn(config_file_data):
    pttrn_data = config_file_data['new_pattern']
    pttrn_size = pttrn_data['pattern_size']
    code_val_start = pttrn_data['code_val_start']
    code_val_end   = 1 - code_val_start # if one is `0`, the other is `1`
    """
      We want a random code that starts with a fixed `code_val_start` (pre-defined in the config file).
      If the `code_val_start` is 1 then the last value of the code is 0, and vice-versa to avoid having
      symmetric codes. Since the first and last value of each code is already pre-defined, we only
      need to calculate the inner parts of the code (of size `code_length` - 2). Therefore, here we will
      create and select the best `n_codes` of size `code_length` - 2 to form a pattern. Finally, we will
      append the `code_val_start` and `code_val_end`, to complete the new pattern.
    """
    # Get a number of possible codes (< max_n_random_codes) that can be used in the pattern
    possible_codes = get_list_of_possible_codes(pttrn_size, pttrn_data['max_n_random_codes'])
    # Given the selected possible codes, get the best pattern (with more differences between codes)
    best_pttrn = get_best_pattern_from_possible_codes(pttrn_size, possible_codes, pttrn_data['max_n_iterations_to_optimise_pttrn'])
    # Finally, append the fixed/pre-defined `code_val_start` and `code_val_end`
    pttrn_completed = append_code_value_start_and_end(best_pttrn, code_val_start, code_val_end)
    return pttrn_completed

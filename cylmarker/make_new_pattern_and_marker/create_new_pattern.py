from cylmarker import keypoints
import random

import itertools
import numpy as np
from tqdm import tqdm


def arreq_in_list(myarr, list_arrays):
    """ ref: https://stackoverflow.com/questions/23979146/check-if-numpy-array-is-in-list-of-numpy-arrays """
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)


def get_list_of_possible_sequences(pattern_size, max_n_random_sequences):
    # Get number of sequences in the pattern (number of columns)
    n_sequences = pattern_size['n_sequences']
    # Get number of elements per sequence
    sequence_length = pattern_size['sequence_length']


    size_of_random_part_of_sequence = sequence_length - 2 # -2 since the `value_start` and `value_end` are fixed
    possible_sequences = []
    """ For large sequences it is better to randomly generate them.
        However, if the sequence is small enough we can store all
        the possible combinations in the `possible_sequences`.

        Although the use may think of the sequence as `0`s and `1`s,
        we will be using booleans instead of ints to consume less memory.
    """
    print('\nFinding the possible sequences...')
    n_possible_sequences = 2 ** size_of_random_part_of_sequence
    if n_possible_sequences < max_n_random_sequences:
        for sequence in tqdm(itertools.product([1, 0], repeat=size_of_random_part_of_sequence), total=n_possible_sequences):
            possible_sequences.append(sequence)
    else:
        for i in tqdm(range(max_n_random_sequences)):
            it = np.random.choice([1, 0], size=size_of_random_part_of_sequence)
            if not arreq_in_list(it, possible_sequences):
                # append if it does not exist yet
                possible_sequences.append(it)
    return possible_sequences


def get_tmp_pattern_and_strength(possible_sequences, comb):
    tmp_pattern = []
    for ind in comb:
        tmp_pattern.append(np.asarray(possible_sequences[ind]))
    strength = 0
    for comb in itertools.combinations(range(len(tmp_pattern)), 2):
        ind_1, ind_2 = comb
        strength += sum([k != l for k, l in zip(tmp_pattern[ind_1], tmp_pattern[ind_2])])
    return tmp_pattern, strength


def get_best_pattern_from_possible_sequences(pttrn_size, possible_sequences, n_iter_max):
    n_sequences = pttrn_size['n_sequences']
    """ Select the best pattern.
        We randomly choose `n_sequences` out of all the `possible_sequences`.
        Then we check its strength by comparing each sequence to all the other
        sequences inside the same pattern. In other words, we want the sequences to
        be as different as possible from each other, so we count the differences
        and keep updating the best pattern found so far.
     """

    # First, we need to decide if we go through all the possible combinations or not
    possible_sequences_len = len(possible_sequences)
    compare_all_sequences = True

    # The next line counts the number of possible combinations, while this counter is smaller than `n_iter_max + 1`
    # ref: https://stackoverflow.com/questions/50652393/itertools-combinations-permutations-size
    # ref: https://stackoverflow.com/questions/23722473/limiting-the-number-of-combinations-permutations-in-python/50057642
    n_possible_combinations = sum(1 for ignore in itertools.islice(itertools.combinations(range(len(possible_sequences)), n_sequences), n_iter_max + 1))
    if n_possible_combinations > n_iter_max:
        compare_all_sequences = False

    best_pattern = None
    best_pattern_strength = -1
    print('\nFinding the best pattern...')
    if compare_all_sequences:
        for comb in tqdm(itertools.combinations(range(possible_sequences_len), n_sequences), total=n_possible_combinations):
            tmp_pattern, strength = get_tmp_pattern_and_strength(possible_sequences, comb)
            if strength > best_pattern_strength:
                best_pattern_strength = strength
                best_pattern = tmp_pattern
    else:
        for _i in tqdm(range(n_iter_max)):
            comb = random.sample(range(len(possible_sequences)), n_sequences)
            tmp_pattern, strength = get_tmp_pattern_and_strength(possible_sequences, comb)
            if strength > best_pattern_strength:
                best_pattern_strength = strength
                best_pattern = tmp_pattern

    # TODO: The sequences should be sorted to maximize the diff between neighbour sequences.
    #       Use `min_detected_sqnc` to define the number of neighbours of each sequence.

    return best_pattern


def append_sequence_value_start_and_end(pttrn, sequence_head, sequence_tail):
    pttrn_completed = []
    for sequence in pttrn:
        # prepend `sequence_head`
        sequence = np.insert(sequence, 0, sequence_head, axis=0)
        # append `sequence_val_end`
        sequence = np.append(sequence, sequence_tail)
        # save it
        pttrn_completed.append(sequence)
    return pttrn_completed


def create_keypoints_sequences_and_pattern(pttrn_completed):
    sqnc_id = 0
    kpt_id = 0
    list_sqnc = []
    for pttrn_sq in pttrn_completed:
        list_kpts = []
        for kpt_val in pttrn_sq:
            kpt = keypoints.Keypoint(int(kpt_val), kpt_id)
            kpt_id += 1
            list_kpts.append(kpt)
        sqnc = keypoints.Sequence(list_kpts, sqnc_id)
        sqnc_id += 1
        list_sqnc.append(sqnc)
    return keypoints.Pattern(list_sqnc)


def get_new_pttrn(config_file_data):
    # TODO: instead of comparing the entire pattern e.g., [1, 0, 1, 0, 1, 0] I could just compare the
    #       associated decimal representation of that binary number.
    pttrn_data = config_file_data['new_pattern']
    pttrn_size = pttrn_data['pattern_size']
    sequence_head = pttrn_data['sequence_head']
    sequence_tail = 1 - sequence_head # if one is `0`, the other is `1`
    """
      We want a random sequence that starts with a fixed `sequence_head` (pre-defined in the config file).
      If the `sequence_head` is 1 then the last value of the sequence is 0, and vice-versa to avoid having
      symmetric sequences. Since the first and last value of each sequence is already pre-defined, we only
      need to calculate the inner parts of the sequence (of size `sequence_length` - 2). Therefore, here we will
      create and select the best `n_sequences` of size `sequence_length` - 2 to form a pattern. Finally, we will
      append the `sequence_head` and `sequence_tail`, to complete the new pattern.
    """
    # Get a number of possible sequences (< max_n_random_sequences) that can be used in the pattern
    possible_sequences = get_list_of_possible_sequences(pttrn_size, pttrn_data['max_n_random_sequences'])
    # Given the selected possible sequences, get the best pattern (with more differences between sequences)
    best_pttrn = get_best_pattern_from_possible_sequences(pttrn_size, possible_sequences, pttrn_data['max_n_iterations_to_optimise_pttrn'])
    # Finally, append the fixed/pre-defined `sequence_head` and `sequence_tail`
    pttrn_completed = append_sequence_value_start_and_end(best_pttrn, sequence_head, sequence_tail)
    # Create a pattern object
    new_pttrn = create_keypoints_sequences_and_pattern(pttrn_completed)
    return new_pttrn

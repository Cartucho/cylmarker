from cylmarker import load_data
import os

from distutils.util import strtobool # for query_yes_no()




def query_yes_no(question):
    # ref: https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    print('{} [y/n]'.format(question))
    while True:
        try:
            return strtobool(input().lower())
        except ValueError:
            print('Please respond with \'y\' or \'n\'.\n')


def check_and_warn_if_files_will_be_replaced(data_dir):
    file_pattern = load_data.get_path_pattern(data_dir)
    file_marker  = load_data.get_path_marker(data_dir)
    if os.path.isfile(file_pattern) or os.path.isfile(file_marker):
        message = ('Warning, the current marker/pattern in {} will be DELETED\n'
                   'and replaced by a new one! Do you wish to continue?'.format(data_dir))
        user_wants_to_continue = query_yes_no(message)
        if not user_wants_to_continue:
            print('Then, please choose another path for the new marker'
                  ', using `python --path new_path --task m`')
            exit()


def save_new_pttrn(data_dir, new_pttrn):
    print(data_dir)
    file_pattern = load_data.get_path_pattern(data_dir)
    if not os.path.isfile(file_pattern):
        # TODO: make file
        pass
    else:
        # TODO: Delete old files, the code already asked for permition
        #pattern_dir_deleted = os.path.join(pattern_dir, 'deleted')
        pass
    # TODO: save pattern


def save_new_marker(new_marker):
    new_marker.save()

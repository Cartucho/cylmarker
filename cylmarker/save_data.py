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


def check_and_warn_if_files_will_be_replaced(dir_path):
    dir_pattern = os.path.join(dir_path, 'pattern')
    dir_marker  = os.path.join(dir_path, 'marker')
    if os.path.exists(dir_pattern) or os.path.exists(dir_marker):
        message = ('Warning, the current marker/pattern in {} will be DELETED\n'
                   'and replaced by a new one! Do you wish to continue?'.format(dir_path))
        user_wants_to_continue = query_yes_no(message)
        if not user_wants_to_continue:
            print('Then, please choose another path for the new marker'
                  ', using `python --path new_path --task m`')
            exit()


def save_new_pttrn(dir_path, new_pttrn):
    print(dir_path)
    pattern_dir = os.path.join(dir_path, 'pattern')
    if not os.path.exists(pattern_dir):
        os.makedir(pattern_dir)
    else:
        # TODO: Delete old files, the code already asked for permition
        #pattern_dir_deleted = os.path.join(pattern_dir, 'deleted')
        pass
    # TODO: save pattern

'''
Helper functions
'''

import subprocess
import os


def _git_commit_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short',
                                   'HEAD']).strip()


def git_hash_dir_path(file_dir):
    """
    Create directory named after git commit hash
    """
    save_path = os.path.join(file_dir, _git_commit_short_hash())
    # Path must be created for saving with tf.train.Saver
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path


def git_hash_file_path(file_dir, file_name):
    """
    Include directory named after git commit hash in path to file for
    reproducibility and trace back
    """
    return os.path.join(git_hash_dir_path(file_dir), file_name)

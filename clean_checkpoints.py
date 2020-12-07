import os
import argparse
import shutil
import warnings

import pdb

sub_dirs = ['checkpoints', 'train_plots', 'train_data']

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

def delete_checkpoints(work_dir):
    i = 0
    for root_dir, dirs, _ in os.walk(work_dir):
        for dir in dirs:
            if dir not in sub_dirs:
                to_delete = os.path.join(work_dir,dir,'checkpoints')
                if os.path.isdir(to_delete):
                    shutil.rmtree(to_delete, ignore_errors=True)
                    i+=1
    print('{} directories deleted.'.format(i))

def delete_error_runs(work_dir):
    i = 0
    for root_dir, dirs, _ in os.walk(work_dir):
        for dir in dirs:
            if dir not in sub_dirs:
                to_delete = os.path.join(work_dir,dir)
                if 'train_data' not in os.listdir(to_delete):
                    shutil.rmtree(to_delete, ignore_errors=True)
                    i+=1
    print('{} directories deleted.'.format(i))

def delete_tree(work_dir):
    shutil.rmtree(work_dir)
    print('done.')

def main():
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir")
    parser.add_argument("--type")
    FLAGS = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()

    if FLAGS.type=='checkpoints':
        delete_checkpoints(FLAGS.work_dir)
    elif FLAGS.type=='error_runs':
        delete_error_runs(FLAGS.work_dir)
    elif FLAGS.type=='dir_tree':
        delete_tree(FLAGS.work_dir)
    else:
        assert False, 'unknow {}'.format(FLAGS.type)


if __name__ == '__main__':
    main()

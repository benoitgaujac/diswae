import os
import argparse
import shutil

import pdb


parser = argparse.ArgumentParser()
parser.add_argument("--work_dir",type=str,
                    help='filename to delete')
FLAGS = parser.parse_args()

sub_dirs = ['checkpoints', 'train_plots', 'train_data']

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


delete_checkpoints(FLAGS.work_dir)

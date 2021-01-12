import os
import sys
import logging

from PIL import Image
import numpy as np
import h5py

import pdb


def stage_to_scratch(dataset, src, dst):
    """Load data and save images
    """
    # loading data
    # data_path = os.path.join(src, 'dsprites')
    assert os.path.isdir(src), 'data dir. doesnt exist.'
    if dataset[-8:]=='dsprites':
        filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        filepath = os.path.join(src,filename)
        assert os.path.isfile(filepath), 'dSprites data file doesnt exist.'
        X = (255 * np.load(filepath, allow_pickle=True)['imgs']).astype(np.uint8)#[:,:,:,None]
        mode = 'L'
    elif dataset=='3dshapes':
        filename = '3dshapes.h5'
        filepath = os.path.join(src,filename)
        assert os.path.isfile(filepath), '3dshapes data file doesnt exist.'
        X = np.array(h5py.File(filepath, 'r')['images'])#[:,:,:,None]
        mode = 'RGB'
    else:
        ValueError('Staging to scratch not implemented for %s' % dataset)
    # init dir, img_list, counter
    if not os.path.isdir(dst):
        os.mkdir(dst)
    dest_path = os.path.join(dst,'images')
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)
    img_list_path = os.path.join(dest_path, 'img_list.txt')
    if os.path.isfile(img_list_path):
        with open(os.path.join(dest_path, 'img_list.txt'), 'r') as f:
            names = f.readlines()
            i = int(names[-1])
            f.close()
    else:
        i = 0
    # looping over images
    for n in range(X[i:].shape[0]):
        file_name = '%.6d.jpg' % (n+1+i)
        im = Image.fromarray(X[n+i], mode)
        im.save(os.path.join(dest_path,file_name))
        with open(os.path.join(dest_path, 'img_list.txt'), 'a') as f:
            f.write(file_name[:-4] + '\n')
            f.close()
        if (n+1+i) % 25000 == 0:
            logging.error('{}/{} images saved.'.format(n+1+i,X.shape[0]))
            print('{}/{} images saved.'.format(n+1+i,X.shape[0]))

def main():
    stage_to_scratch('3dshapes','../data/3dshapes','../data/3dshapes')

if __name__ == '__main__':

    main()

import os
from PIL import Image
import numpy as np
import pdb


def exctract_dsprites(data_dir):
    """Load daat and save images
    """
    # loading data
    data_path = os.path.join(data_dir, 'dsprites')
    assert os.path.isdir(data_path), 'dSprites dir. doesnt exist.'
    filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    filepath = os.path.join(data_path,filename)
    assert os.path.isfile(filepath), 'dSprites data file doesnt exist.'
    X = (255 * np.load(filepath, allow_pickle=True)['imgs']).astype(np.uint8)#[:,:,:,None]
    # init dir, img_list, counter
    dest_path = os.path.join(data_path,'images')
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

        im = Image.fromarray(X[n+i], mode='L')
        im.save(os.path.join(dest_path,file_name))
        with open(os.path.join(dest_path, 'img_list.txt'), 'a') as f:
            f.write(file_name[:-4] + '\n')
            f.close()
        if (n+1+i) % 10000 == 0:
            print('{}/{} images saved.'.format(n+1+i,X.shape[0]))

def main():
    exctract_dsprites('../data')

if __name__ == '__main__':

    main()

# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""This class helps to handle the data.

"""

import os
import shutil
import random
import logging
import gzip
import zipfile
import tensorflow as tf
import numpy as np
from six.moves import cPickle
import urllib.request
import requests
from scipy.io import loadmat
from sklearn.feature_extraction import image
import struct
from tqdm import tqdm
from PIL import Image
import sys
import tarfile
import h5py

import utils

import pdb

datashapes = {}
datashapes['dsprites'] = [64, 64, 1]
datashapes['noisydsprites'] = [64, 64, 3]
datashapes['screamdsprites'] = [64, 64, 3]
datashapes['3dshapes'] = [64, 64, 3]
datashapes['smallNORB'] = [64, 64, 1]
datashapes['3Dchairs'] = [64, 64, 3]
datashapes['celebA'] = [64, 64, 3]
datashapes['mnist'] = [28, 28, 1]
datashapes['svhn'] = [32, 32, 3]

SCREAM_PATH = '../data/dsprites/scream.jpg'

def _data_dir(opts):
    data_path = maybe_download(opts)
    return data_path

def maybe_download(opts):
    """Download the data from url, unless it's already here."""
    if not tf.io.gfile.exists(opts['data_dir']):
        tf.io.gfile.makedirs(opts['data_dir'])
    if opts['dataset']=='noisydsprites' or opts['dataset']=='screamdsprites':
        data_path = os.path.join(opts['data_dir'], opts['dataset'][-8:])
    else:
        data_path = os.path.join(opts['data_dir'], opts['dataset'])
    if not tf.io.gfile.exists(data_path):
        tf.io.gfile.makedirs(data_path)
    if opts['dataset'][-8:]=='dsprites':
        filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true'
        file_path = os.path.join(data_path, filename[:-9])
        if not tf.io.gfile.exists(file_path):
            download_file(file_path,filename,opts['DSprites_data_source_url'])
    elif opts['dataset']=='3dshapes':
        filename = '3dshapes.h5'
        file_path = os.path.join(data_path, filename)
        if not tf.io.gfile.exists(file_path):
            assert False, 'To implement'
            download_file(file_path,filename,opts['3dshapes_data_source_url'])
    elif opts['dataset']=='smallNORB':
        filename = 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz'
        file_path = os.path.join(data_path, filename)
        if not tf.io.gfile.exists(file_path):
            download_file(file_path,filename,opts['smallNORB_data_source_url'])
        filename = 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz'
        file_path = os.path.join(data_path, filename)
        if not tf.io.gfile.exists(file_path):
            download_file(file_path,filename,opts['smallNORB_data_source_url'])
    elif opts['dataset']=='3Dchairs':
        filename = 'rendered_chairs.tar'
        file_path = os.path.join(data_path, filename)
        if not tf.io.gfile.exists(file_path):
            download_file(file_path,filename,opts['3Dchairs_data_source_url'])
    elif opts['dataset']=='celebA':
        filename = 'img_align_celeba'
        file_path = os.path.join(data_path, filename)
        if not tf.io.gfile.exists(file_path):
            filename = 'img_align_celeba.zip'
            file_path = os.path.join(data_path, filename)
            if not tf.io.gfile.exists(file_path):
                assert False, '{} dataset does not exist'.format(opts['dataset'])
                download_file_from_google_drive(file_path,filename,opts['celebA_data_source_url'])
            # Unzipping
            print('Unzipping celebA...')
            with zipfile.ZipFile(file_path) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(data_path)
            print('Unzipping done.')
            os.remove(file_path)
            # os.rename(os.path.join(data_path, zip_dir), os.path.join(data_path, 'img_align_celeba'))
        data_path = os.path.join(data_path,'img_align_celeba')
    elif opts['dataset']=='mnist':
        download_file(data_path,'train-images-idx3-ubyte.gz',opts['MNIST_data_source_url'])
        download_file(data_path,'train-labels-idx1-ubyte.gz',opts['MNIST_data_source_url'])
        download_file(data_path,'t10k-images-idx3-ubyte.gz',opts['MNIST_data_source_url'])
        download_file(data_path,'t10k-labels-idx1-ubyte.gz',opts['MNIST_data_source_url'])
    elif opts['dataset']=='svhn':
        download_file(data_path,'train_32x32.mat',opts['SVHN_data_source_url'])
        download_file(data_path,'test_32x32.mat',opts['SVHN_data_source_url'])
        if opts['use_extra']:
            download_file(data_path,'extra_32x32.mat',opts['SVHN_data_source_url'])
    else:
        assert False, 'Unknow dataset'

    return data_path

def download_file(file_path,filename,url):
    file_path, _ = urllib.request.urlretrieve(url + filename, file_path)
    with tf.gfile.GFile(file_path) as f:
        size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')

def download_file_from_google_drive(file_path, filename, url):

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    session = requests.Session()
    id = '0B7EVK8r0v71pZjFTYXZWM3FlRnM'
    response = session.get(url, params={ 'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = { 'id': id, 'confirm': token }
        response = session.get(url, params=params, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(file_path, "wb") as f:
        for chunk in tqdm(response.iter_content(32*1024), total=total_size,
            unit='B', unit_scale=True, desc=file_path):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def load_cifar_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.

    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    # Returns
        A tuple `(data, labels)`.
    """
    f = utils.o_gfile(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

def transform_mnist(pic, mode='n'):
    """Take an MNIST picture normalized into [0, 1] and transform
        it according to the mode:
        n   -   noise
        i   -   colour invert
        s*  -   shift
    """
    pic = np.copy(pic)
    if mode == 'n':
        noise = np.random.randn(28, 28, 1)
        return np.clip(pic + 0.25 * noise, 0, 1)
    elif mode == 'i':
        return 1. - pic
    pixels = 3 + np.random.randint(5)
    if mode == 'sl':
        pic[:, :-pixels] = pic[:, pixels:] + 0.0
        pic[:, -pixels:] = 0.
    elif mode == 'sr':
        pic[:, pixels:] = pic[:, :-pixels] + 0.0
        pic[:, :pixels] = 0.
    elif mode == 'sd':
        pic[pixels:, :] = pic[:-pixels, :] + 0.0
        pic[:pixels, :] = 0.
    elif mode == 'su':
        pic[:-pixels, :] = pic[pixels:, :] + 0.0
        pic[-pixels:, :] = 0.
    return pic


class Data(object):
    """
    If the dataset can be quickly loaded to memory self.X will contain np.ndarray
    Otherwise we will be reading files as we train. In this case self.X is a structure:
        self.X.paths        list of paths to the files containing pictures
        self.X.dict_loaded  dictionary of (key, val), where key is the index of the
                            already loaded datapoint and val is the corresponding index
                            in self.X.loaded
        self.X.loaded       list containing already loaded pictures
    """
    def __init__(self, opts, X, type='data', paths=None, dict_loaded=None, loaded=None):
        """
        X is either np.ndarray or paths
        """
        self.data_dir = _data_dir(opts)
        self.dataset_name = opts['dataset']
        self.normalize = opts['input_normalize_sym']
        self.opts = opts
        self.X = None
        self.type = type
        self.paths = None
        self.dict_loaded = None
        self.loaded = None
        # Load scream for screamdSprites
        if opts['dataset']=='screamdsprites':
            with utils.o_gfile(SCREAM_PATH, 'rb') as f:
                scream = Image.open(f)
                scream.thumbnail((350, 274, 3))
                self.scream = np.array(scream) / 255.
                scream.close()

        if isinstance(X, np.ndarray):
            self.X = X
            self.shape = X.shape
        else:
            assert isinstance(self.data_dir, str), 'Data directory not provided'
            assert paths is not None and len(paths) > 0, 'No paths provided for the data'
            self.paths = paths[:]
            self.dict_loaded = {} if dict_loaded is None else dict_loaded
            self.loaded = [] if loaded is None else loaded
            if self.dataset_name == 'celebA':
                self.crop_style = opts['celebA_crop']
            self.shape = (len(self.paths), None, None, None)

    def __len__(self):
        if isinstance(self.X, np.ndarray):
            return len(self.X)
        else:
            # Our dataset was too large to fit in the memory
            return len(self.paths)

    def drop_loaded(self):
        if not isinstance(self.X, np.ndarray):
            self.dict_loaded = {}
            self.loaded = []

    def __getitem__(self, key):
        if isinstance(self.X, np.ndarray):
            obs = self.X[key]
            # add noise to dsprites
            if self.dataset_name == 'noisydsprites' and self.type=='data':
                obs = np.repeat(obs, 3, axis=-1)
                color = np.random.uniform(0, 1, obs.shape[:-1] + (3,))
                obs = np.minimum(obs + color, 1.)
            elif self.dataset_name == 'screamdsprites'and self.type=='data':
                obs = np.repeat(obs, 3, axis=-1)
                if len(obs.shape)==3:
                    npatch = 1
                    patches = image.extract_patches_2d(self.scream, (64, 64), npatch)[0]
                else:
                    npatch = obs.shape[0]
                    patches = image.extract_patches_2d(self.scream, (64, 64), npatch)
                background = (patches + np.random.uniform(0, 1, size=3)) / 2
                mask = (obs == 1.)
                background[mask] = 1 - background[mask]
                obs = background
            return obs
        else:
            # Our dataset was too large to fit in the memory
            if isinstance(key, int):
                keys = [key]
            elif isinstance(key, list):
                keys = key
            elif isinstance(key, np.ndarray):
                keys = list(key)
            elif isinstance(key, slice):
                start = key.start
                stop = key.stop
                step = key.step
                start = start if start is not None else 0
                if start < 0:
                    start += len(self.paths)
                stop = stop if stop is not None else len(self.paths) - 1
                if stop < 0:
                    stop += len(self.paths)
                step = step if step is not None else 1
                keys = range(start, stop, step)
            else:
                print(type(key))
                raise Exception('This type of indexing yet not supported for the dataset')
            res = []
            new_keys = []
            new_points = []
            for key in keys:
                if key in self.dict_loaded:
                    idx = self.dict_loaded[key]
                    res.append(self.loaded[idx])
                else:
                    if self.dataset_name == 'celebA':
                        point = self._read_celeba_image(self.data_dir, self.paths[key])
                    elif self.dataset_name[-8:] == 'dsprites':
                        data_dir = os.path.join(self.data_dir, 'images')
                        point = self._read_dsprites_image(data_dir, self.paths[key])
                        if self.dataset_name == 'noisydsprites' and self.type=='data':
                            point = np.repeat(point, 3, axis=-1)
                            color = np.random.uniform(0, 1, point.shape[:-1] + (3,))
                            point = np.minimum(point + color, 1.)
                        elif self.dataset_name == 'screamdsprites'and self.type=='data':
                            point = np.repeat(point, 3, axis=-1)
                            patches = image.extract_patches_2d(self.scream, (64, 64), 1)[0]
                            background = (patches + np.random.uniform(0, 1, size=3)) / 2
                            mask = (point > .8)
                            background[mask] = 1 - background[mask]
                            point = background
                    else:
                        raise Exception('Disc read for this dataset not implemented yet...')
                    if self.normalize:
                        point = (point - 0.5) * 2.
                    res.append(point)
                    new_points.append(point)
                    new_keys.append(key)
            n = len(self.loaded)
            cnt = 0
            for key in new_keys:
                self.dict_loaded[key] = n + cnt
                cnt += 1
            self.loaded.extend(new_points)
            if len(self.loaded)>30000:
                # droping loaded images every 50000
                self.drop_loaded()

            return np.array(res)

    def _read_celeba_image(self, data_dir, filename):
        width = 178
        height = 218
        new_width = 140
        new_height = 140
        im = Image.open(utils.o_gfile((data_dir, filename), 'rb'))
        if self.crop_style == 'closecrop':
            # This method was used in DCGAN, pytorch-gan-collection, AVB, ...
            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height)/2
            im = im.crop((left, top, right, bottom))
            im = im.resize((64, 64), Image.ANTIALIAS)
        elif self.crop_style == 'resizecrop':
            # This method was used in ALI, AGE, ...
            im = im.resize((64, 78), Image.ANTIALIAS)
            im = im.crop((0, 7, 64, 64 + 7))
        else:
            raise Exception('Unknown crop style specified')
        im_array = np.array(im).reshape(datashapes['celebA']) / 255.
        im.close()
        return im_array

    def _read_dsprites_image(self, data_dir, filename):
        im = Image.open(utils.o_gfile((data_dir, filename), 'rb'))
        im_array =np.array(im).reshape(datashapes['dsprites']).astype(np.float32) / 255.
        im.close()
        return im_array


class DataHandler(object):
    """A class storing and manipulating the dataset.

    In this code we asume a data point is a 3-dimensional array, for
    instance a 28*28 grayscale picture would correspond to (28,28,1),
    a 16*16 picture of 3 channels corresponds to (16,16,3) and a 2d point
    corresponds to (2,1,1). The shape is contained in self.data_shape
    """


    def __init__(self, opts):
        # load data
        self._load_data(opts)
        # normalize if needed, else we will normalyze while reading from disk
        if opts['input_normalize_sym']:
            # Normalize data to [-1, 1]
            if isinstance(self.data.X, np.ndarray):
                self.data.X = (self.data.X - 0.5) * 2.
        # creating random masks
        self._data_randomization(opts)
        # data informations
        self.data_shape = datashapes[opts['dataset']]
        self.train_size = len(self.rand_masks['train'])
        self.test_size = len(self.rand_masks['test'])

    def _load_data(self, opts):
        """Load a dataset and fill all the necessary variables.

        """
        if opts['dataset'][-8:] == 'dsprites':
            self._load_dsprites(opts)
        elif opts['dataset'] == '3dshapes':
            self._load_3dshapes(opts)
        elif opts['dataset'] == 'smallNORB':
            self._load_smallNORB(opts)
        elif opts['dataset'] == '3Dchairs':
            self._load_3Dchairs(opts)
        elif opts['dataset'] == 'celebA':
            self._load_celebA(opts)
        elif opts['dataset'] == 'mnist':
            self._load_mnist(opts)
        elif opts['dataset'] == 'svhn':
            self._load_svhn(opts)
        else:
            raise ValueError('Unknown %s' % opts['dataset'])

    def _data_randomization(self, opts):
        """Create random masks to shuffle the data for ewach experience

        """
        shuffling_mask = np.arange(len(self.data))
        seed = 123
        np.random.seed(seed)
        np.random.shuffle(shuffling_mask)
        np.random.seed()
        np.random.shuffle(shuffling_mask[:-opts['plot_num_pics']])
        # self.data_order_idx = np.argsort(shuffling_mask)
        self.rand_masks = {}
        self.rand_masks['train'] = shuffling_mask[:-10000]
        self.rand_masks['test'] = shuffling_mask[-10000:-opts['plot_num_pics']]
        self.rand_masks['vizu'] = shuffling_mask[-opts['plot_num_pics']:]

    def _load_dsprites(self, opts):
        """Load data from dsprites dataset

        """

        logging.error('Loading dsprites...')
        # Loading labels and data
        data_dir = _data_dir(opts)
        data_path = os.path.join(data_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        Y = np.load(data_path, allow_pickle=True)['latents_classes'][:,1:]
        self.labels = Data(opts, Y, type='label')
        num_samples = Y.shape[0]
        paths = np.array(['%.6d.jpg' % i for i in range(1, num_samples + 1)])
        self.data = Data(opts, None, 'data', paths)
        # plot set
        self.plot_data_idx = np.arange(41488,len(Y),34816)
        # labels informations
        self.factor_indices = list(range(5))
        self.factor_sizes = np.array(np.load(data_path, allow_pickle=True, encoding="latin1")['metadata'][()]["latents_sizes"],dtype=np.int64)[1:]
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
            self.factor_sizes)

        logging.error('Loading Done.')

    def _load_3dshapes(self, opts):
        """Load data from 3Dshapes dataset

        """

        def get_factors_from_labels(labels):
            """Convert labels values to factors categories
            """
            num_factors = labels.shape[-1]
            factors = []
            for i in range(num_factors):
                _, factor = np.unique(labels[:,i],return_inverse=True)
                factors.append(factor)
            return np.stack(factors,axis=-1)

        logging.error('Loading 3Dshapes...')
        # Loading data
        data_dir = _data_dir(opts)
        data_path = os.path.join(data_dir, '3dshapes.h5')
        dataset = h5py.File(data_path, 'r')
        X = np.array(dataset['images']).astype(np.float32) / 255.
        Y = np.array(dataset['labels'])
        self.data = Data(opts, X)
        self.labels = Data(opts, Y, type='label')
        # plot set
        self.plot_data_idx = np.arange(10)
        # labels informations
        self.factor_indices = list(range(6))
        self.factor_sizes = np.array([10,10,10,8,4,15])
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
            self.factor_sizes)

        logging.error('Loading Done.')

    def _load_smallNORB(self, opts):
        """Load data from smallNORB dataset

        """

        logging.error('Loading smallNORB...')
        # Loading data
        data_dir = _data_dir(opts)
        SMALLNORB_CHUNKS = ['smallnorb-5x46789x9x18x6x2x96x96-training-{0}.mat.gz',
                            'smallnorb-5x01235x9x18x6x2x96x96-testing-{0}.mat.gz']
        list_of_images = []
        list_of_labels = []
        list_of_infos = []
        for chunk_name in SMALLNORB_CHUNKS:
            # Loading data
            file_path = os.path.join(data_dir, chunk_name.format('dat'))
            with gzip.open(file_path, mode='rb') as f:
                header = _parse_smallNORB_header(f)
                num_examples, channels, height, width = header['dimensions']
                images = np.zeros(shape=(num_examples, 2, height, width), dtype=np.uint8)
                for i in range(num_examples):
                    # Read raw image data and restore shape as appropriate
                    image = struct.unpack('<' + height * width * 'B', f.read(height * width))
                    image = np.uint8(np.reshape(image, newshape=(height, width)))
                    images[i] = image
            list_of_images.append(_resize_images(images[:, 0]))
            # Loading category
            file_path = os.path.join(data_dir, chunk_name.format('cat'))
            with gzip.open(file_path, mode='rb') as f:
                header = _parse_smallNORB_header(f)
                num_examples, = header['dimensions']
                struct.unpack('<BBBB', f.read(4))  # ignore this integer
                struct.unpack('<BBBB', f.read(4))  # ignore this integer
                categories = np.zeros(shape=num_examples, dtype=np.int32)
                for i in tqdm(range(num_examples), disable=True, desc='Loading categories...'):
                    category, = struct.unpack('<i', f.read(4))
                    categories[i] = category
            # Loading infos
            file_path = os.path.join(data_dir, chunk_name.format('info'))
            with gzip.open(file_path, mode='rb') as f:
                header = _parse_smallNORB_header(f)
                struct.unpack('<BBBB', f.read(4))  # ignore this integer
                num_examples, num_info = header['dimensions']
                infos = np.zeros(shape=(num_examples, num_info), dtype=np.int32)
                for r in tqdm(range(num_examples), disable=True, desc='Loading info...'):
                    for c in range(num_info):
                        info, = struct.unpack('<i', f.read(4))
                        infos[r, c] = info
            list_of_labels.append((np.column_stack((categories, infos))))
        X = np.concatenate(list_of_images, axis=0)
        Y = np.concatenate(list_of_labels, axis=0)
        X = np.expand_dims(X,axis=-1)
        Y[:, 3] = Y[:, 3] / 2  # azimuth values are 0, 2, 4, ..., 24
        self.data = Data(opts, X)
        self.labels = Data(opts, Y, type='label')
        # plot set
        self.plot_data_idx = np.arange(0+18*6,40+18*6,2)
        # labels informations
        self.factor_indices = [0, 2, 3, 4]
        self.factor_sizes = np.array([5, 10, 9, 18, 6])
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
            self.factor_sizes)

        logging.error('Loading Done.')

    def _load_3Dchairs(self, opts):
        """Load data from 3Dchairs dataset

        """

        logging.error('Loading 3Dchairs')
        filename = os.path.join(_data_dir(opts), 'rendered_chairs.npz')
        # Extracting data and saving as npz if necessary
        if not tf.io.gfile.exists(filename):
            tar = tarfile.open(filename[:-4] +'.tar')
            tar.extractall(path=_data_dir(opts))
            tar.close()
            X = []
            n = 0
            root_dir = os.path.join(_data_dir(opts), 'rendered_chairs')
            # Iterate over all the dir
            for dir in os.listdir(root_dir):
                # Create full path
                if dir!='all_chair_names.mat':
                    subdir = os.path.join(root_dir, dir, 'renders')
                    for file in os.listdir(subdir):
                        path_to_file = os.path.join(subdir,file)
                        im = Image.open(path_to_file)
                        im = im.resize((64, 64), Image.ANTIALIAS)
                        X.append(np.array(im.getdata()))
                        im.close()
                        n += 1
                        if n%10000==0:
                            print('{} images unizped'.format(n))
            np.savez_compressed(filename,data=np.array(X).reshape([-1,]+datashapes['3Dchairs']) / 255.)
            shutil.rmtree(root_dir)
        # loading data
        X = np.load(filename,allow_pickle=True)['data']
        self.data = Data(opts, X)
        # plot set
        self.plot_data_idx = np.arange(10)

        logging.error('Loading Done.')

    def _load_celebA(self, opts):
        """Load CelebA
        """
        logging.error('Loading CelebA dataset')

        num_samples = 202599
        paths = np.array(['%.6d.jpg' % i for i in range(1, num_samples + 1)])
        self.data = Data(opts, None, 'data', paths)
        # plot set
        # idx = [5,6,14,17,39,50,60,70,80,90]
        self.plot_data = np.arange(5,5+50)

        logging.error('Loading Done.')

    def _load_mnist(self, opts, zalando=False, modified=False):
        """Load data from MNIST or ZALANDO files.

        """
        if zalando:
            logging.error('Loading Fashion MNIST')
        elif modified:
            logging.error('Loading modified MNIST')
        else:
            logging.error('Loading MNIST')
        data_dir = _data_dir(opts)
        # pylint: disable=invalid-name
        # Let us use all the bad variable names!
        tr_X = None
        tr_Y = None
        te_X = None
        te_Y = None

        with gzip.open(os.path.join(data_dir, 'train-images-idx3-ubyte.gz')) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(60000*28*28*1), dtype=np.uint8)
            tr_X = loaded.reshape((60000, 28, 28, 1)).astype(np.float32)

        with gzip.open(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')) as fd:
            fd.read(8)
            loaded = np.frombuffer(fd.read(60000), dtype=np.uint8)
            tr_Y = loaded.reshape((60000)).astype(np.int)

        with gzip.open(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(10000*28*28*1), dtype=np.uint8)
            te_X = loaded.reshape((10000, 28, 28, 1)).astype(np.float32)

        with gzip.open(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')) as fd:
            fd.read(8)
            loaded = np.frombuffer(fd.read(10000), dtype=np.uint8)
            te_Y = loaded.reshape((10000)).astype(np.int)

        tr_Y = np.asarray(tr_Y)
        te_Y = np.asarray(te_Y)

        X = np.concatenate((tr_X, te_X), axis=0)
        y = np.concatenate((tr_Y, te_Y), axis=0)
        X = X / 255.

        seed = 123
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        np.random.seed()

        self.data_shape = (28, 28, 1)
        test_size = 10000

        if modified:
            self.original_mnist = X
            n = opts['toy_dataset_size']
            n += test_size
            points = []
            labels = []
            for _ in range(n):
                idx = np.random.randint(len(X))
                point = X[idx]
                modes = ['n', 'i', 'sl', 'sr', 'su', 'sd']
                mode = modes[np.random.randint(len(modes))]
                point = transform_mnist(point, mode)
                points.append(point)
                labels.append(y[idx])
            X = np.array(points)
            y = np.array(y)
        if opts['train_dataset_size']==-1:
            self.data = Data(opts, X[:-test_size])
        else:
            self.data = Data(opts, X[:opts['train_dataset_size']])
        self.test_data = Data(opts, X[-test_size:])
        self.labels = y[:-test_size]
        self.test_labels = y[-test_size:]
        self.num_points = len(self.data)

        logging.error('Loading Done: Train size: %d, Test size: %d' % (self.num_points,len(self.test_data)))

    def _load_svhn(self, opts):
        """Load data from SVHN files.

        """
        NUM_LABELS = 10

        # Helpers to process raw data
        def convert_imgs_to_array(img_array):
            rows = datashapes['svhn'][0]
            cols = datashapes['svhn'][1]
            chans = datashapes['svhn'][2]
            num_imgs = img_array.shape[3]
            # Note: not the most efficent way but can monitor what is happening
            new_array = np.empty(shape=(num_imgs, rows, cols, chans), dtype=np.float32)
            for x in range(0, num_imgs):
                # TODO reuse normalize_img here
                chans = img_array[:, :, :, x]
                # # normalize pixels to 0 and 1. 0 is pure white, 1 is pure channel color
                # norm_vec = (255-chans)*1.0/255.0
                new_array[x] = chans
            return new_array

        # Extracting data
        data_dir = _data_dir(opts)

        # Training data
        file_path = os.path.join(data_dir,'train_32x32.mat')
        file = open(file_path, 'rb')
        data = loadmat(file)
        imgs = data['X']
        labels = data['y'].flatten()
        labels[labels == 10] = 0  # Fix for weird labeling in dataset
        tr_Y = labels
        tr_X = convert_imgs_to_array(imgs)
        tr_X = tr_X / 255.
        file.close()
        if opts['use_extra']:
            file_path = os.path.join(data_dir,'extra_32x32.mat')
            file = open(file_path, 'rb')
            data = loadmat(file)
            imgs = data['X']
            labels = data['y'].flatten()
            labels[labels == 10] = 0  # Fix for weird labeling in dataset
            extra_Y = labels
            extra_X = convert_imgs_to_array(imgs)
            extra_X = extra_X / 255.
            file.close()
            # concatenate training and extra
            tr_X = np.concatenate((tr_X,extra_X), axis=0)
            tr_Y = np.concatenate((tr_Y,extra_Y), axis=0)
        seed = 123
        np.random.seed(seed)
        np.random.shuffle(tr_X)
        np.random.seed(seed)
        np.random.shuffle(tr_Y)
        np.random.seed()

        # Testing data
        file_path = os.path.join(data_dir,'test_32x32.mat')
        file = open(file_path, 'rb')
        data = loadmat(file)
        imgs = data['X']
        labels = data['y'].flatten()
        labels[labels == 10] = 0  # Fix for weird labeling in dataset
        te_Y = labels
        te_X = convert_imgs_to_array(imgs)
        te_X = te_X / 255.
        file.close()

        self.data_shape = (32,32,3)

        self.data = Data(opts, tr_X)
        self.labels = tr_Y
        self.test_data = Data(opts, te_X)
        self.test_labels = te_Y
        self.num_points = len(self.data)

        logging.error('Loading Done: Train size: %d, Test size: %d' % (self.num_points,len(self.test_data)))

    def get_batch_img(self,idx, type):
        return self.data[self.rand_masks[type][idx]]

    def get_batch_label(self,idx, type):
        return self.labels[self.rand_masks[type][idx]]

    def sample_observations_from_factors(self, dataset, factors):
        if dataset[-8:] == 'dsprites':
            indices = np.dot(factors, self.factor_bases).astype(dtype=np.int32)
            images = self.data[indices]
        elif dataset == '3dshapes':
            indices = np.dot(factors, self.factor_bases).astype(dtype=np.int32)
            images = self.data[indices]
        elif dataset == 'smallNORB':
            feature_state_space_index = np.array(np.dot(self.labels.X, self.factor_bases), dtype=np.int32)
            num_total_atoms = np.prod(self.factor_sizes)
            state_space_to_save_space_index = np.zeros(num_total_atoms, dtype=np.int32)
            state_space_to_save_space_index[feature_state_space_index] = np.arange(num_total_atoms)
            state_space_index = np.dot(factors, self.factor_bases).astype(dtype=np.int32)
            indices = state_space_to_save_space_index[state_space_index]
            images = self.data[indices]
        elif dataset == '3Dchairs':
            assert False, 'No factors for {}'.format(dataset)
        elif dataset == 'celebA':
            assert False, 'No factors for {}'.format(dataset)
        elif dataset == 'mnist':
            assert False, 'No factors for {}'.format(dataset)
        elif dataset == 'svhn':
            assert False, 'No factors for {}'.format(dataset)
        else:
            raise ValueError('Unknown {}'.format(opts['dataset']))

        return images

def matrix_type_from_magic(magic_number):
    """
    Get matrix data type from magic number
    See here: https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/readme for details.
    Parameters
    ----------
    magic_number: tuple
        First 4 bytes read from small NORB files
    Returns
    -------
    element type of the matrix
    """
    convention = {'1E3D4C51': 'single precision matrix',
                  '1E3D4C52': 'packed matrix',
                  '1E3D4C53': 'double precision matrix',
                  '1E3D4C54': 'integer matrix',
                  '1E3D4C55': 'byte matrix',
                  '1E3D4C56': 'short matrix'}
    magic_str = bytearray(reversed(magic_number)).hex().upper()
    return convention[magic_str]

def _parse_smallNORB_header(file_pointer):
    """
    Parse header of small NORB binary file

    Parameters
    ----------
    file_pointer: BufferedReader
        File pointer just opened in a small NORB binary file
    Returns
    -------
    file_header_data: dict
        Dictionary containing header information
    """
    # Read magic number
    magic = struct.unpack('<BBBB', file_pointer.read(4))  # '<' is little endian)

    # Read dimensions
    dimensions = []
    num_dims, = struct.unpack('<i', file_pointer.read(4))  # '<' is little endian)
    for _ in range(num_dims):
        dimensions.extend(struct.unpack('<i', file_pointer.read(4)))

    file_header_data = {'magic_number': magic,
                        'matrix_type': matrix_type_from_magic(magic),
                        'dimensions': dimensions}
    return file_header_data

def _read_binary_matrix(filename):
    """Reads and returns binary formatted matrix stored in filename."""
    with tf.gfile.GFile(filename, "rb") as f:
        s = f.read()
        magic = int(np.frombuffer(s, "int32", 1))
        ndim = int(np.frombuffer(s, "int32", 1, 4))
        eff_dim = max(3, ndim)
        raw_dims = np.frombuffer(s, "int32", eff_dim, 8)
        dims = []
        for i in range(0, ndim):
            dims.append(raw_dims[i])

        dtype_map = {507333717: "int8",
                    507333716: "int32",
                    507333713: "float",
                    507333715: "double"}
        data = np.frombuffer(s, dtype_map[magic], offset=8 + eff_dim * 4)
    data = data.reshape(tuple(dims))
    return data

def _resize_images(integer_images):
    resized_images = np.zeros((integer_images.shape[0], 64, 64))
    for i in range(integer_images.shape[0]):
        image = Image.fromarray(integer_images[i, :, :])
        image = image.resize((64, 64), Image.ANTIALIAS)
        resized_images[i, :, :] = image
    return resized_images.astype(np.float32) / 255.

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
from math import ceil

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

# Loading scream
SCREAM_PATH = '../data/dsprites/scream.jpg'
with utils.o_gfile(SCREAM_PATH, 'rb') as f:
    img = Image.open(f)
    img.thumbnail((350, 274, 3))
    scream = np.array(img) / 255.
    img.close()


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

def add_color_noise(x):
    # sample noise
    noise = tf.random.uniform(datashapes['noisydsprites'],0,1, dtype=tf.dtypes.float32)

    return tf.math.minimum(x+noise,tf.constant([1.], dtype=tf.dtypes.float32))

def add_scream_noise(x):
    scream_tf = tf.constant(scream, dtype=tf.dtypes.float32)
    # creating noisy background
    backgrounds = tf.image.random_crop(scream_tf, [64, 64, 3])
    backgrounds = (backgrounds + tf.random.uniform([3,], 0, 1)) / 2.
    return tf.where(tf.repeat(x, repeats=[3,], axis=-1)>.85, 1. - backgrounds, backgrounds)

class DataHandler(object):
    """A class storing and manipulating the dataset.

    In this code we asume a data point is a 3-dimensional array, for
    instance a 28*28 grayscale picture would correspond to (28,28,1),
    a 16*16 picture of 3 channels corresponds to (16,16,3) and a 2d point
    corresponds to (2,1,1). The shape is contained in self.data_shape
    """


    def __init__(self, opts):
        self.dataset = opts['dataset']
        self.crop_style = opts['celebA_crop']
        # load data
        logging.error('\n Loading {}.'.format(self.dataset))
        self._init_dataset(opts)
        logging.error('Loading Done.')

    def _init_dataset(self, opts):
        """Load a dataset and fill all the necessary variables.

        """
        if self.dataset[-8:] == 'dsprites':
            self._load_dsprites(opts)
        elif self.dataset == '3dshapes':
            self._load_3dshapes(opts)
        elif self.dataset == 'smallNORB':
            self._load_smallNORB(opts)
        elif self.dataset == '3Dchairs':
            self._load_3Dchairs(opts)
        elif self.dataset == 'celebA':
            self._load_celebA(opts)
        elif self.dataset == 'mnist':
            self._load_mnist(opts)
        elif self.dataset == 'svhn':
            self._load_svhn(opts)
        else:
            raise ValueError('Unknown %s' % self.dataset)

    def _load_dsprites(self, opts):
        """Init data from dsprites dataset

        """
        # Loading labels and data
        """ just create paths_list & load labels info"""
        num_data = 737280
        self.data_dir = _data_dir(opts)
        self.all_data = np.array([os.path.join(self.data_dir,'images','%.6d.jpg') % i for i in range(1, num_data + 1)])
        data_path = os.path.join(self.data_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        with np.load(data_path, encoding="latin1", allow_pickle=True) as data:
            self.factor_sizes = np.array(data['metadata'][()]["latents_sizes"], dtype=np.int64)[1:]
        # labels informations
        self.factor_indices = list(range(5))
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
                            self.factor_sizes)
        # plot set
        idx = np.random.randint(self.all_data.shape[0],size=opts['evaluate_num_pics'])
        self.data_plot = self._sample_observations(idx)
        # shuffling data
        np.random.seed()
        idx_random = np.random.permutation(num_data)
        if opts['train_dataset_size']==-1 or opts['train_dataset_size']>num_data-10000:
            tr_stop = num_data - 10000
        else:
            tr_stop = opts['train_dataset_size']
        data_train = self.all_data[idx_random[:tr_stop]]
        data_test = self.all_data[idx_random[-10000:]]
        # dataset size
        self.train_size = data_train.shape[0]
        self.test_size = data_test.shape[0]
        # data for vizualisation
        seed = 123
        np.random.seed(seed)
        idx = np.random.randint(self.test_size, size=opts['plot_num_pics'])
        self.data_vizu = self._sample_observations(idx)
        # datashape
        self.data_shape = datashapes[self.dataset]
        # Create tf.dataset
        dataset_train = tf.data.Dataset.from_tensor_slices(data_train)
        dataset_test = tf.data.Dataset.from_tensor_slices(data_test)
        # map files paths to image with tf.io.decode_jpeg
        def process_path(file_path):
            image_file = tf.read_file(file_path)
            img_decoded = tf.cast(tf.image.decode_jpeg(image_file, channels=0), dtype=tf.dtypes.float32) / 255.
            # tf.cast(tf.image.decode_jpeg(image_file, channels=0), dtype=tf.dtypes.float32) / 255.
            return tf.reshape(img_decoded, datashapes['dsprites'])
        dataset_train = dataset_train.map(process_path,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_test = dataset_test.map(process_path,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # add noise if needed
        if self.dataset=='noisydsprites':
            dataset_train = dataset_train.map(add_color_noise,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset_test = dataset_test.map(add_color_noise,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.dataset=='screamdsprites':
            dataset_train = dataset_train.map(add_scream_noise,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset_test = dataset_test.map(add_scream_noise,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # normalize data if needed
        if opts['input_normalize_sym']:
            dataset_train = dataset_train.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset_test = dataset_test.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.data_vizu = (self.data_vizu - 0.5) * 2.
            self.data_plot = (self.data_plot - 0.5) * 2.
        # Shuffle dataset
        dataset_train = dataset_train.shuffle(buffer_size=50*opts['batch_size'])
        dataset_test = dataset_test.shuffle(buffer_size=50*opts['batch_size'])
        # repeat for multiple epochs
        dataset_train = dataset_train.repeat()
        dataset_test = dataset_test.repeat()
        # Random batching
        dataset_train = dataset_train.batch(batch_size=opts['batch_size'])
        dataset_test = dataset_test.batch(batch_size=opts['batch_size'])
        # Prefetch
        self.dataset_train = dataset_train.prefetch(buffer_size=4*opts['batch_size'])
        self.dataset_test = dataset_test.prefetch(buffer_size=4*opts['batch_size'])
        # Iterator for each split
        self.iterator_train = dataset_train.make_initializable_iterator()
        self.iterator_test = dataset_test.make_initializable_iterator()

        # Global iterator
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_element = tf.data.Iterator.from_string_handle(
            self.handle, dataset_train.output_types, dataset_train.output_shapes).get_next()

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

        # Loading data
        data_dir = _data_dir(opts)
        data_path = os.path.join(self.data_dir, '3dshapes.h5')
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

    def _load_smallNORB(self, opts):
        """Load data from smallNORB dataset

        """

        # Loading data
        self.data_dir = _data_dir(opts)
        SMALLNORB_CHUNKS = ['smallnorb-5x46789x9x18x6x2x96x96-training-{0}.mat.gz',
                            'smallnorb-5x01235x9x18x6x2x96x96-testing-{0}.mat.gz']
        list_of_images = []
        list_of_labels = []
        list_of_infos = []
        for chunk_name in SMALLNORB_CHUNKS:
            # Loading data
            file_path = os.path.join(self.data_dir, chunk_name.format('dat'))
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
            file_path = os.path.join(self.data_dir, chunk_name.format('cat'))
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
            file_path = os.path.join(self.data_dir, chunk_name.format('info'))
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
        num_data = Y.shape[0]
        # data
        self.all_data = X
        # labels
        self.all_labels = Y
        self.factor_indices = [0, 2, 3, 4]
        self.factor_sizes = np.array([5, 10, 9, 18, 6])
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
            self.factor_sizes)
        # plot set
        idx = np.random.randint(self.all_data.shape[0],size=opts['evaluate_num_pics'])
        self.data_plot = self._sample_observations(idx)
        # shuffling data
        np.random.seed()
        idx_random = np.random.permutation(num_data)
        if opts['train_dataset_size']==-1 or opts['train_dataset_size']>num_data-10000:
            tr_stop = num_data - 10000
        else:
            tr_stop = opts['train_dataset_size']
        data_train = self.all_data[idx_random[:tr_stop]]
        data_test = self.all_data[idx_random[-10000:]]
        # dataset size
        self.train_size = data_train.shape[0]
        self.test_size = data_test.shape[0]
        # data for vizualisation
        seed = 123
        np.random.seed(seed)
        idx = np.random.randint(self.test_size, size=opts['plot_num_pics'])
        self.data_vizu = self._sample_observations(idx)
        # datashape
        self.data_shape = datashapes[self.dataset]
        # Create tf.dataset
        dataset_train = tf.data.Dataset.from_tensor_slices(data_train)
        dataset_test = tf.data.Dataset.from_tensor_slices(data_test)
        # # map files paths to image with tf.io.decode_jpeg
        # def process_path(file_path):
        #     image_file = tf.read_file(file_path)
        #     return tf.cast(tf.image.decode_jpeg(image_file, channels=0), dtype=tf.dtypes.float32) / 255.
        # dataset_train = dataset_train.map(process_path,
        #                         num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # dataset_test = dataset_test.map(process_path,
        #                         num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # normalize data if needed
        if opts['input_normalize_sym']:
            dataset_train = dataset_train.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset_test = dataset_test.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.data_vizu = (self.data_vizu - 0.5) * 2.
            self.data_plot = (self.data_plot - 0.5) * 2.
        # Shuffle dataset
        dataset_train = dataset_train.shuffle(buffer_size=50*opts['batch_size'])
        dataset_test = dataset_test.shuffle(buffer_size=50*opts['batch_size'])
        # repeat for multiple epochs
        dataset_train = dataset_train.repeat()
        dataset_test = dataset_test.repeat()
        # Random batching
        dataset_train = dataset_train.batch(batch_size=opts['batch_size'])
        dataset_test = dataset_test.batch(batch_size=opts['batch_size'])
        # Prefetch
        self.dataset_train = dataset_train.prefetch(buffer_size=4*opts['batch_size'])
        self.dataset_test = dataset_test.prefetch(buffer_size=4*opts['batch_size'])
        # Iterator for each split
        self.iterator_train = dataset_train.make_initializable_iterator()
        self.iterator_test = dataset_test.make_initializable_iterator()

        # Global iterator
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_element = tf.data.Iterator.from_string_handle(
            self.handle, dataset_train.output_types, dataset_train.output_shapes).get_next()

    def _load_3Dchairs(self, opts):
        """Load data from 3Dchairs dataset

        """
        self.data_dir = _data_dir(opts)
        filename = os.path.join(self.data_dir, 'rendered_chairs.npz')
        # Extracting data and saving as npz if necessary
        if not tf.io.gfile.exists(filename):
            tar = tarfile.open(filename[:-4] +'.tar')
            tar.extractall(self.data_dir)
            tar.close()
            X = []
            n = 0
            root_dir = os.path.join(self.data_dir, 'rendered_chairs')
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

    def _load_celebA(self, opts):
        """Load CelebA
        """
        num_data = 202599
        self.data_dir = _data_dir(opts)
        self.all_data = np.array([os.path.join(self.data_dir,'%.6d.jpg') % i for i in range(1, num_data + 1)])
        # plot set
        idx = np.random.randint(self.all_data.shape[0],size=opts['evaluate_num_pics'])
        self.data_plot = self._sample_observations(idx)
        # shuffling data
        np.random.seed()
        idx_random = np.random.permutation(num_data)
        if opts['train_dataset_size']==-1 or opts['train_dataset_size']>num_data-10000:
            tr_stop = num_data - 10000
        else:
            tr_stop = opts['train_dataset_size']
        data_train = self.all_data[idx_random[:tr_stop]]
        data_test = self.all_data[idx_random[-10000:]]
        # dataset size
        self.train_size = data_train.shape[0]
        self.test_size = data_test.shape[0]
        # data for vizualisation
        seed = 123
        np.random.seed(seed)
        idx = np.random.randint(self.test_size, size=opts['plot_num_pics'])
        self.data_vizu = self._sample_observations(idx)
        # datashape
        self.data_shape = datashapes[self.dataset]
        # Create tf.dataset
        dataset_train = tf.data.Dataset.from_tensor_slices(data_train)
        dataset_test = tf.data.Dataset.from_tensor_slices(data_test)
        # map files paths to image with tf.io.decode_jpeg
        def process_path(file_path):
            # reading .jpg file
            image_file = tf.read_file(file_path)
            im_decoded = tf.cast(tf.image.decode_jpeg(image_file, channels=3), dtype=tf.dtypes.float32)
            # crop and resize
            width = 178
            height = 218
            new_width = 140
            new_height = 140
            if self.crop_style == 'closecrop':
                # This method was used in DCGAN, pytorch-gan-collection, AVB, ...
                left = (width - new_width) / 2
                top = (height - new_height) / 2
                right = (width + new_width) / 2
                bottom = (height + new_height) / 2
                im = tf.image.crop_and_resize(tf.expand_dims(im_decoded,axis=0),
                                        np.array([[top / height, right / width, bottom / height, left / width]]),
                                        [0,],
                                        (64,64),
                                        method='bilinear', extrapolation_value=0)
            # elif self.crop_style == 'resizecrop':
            #     # This method was used in ALI, AGE, ...
            #     im = im.resize((64, 78), Image.ANTIALIAS)
            #     im = im.crop((0, 7, 64, 64 + 7))
            else:
                assert False, '{} not implemented.'.format(self.crop_style)
            return tf.reshape(im, datashapes['celebA']) / 255.
        dataset_train = dataset_train.map(process_path,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_test = dataset_test.map(process_path,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # normalize data if needed
        if opts['input_normalize_sym']:
            dataset_train = dataset_train.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset_test = dataset_test.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.data_vizu = (self.data_vizu - 0.5) * 2.
            self.data_plot = (self.data_plot - 0.5) * 2.
        # Shuffle dataset
        dataset_train = dataset_train.shuffle(buffer_size=50*opts['batch_size'])
        dataset_test = dataset_test.shuffle(buffer_size=50*opts['batch_size'])
        # repeat for multiple epochs
        dataset_train = dataset_train.repeat()
        dataset_test = dataset_test.repeat()
        # Random batching
        dataset_train = dataset_train.batch(batch_size=opts['batch_size'])
        dataset_test = dataset_test.batch(batch_size=opts['batch_size'])
        # Prefetch
        self.dataset_train = dataset_train.prefetch(buffer_size=4*opts['batch_size'])
        self.dataset_test = dataset_test.prefetch(buffer_size=4*opts['batch_size'])
        # Iterator for each split
        self.iterator_train = dataset_train.make_initializable_iterator()
        self.iterator_test = dataset_test.make_initializable_iterator()

        # Global iterator
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_element = tf.data.Iterator.from_string_handle(
            self.handle, dataset_train.output_types, dataset_train.output_shapes).get_next()

    def _load_mnist(self, opts, zalando=False, modified=False):
        """Load data from MNIST or ZALANDO files.

        """

        self.data_dir = _data_dir(opts)
        # pylint: disable=invalid-name
        # Let us use all the bad variable names!
        tr_X = None
        tr_Y = None
        te_X = None
        te_Y = None

        with gzip.open(os.path.join(self.data_dir, 'train-images-idx3-ubyte.gz')) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(60000*28*28*1), dtype=np.uint8)
            tr_X = loaded.reshape((60000, 28, 28, 1)).astype(np.float32)

        with gzip.open(os.path.join(self.data_dir, 'train-labels-idx1-ubyte.gz')) as fd:
            fd.read(8)
            loaded = np.frombuffer(fd.read(60000), dtype=np.uint8)
            tr_Y = loaded.reshape((60000)).astype(np.int)

        with gzip.open(os.path.join(self.data_dir, 't10k-images-idx3-ubyte.gz')) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(10000*28*28*1), dtype=np.uint8)
            te_X = loaded.reshape((10000, 28, 28, 1)).astype(np.float32)

        with gzip.open(os.path.join(self.data_dir, 't10k-labels-idx1-ubyte.gz')) as fd:
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
        self.data_dir = _data_dir(opts)
        # Training data
        file_path = os.path.join(self.data_dir,'train_32x32.mat')
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
            file_path = os.path.join(self.data_dir,'extra_32x32.mat')
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
        file_path = os.path.join(self.data_dir,'test_32x32.mat')
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

    def init_iterator(self, sess):
        sess.run([self.iterator_train.initializer,self.iterator_test.initializer])
        # handle = sess.run(iterator.string_handle())
        train_handle, test_handle = sess.run([self.iterator_train.string_handle(),self.iterator_test.string_handle()])

        return train_handle, test_handle

    def sample_observations_from_factors(self, opts, factors):
        if self.dataset[-8:] == 'dsprites':
            indices = np.dot(factors, self.factor_bases).astype(dtype=np.int32)
            images = self._sample_observations(indices)
        elif self.dataset == '3dshapes':
            indices = np.dot(factors, self.factor_bases).astype(dtype=np.int32)
            images = self._sample_observations(indices)
        elif self.dataset == 'smallNORB':
            feature_state_space_index = np.array(np.dot(self.all_labels, self.factor_bases), dtype=np.int32)
            num_total_atoms = np.prod(self.factor_sizes)
            state_space_to_save_space_index = np.zeros(num_total_atoms, dtype=np.int32)
            state_space_to_save_space_index[feature_state_space_index] = np.arange(num_total_atoms)
            state_space_index = np.dot(factors, self.factor_bases).astype(dtype=np.int32)
            indices = state_space_to_save_space_index[state_space_index]
            images = self._sample_observations(indices)
        elif self.dataset == '3Dchairs':
            assert False, 'No factors for {}'.format(self.dataset)
        elif dataset == 'celebA':
            assert False, 'No factors for {}'.format(self.dataset)
        elif self.dataset == 'mnist':
            assert False, 'No factors for {}'.format(self.dataset)
        elif self.dataset == 'svhn':
            assert False, 'No factors for {}'.format(self.dataset)
        else:
            raise ValueError('Unknown {}'.format(self.dataset))

        return images

    def _sample_observations(self, keys):
        if len(self.all_data.shape)>1:
            # all_data is an np.ndarray already loaded into the memory
            return self.all_data[keys]
        else:
            # all_data is a 1d array of paths
            obs = []
            keys = list(keys)
            for key in keys:
                img = self._read_image(key)
                obs.append(img)
            return np.stack(obs)

    def _read_image(self, key):
        seed = 123
        assert key==int(os.path.split(self.all_data[key])[1][:-4])-1, 'Mismatch between key and img_file_name'
        if self.dataset == 'celebA':
            point = self._read_celeba_image(self.all_data[key])
        elif self.dataset[-8:] == 'dsprites':
            point = self._read_dsprites_image(self.all_data[key])
            if self.dataset == 'noisydsprites':
                point = np.repeat(point, 3, axis=-1)
                np.random.seed(seed + key)
                color = np.random.uniform(0, 1, point.shape[:-1] + (3,))
                np.random.seed()
                point = np.minimum(point + color, 1.)
            elif self.dataset == 'screamdsprites':
                point = np.repeat(point, 3, axis=-1)
                patches = image.extract_patches_2d(scream, (64, 64), 1)[0]
                np.random.seed(seed + key)
                background = (patches + np.random.uniform(0, 1, size=3)) / 2
                np.random.seed()
                mask = (point > .85)
                background[mask] = 1 - background[mask]
                point = background
        else:
            raise Exception('Disc read for {} not implemented yet...'.format(self.dataset))

        return point

    def _read_celeba_image(self, file_path):
        width = 178
        height = 218
        new_width = 140
        new_height = 140
        im = Image.open(file_path)
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

    def _read_dsprites_image(self, file_path):
        im = Image.open(file_path)
        im_array =np.array(im).reshape(datashapes['dsprites']).astype(np.float32) / 255.
        im.close()
        return im_array


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

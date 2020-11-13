'''  PyTorch implementation of Efficient Deep Learning for Stereo Matching by  Wenjie Luo, Alexander G. Schwing, & Raquel Urtasun
'''

import os
import sys
import glob
import pickle

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import logging
from os.path import join
import argparse
import random
from random import shuffle

import matplotlib

matplotlib.use('agg')

import torchvision
import torch.utils.data as utils
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torchvision.utils
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import pandas as pd
import time
import copy
import datetime
from pytz import timezone
import pytz
import cv2
from torch.utils.tensorboard import SummaryWriter
import torchsummary

##########################################################################
# Load Settings
##########################################################################

if False: # 'google.colab' in str(get_ipython()):
    print("Running in colab mode")


    class Args:
        resume = False  # help='resume from checkpoint')
        data_path = '/content/kitti2015_full_min/'  # join('data', settings.dataset, settings.phase))
        exp_name = '37x37_ref'
        result_dir = 'results'  # result directory
        log_level = 'INFO'  # choices = ['DEBUG', 'INFO'], help='log-level to use')
        batch_size = 32  # type=int, help='batch-size to use')
        dataset = 'kitti_2015'  # , choices=['kitti_2012', 'kitti_2015'], help='dataset')
        seed = 3  # , type=int, help='random seed')
        patch_size = 37  # , type=int, help='patch size from left image')
        disparity_range = 201  # , type=int, help='disparity range')
        learning_rate = 0.01  # , type=float, help='initial learning rate')
        reduction_factor = 25  # , type=int, help='ratio of the end learning rate to the starting learning rate')
        find_patch_locations = False  # , help='find and store patch locations')
        num_iterations = 1  # , type=int, help='number of training iterations')
        phase = 'both'  # , choices=['training', 'testing', 'both'], help='training or testing, if testing perform inference on test set.')
        post_process = False  # , help='toggle use of post-processing.')
        eval = False  # , help='compute error on validation set.')
        test_all = False  # , help='run testing on all image pairs')
        max_batches = 100  # regardless of dataset size limit the number of batches
        shuffle_images = False  # Shuffle the selection of images fed into the patch generator)


    settings = Args()
else:
    print("Running in script mode")
    parser = argparse.ArgumentParser(
        description='Re-implementation of Efficient Deep Learning for Stereo Matching')
    parser.add_argument('--resume', '-r', default=False, help='resume from checkpoint - not supported')
    parser.add_argument('--data-path', default='kitti_2015', type=str, help='root location of kitti_dataset')
    parser.add_argument('--exp-name', default='bs_128_lr_0.2g', type=str, help='name of experiment')
    parser.add_argument('--result-dir', default='results', type=str, help='results directory')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO'], help='log-level to use')
    parser.add_argument('--batch-size', default=128, type=int, help='batch-size to use')
    parser.add_argument('--dataset', default='kitti_2015', choices=['kitti_2012', 'kitti_2015'], help='dataset')
    parser.add_argument('--seed', default=3, type=int, help='random seed')
    parser.add_argument('--patch-size', default=37, type=int, help='patch size from left image: 9 or 37')
    parser.add_argument('--disparity-range', default=201, type=int, help='disparity range')
    parser.add_argument('--learning-rate', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--reduction-factor', default=50, type=int,
                        help='ratio of the end learning rate to the starting learning rate')
    parser.add_argument('--find-patch-locations', default=False, help='find and store patch locations')
    parser.add_argument('--num_iterations', default=1, type=int, help='number of training iterations')
    parser.add_argument('--phase', default='training', choices=['training', 'testing', 'both'],
                        help='training or testing, if testing perform inference on test set.')
    parser.add_argument('--post-process', default=False, help='toggle use of post-processing.')
    parser.add_argument('--eval', default=False, help='compute error on validation set.')
    parser.add_argument('--test-all', default=False, help='run testing on all image pairs')
    parser.add_argument('--max-batches', default=100000, type=int,
                        help='regardless of dataset size limit the number of batches')
    parser.add_argument('--shuffle-images', default=False,
                        help='Shuffle the selection of images fed into the patch generator')

    settings = parser.parse_args()

# Settings, hyper-parameters.
settings.data_path = join(settings.data_path, 'training')
setattr(settings, 'exp_path', join(settings.result_dir, settings.exp_name))
setattr(settings, 'img_height', 370)
setattr(settings, 'img_width', 1224)
setattr(settings, 'half_patch_size', (settings.patch_size // 2))
setattr(settings, 'half_range', settings.disparity_range // 2)
setattr(settings, 'num_train', 160)
setattr(settings, 'model_name', join('model_', str(settings.patch_size)))

setattr(settings, 'left_img_folder', 'image_2')
setattr(settings, 'right_img_folder', 'image_3')
setattr(settings, 'disparity_folder', 'disp_noc_0')
setattr(settings, 'num_val', 40)
setattr(settings, 'num_input_channels', 3)
setattr(settings, 'image_dir', join(settings.exp_path, 'images'))
setattr(settings, 'patch_locations_path', join(settings.exp_path, 'patch_locations.pkl'))
setattr(settings, 'model_path', join(settings.exp_path, 'model.pt'))

os.makedirs(settings.exp_path, exist_ok=True)
os.makedirs(settings.image_dir, exist_ok=True)
settings_file = join(settings.exp_path, 'settings.log')
with open(settings_file, 'w') as the_file:
    the_file.write(str(settings))

##########################################################################
# Utilities
##########################################################################


LOG_FORMAT = '%(asctime)-15s %(levelname)-5s %(name)-15s - %(message)s'


def trim_image(img, img_height, img_width):
    """Trim image to remove pixels on the right and bottom.

    Args:
        img (numpy.ndarray): input image.
        img_height (int): desired image height.
        img_width (int): desired image width.

    Returns:
        (numpy.ndarray): trimmed image.

    """
    # PIL im.crop((left, upper, right, lower))
    return img[0:img_height, 0:img_width]


def setup_logging(log_path=None, log_level='DEBUG', logger=None, fmt=LOG_FORMAT):
    """Prepare logging for the provided logger.

    Args:
        log_path (str, optional): full path to the desired log file
        debug (bool, optional): log in verbose mode or not
        logger (logging.Logger, optional): logger to setup logging upon,
            if it's None, root logger will be used
        fmt (str, optional): format for the logging message

    """
    logger = logger if logger else logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers = []

    fmt = logging.Formatter(fmt=fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info('Log file is %s', log_path)


##########################################################################
# Input Pre Procesing
##########################################################################

def load_image_paths(data_path, left_img_folder, right_img_folder,
                     disparity_folder):
    """Load paths to images.

    Args:
        data_path (str): path to main dataset folder.
        left_img_folder (str): path to image folder with left camera images.
        right_img_folder (str): path to image folder with right camera images.
        disparity_folder (str): path to image folder with disparity images.

    Returns:
        (tuple): tuple of lists with paths to images.

    """
    left_image_paths = sorted(glob.glob(join(data_path, left_img_folder, '*10.png')))
    right_image_paths = sorted(glob.glob(join(data_path, right_img_folder, '*10.png')))
    disparity_image_paths = sorted(glob.glob(join(data_path, disparity_folder, '*10.png')))

    assert len(left_image_paths) == len(right_image_paths)
    assert len(left_image_paths) == len(disparity_image_paths)

    return left_image_paths, right_image_paths, disparity_image_paths


def _is_valid_location(sample_locations, img_width, img_height,
                       half_patch_size, half_range):
    """determines if the current location is valid, specifically that the patch
       is within the image.

    Args:
        sample locations (4-tuple): the co-ordinates for the center of the left
        and right patches.
        img_height (int): image height.
        img_width (int): image width.
        half_patch_size (int): half of the patch size.
        half_range (int): half of the disparity range.

    Returns:
        (boolean): locations are valid for processing

    """

    left_center_x, left_center_y, right_center_x, right_center_y = sample_locations
    is_valid_loc_left = ((left_center_x + half_patch_size + 1) <= img_width) and \
                        ((left_center_x - half_patch_size) >= 0) and \
                        ((left_center_y + half_patch_size + 1) <= img_height) and \
                        ((left_center_y - half_patch_size) >= 0)
    is_valid_loc_right = ((right_center_x - half_range - half_patch_size) >= 0) and \
                         ((right_center_x + half_range + half_patch_size + 1) <= img_width) and \
                         ((right_center_y - half_patch_size) >= 0) and \
                         ((right_center_y + half_patch_size + 1) <= img_height)

    return (is_valid_loc_left and is_valid_loc_right)


def _compute_valid_locations(disparity_image_paths, sample_ids, img_height,
                             img_width, half_patch_size, half_range):
    """compile a list of all the valid patch locations, taking those which are
       non-zero from the disparity images (noc or not occluded) and removing
       those which are too close to the perimeter to keep the patches within
       the image.

    Args:
        disparity_image_paths (list): list of paths to disparity images.
        sample_ids (list): list of indexes into the disparity_image_paths
        img_height (int): image height.
        img_width (int): image width.
        half_patch_size (int): half of the patch size.
        half_range (int): half of the disparity range.

    Returns:
        (np array of tuples): list of all valid locations, a tuple comprised of
        corresponding image index, the left_pixel column, the row for both
        pixels and the right pixel column.

    """
    num_samples = len(sample_ids)
    num_valid_locations = np.zeros(num_samples)

    # print("Number of images : ", len(sample_ids))
    for i, idx in enumerate(sample_ids):
        disp_img = np.array(Image.open(disparity_image_paths[idx])).astype('float64')
        # NOTE: We want images of same size for efficient loading.
        disp_img = trim_image(disp_img, img_height, img_width)
        disp_img /= 256
        max_disp_img = np.max(disp_img)
        num_valid_locations[i] = (disp_img != 0).sum()
        # print(disparity_image_paths[idx], " Max disp: ", max_disp_img, "Num valid locations : ",  num_valid_locations[i])

    num_valid_locations = int(num_valid_locations.sum())
    valid_locations = np.zeros((num_valid_locations, 4))
    valid_count = 0

    for i, idx in enumerate(sample_ids):
        disp_img = np.array(Image.open(disparity_image_paths[idx])).astype('float64')
        # NOTE: We want images of same size for efficient loading.
        disp_img = disp_img[0:img_height, 0:img_width]
        disp_img /= 256
        row_locs, col_locs = np.where(disp_img != 0)
        img_height, img_width = disp_img.shape

        for j, row_loc in enumerate(row_locs):
            left_center_x = col_locs[j]
            left_center_y = row_loc
            right_center_x = int(round(col_locs[j] - disp_img[left_center_y,
                                                              left_center_x]))
            right_center_y = left_center_y  # Stereo pair is assumed to be rectified.

            sample_locations = (left_center_x, left_center_y, right_center_x, right_center_y)
            if _is_valid_location(sample_locations, img_width, img_height,
                                  half_patch_size, half_range):
                location_info = np.array([idx, left_center_x,
                                          left_center_y,
                                          right_center_x])
                valid_locations[valid_count, :] = location_info
                valid_count += 1
    valid_locations = valid_locations[0:valid_count, :]
    print("Total Number of valid locations: ", valid_count)
    # NOTE: Shuffle patch locations info here, this will be used to directly
    # present samples in a min-batch while training.
    np.random.shuffle(valid_locations)

    return valid_locations


def find_and_store_patch_locations(settings):
    """Find patch locations, save locations array to pkl formatted file.  Data
       is organized as a pair of dictionairies within a dictionary.  The root
       dictionary divides the data into training and validation data.  The sub
       dictionaries are identical in format and store the image indicies and a
       tuple for each valid location comprised of corresponding image index, the
       left_pixel column, the row for both pixels and the right pixel column.

    Args:
        settings (argparse.Namespace): settings for the project derived from
        main script.

    Returns:
        Nothing

    """
    left_image_paths, right_image_paths, disparity_image_paths = \
        load_image_paths(settings.data_path, settings.left_img_folder,
                         settings.right_img_folder, settings.disparity_folder)
    sample_indices = list(range(len(left_image_paths)))
    if settings.shuffle_images:
        shuffle(sample_indices)
    train_ids = sample_indices[0:settings.num_train]
    val_ids = sample_indices[settings.num_train:]

    # Training set.
    valid_locations_train = _compute_valid_locations(disparity_image_paths,
                                                     train_ids,
                                                     settings.img_height,
                                                     settings.img_width,
                                                     settings.half_patch_size,
                                                     settings.half_range)
    # Validation set.
    valid_locations_val = _compute_valid_locations(disparity_image_paths,
                                                   val_ids,
                                                   settings.img_height,
                                                   settings.img_width,
                                                   settings.half_patch_size,
                                                   settings.half_range)

    # Save to disk

    contents_to_save = {'train': {'ids': train_ids, 'valid_locations': valid_locations_train},
                        'val': {'ids': val_ids, 'valid_locations': valid_locations_val}}

    os.makedirs(settings.exp_path, exist_ok=True)
    with open(settings.patch_locations_path, 'wb') as handle:
        pickle.dump(contents_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


##########################################################################
# Dataset Class
##########################################################################


def _load_image(image_path, img_height, img_width):
    """Load image and convert to Tensor.

    Args:
        image_path (str): path to image.
        img_height (int): desired height of output image (excess trimmed).
        img_width (int): desired width of output image (excess trimmed).

    Returns:
        img (Tensor): image array as tensor.

    """

    img = Image.open(image_path)
    tran = transforms.ToTensor()
    img = tran(img)
    img = img[:, :img_height, :img_width]

    return img


def _load_disparity(image_path, img_height, img_width):
    """Load disparity image as numpy array.

    Args:
        image_path (str): path to disparity image.
        img_height (int): desired height of output image (excess trimmed).
        img_width (int): desired width of output image (excess trimmed).

    Returns:
        disp_img (numpy.ndarray): disparity image array as tensor.

    """
    disp_img = np.array(Image.open(image_path)).astype('float64')
    disp_img = trim_image(disp_img, img_height, img_width)
    disp_img /= 256

    return disp_img


def _load_images(left_image_paths, right_image_paths, disparity_paths, img_height, img_width):
    """Load left, right images as tensors and disparity as a numpy array.

    Args:
        left_image_paths (list): list of paths to left images.
        right_image_paths (list): list of paths to right images.
        disparity_paths (list): list of paths to disparity files.
        img_height (int): desired height of output image (excess trimmed).
        img_width (int): desired width of output image (excess trimmed).

    Returns:
        tuple of (left (Tensor), right (Tensor), disparity images (numpy array))

    """
    left_images = []
    right_images = []
    disparity_images = []
    num_images = len(left_image_paths)
    print("Num Images: ", num_images)
    for idx in range(num_images):
        left_images.append(_load_image(left_image_paths[idx], img_height, img_width))
        right_images.append(_load_image(right_image_paths[idx], img_height, img_width))

        if disparity_paths:
            disparity_images.append(_load_disparity(disparity_paths[idx], img_height, img_width))

    return (left_images, right_images, np.array(disparity_images))


def _get_labels(disparity_range, half_range):
    """Creates the default disparity range for ground truth.  This does not shift
       the center of the target based upon the disparity value, but creates one
       array for all disparities.

    Args:
        disparity range (int): the maximum possible disparity value.
        half_range (int): half of the maximum possible disparity value.

    Returns:
        gt (numpy array): the array used as "ground truth"

    """
    gt = np.zeros((disparity_range))

    # NOTE: Smooth targets are [0.05, 0.2, 0.5, 0.2, 0.05], hard-coded.
    gt[half_range - 2: half_range + 3] = np.array([0.05, 0.2, 0.5, 0.2, 0.05])

    return gt


class SiameseDataset(Dataset):
    """Dataset class to provide training and validation data.

    When initialized, loads patch locations info from file, loads all left and
    right camera images into memory for enabling fast loading.

    Attributes:
        left_images (Tensor): tensor of all left camera images.
        right_images (Tensor): tensor of all right camera images.

    """

    def __init__(self, settings, patch_locations, transform=None):
        """Constructor.

        Args:
            settings (argparse.Namespace): settings for the project derived from
            main script.
            patch_locations (dict): dict with arrays containing patch locations
            info.
            transform (function): not implimented

        """
        self._settings = settings
        left_image_paths, right_image_paths, disparity_paths = \
            load_image_paths(settings.data_path,
                             settings.left_img_folder,
                             settings.right_img_folder,
                             settings.disparity_folder)

        self.left_images, self.right_images, self.disparity_images = \
            _load_images(left_image_paths,
                         right_image_paths,
                         disparity_paths,
                         settings.img_height,
                         settings.img_width)

        self.sample_ids = patch_locations['ids']
        self.patch_locations = patch_locations
        self.length = len(self.patch_locations['valid_locations'])
        self.transform = transform

    def __len__(self):
        'Returns the total number of samples'
        return self.length

    def _pytorch_parse_function(self, sample_info):
        """Creates the default disparity range for ground truth.  This does not
           shift the center of the target based upon the disparity value, but
           creates one array for all disparities.

        Args:
          sample_info (list): the encoded sample patch location information of
                              index, left column, row, right column

        Returns:
          left_patch (Tensor): sub-image left patch
          right_patch (Tensor): sub-image right patch
          labels (numpy array): the array used as "ground truth"
        """

        idx = sample_info[0]
        left_center_x = sample_info[1]
        left_center_y = sample_info[2]
        right_center_x = sample_info[3]

        left_image = self.left_images[idx]
        right_image = self.right_images[idx]

        left_patch = left_image[:,
                     left_center_y - self._settings.half_patch_size:
                     left_center_y + self._settings.half_patch_size + 1,
                     left_center_x - self._settings.half_patch_size:
                     left_center_x + self._settings.half_patch_size + 1]
        right_patch = right_image[:,
                      left_center_y - self._settings.half_patch_size:
                      left_center_y + self._settings.half_patch_size + 1,
                      right_center_x - self._settings.half_patch_size - self._settings.half_range:
                      right_center_x + self._settings.half_patch_size + self._settings.half_range + 1]

        labels = _get_labels(self._settings.disparity_range, self._settings.half_range)

        return left_patch, right_patch, labels

    def __getitem__(self, index):
        """Generates one sample of data to itterate training on.  __getitem__
           can be called by using the index of the assigend variable.  I.e.
           dataset = SiameseDataset(self, settings, patch_locations)
           dataset[0] will call __getitem__ with an index of 0
        Args:
          index (int): the index into the array of patch sets.

        Returns:
          left_patch (Tensor): sub-image left patch
          right_patch (Tensor): sub-image right patch
          labels (Tensor): the array used as "ground truth"
        """

        # Loading the image
        if index > self.length - 1:
            print("Index is too large : ", index, "Dataset length : ", self.length)
        sample_info = np.zeros((4,), dtype=int)
        # Convert location information from floats into ints
        sample_info[0] = int(self.patch_locations['valid_locations'][index][0])
        sample_info[1] = int(self.patch_locations['valid_locations'][index][1])
        sample_info[2] = int(self.patch_locations['valid_locations'][index][2])
        sample_info[3] = int(self.patch_locations['valid_locations'][index][3])
        left_patch, right_patch, labels = self._pytorch_parse_function(sample_info)

        # Apply image transformations (not currently used)
        if self.transform is not None:
            left_patch = self.transform(left_patch)
            right_patch = self.transform(right_patch)
        return left_patch, right_patch, torch.from_numpy(np.array(labels, dtype=np.float32))


##########################################################################
# Models
##########################################################################

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers for 37x37 input patches
        self.cnn1 = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=5),  # 1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=5),  # 2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=5),  # 3
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=5),  # 4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=5),  # 5
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=5),  # 6
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=5),  # 7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=5),  # 8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=5),  # 9 - no ReLu on Output
            nn.BatchNorm2d(64),

        )

    def forward(self, left_patch, right_patch):
        left_feature = self.cnn1(left_patch)
        right_feature = self.cnn1(right_patch)
        return left_feature, right_feature


class SiameseNetwork13(nn.Module):
    def __init__(self):
        super(SiameseNetwork13, self).__init__()

        # Setting up the Sequential of CNN Layers for 13x13 input patches
        self.cnn1 = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=5),  # 1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=5),  # 2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=5),  # 9 - no ReLu on Output
            nn.BatchNorm2d(64),

        )

    def forward(self, left_patch, right_patch):
        left_feature = self.cnn1(left_patch)
        right_feature = self.cnn1(right_patch)
        return left_feature, right_feature


##########################################################################
# Loss Function
##########################################################################

class InnerProductLoss(torch.nn.Module):
    """
       To aid in training the inner product loss calculates the softmax with
       logits on a lables which have a probability distribution around the
       ground truth.  As a result the labels are not 0/1 integers, but floats.
    """

    def __init__(self):
        super(InnerProductLoss, self).__init__()

    def forward(self, left_feature, right_feature, labels):
        """Calculate the loss describe above.

        Args:
          left_feature (Tensor): output of from the left patch passing through
                                 the Siamese network of dimensions (1, 1, 1, 64)
          right_feature (Tensor): output of from the right patch passing through
                                  the Siamese network of dimensions
                                  (1, 1, disparity_range, 64)
          lables (Tensor): the "ground truth" of the expected disparity for the
                           patches of dimensions (1, disparity_range)


        Returns:
          left_patch (Tensor): sub-image left patch
          right_patch (Tensor): sub-image right patch
          labels (numpy array): the array used as "ground truth"
        """
        left_feature = torch.squeeze(left_feature)
        # perform inner product of left and right features
        inner_product = torch.einsum('il,iljk->ik', left_feature, right_feature)
        # peform the softmax with logits in two steps.  torch does not support
        # softmax with logits on float labels, so the calculation is broken
        # into calculating yhat and then the loss
        yhat = F.log_softmax(inner_product, dim=-1)
        loss = -1.0 * torch.einsum('ij,ij->i', yhat, labels).sum() / yhat.size(0)

        return loss


##########################################################################
# Adaptive Learning Rate
##########################################################################

def adjust_learning_rate(optimizer, step, num_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    step_size = settings.reduction_factor ** (1.0 / num_steps)
    lr = settings.learning_rate * ((1.0 / step_size) ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


##########################################################################
# Training
##########################################################################

def train():
    counter = []
    loss_history = []
    lr = settings.learning_rate

    print("Number of iterrations : ", settings.num_iterations)
    for epoch in range(settings.num_iterations):
        # print ("Epoch : ", epoch)
        for i, data in enumerate(train_dataloader, 0):
            # print("Batch : ", i)
            left_patch, right_patch, labels = data
            # print("Data Size : ", left_patch.size())
            left_patch, right_patch, labels = left_patch.cuda(), right_patch.cuda(), labels.cuda()
            optimizer.zero_grad()
            left_feature, right_feature = net(left_patch, right_patch)
            loss_inner_product = criterion(left_feature, right_feature, labels)
            loss_inner_product.backward()
            optimizer.step()
            if i % 1000 == 0:
                # Note validation set must be >= %1 of training set for iterator to not break when it runs out of validation data
                left_patch, right_patch, labels = next(val_dataset_iterator)
                left_patch, right_patch, labels = left_patch.cuda(), right_patch.cuda(), labels.cuda()
                optimizer.zero_grad()
                left_feature, right_feature = net(left_patch, right_patch)
                val_loss_inner_product = criterion(left_feature, right_feature, labels)
                print("{}, Epoch: {}, Batch: {}, Learning Rate: {}, Training loss: {}, Validation loss: {}".format(
                    datetime.datetime.now(tz=pytz.utc), epoch, i, lr, loss_inner_product.item(),
                    val_loss_inner_product.item()))
                loss_history.append(loss_inner_product.item())
                lr = adjust_learning_rate(optimizer, int(i / 100), int(num_batches / 100))
            if i == settings.max_batches:
                break
    return net


##########################################################################
# Inference Functions
##########################################################################

def save_images(images, cols, titles, directory, filename):
    """Save multiple images arranged as a table.

    Args:
        images (list): list of images to display as numpy arrays.
        cols (int): number of columns.
        titles (list): list of title strings for each image.
        iteration (int): iteration counter or plot interval.

    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure(figsize=(20, 10))
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        image = np.squeeze(image)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)

        a.axis('off')
        a.set_title(title, fontsize=40)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.savefig(join(directory, filename), bbox_inches='tight')
    plt.close(fig)


def prediction_to_image(disp_prediction):
    """
    Args:
        disp_prediction (Tensor): disparity prediction.

    Returns:
        image (float): PNG formatted image

    """
    disp_img = np.array(disp_prediction)
    disp_img[disp_img < 0] = 0
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=disp_img.min(), vmax=disp_img.max())
    return cmap(norm(disp_img))


def calc_error(disparity_prediction, disparity_ground_truth, idx):
    """Meaures the disparity prediction pixel by pixel against the ground truth
       printing the result for each image.  Prints and returns the average error
       over all of the predictions.

    Args:
        disparity_prediction (Tensor): images of per pixel disparity.
        disparity_ground_truth (Tensor): dataset supplied image of per pixel
                                         disparity.

    Returns:
        mean_error (float): 3-pixel mean error over disparity set.

    """
    error = 0

    valid_gt_pixels = (disparity_ground_truth != 0).astype('float')
    masked_prediction_valid = disparity_prediction * valid_gt_pixels
    num_valid_gt_pixels = valid_gt_pixels.sum()

    # Use 3-pixel error metric for now.
    num_error_pixels = (np.abs(masked_prediction_valid - disparity_ground_truth) > 3).sum()
    error += num_error_pixels / num_valid_gt_pixels

    print('Error: {:04f}, for image index {}'.format(error, idx))

    return error


def apply_cost_aggregation(cost_volume):
    """Apply cost-aggregation post-processing to network predictions.

    Performs an average pooling operation over raw network predictions to smoothen
    the output.

    Args:
        cost_volume (Tensor): cost volume predictions from network.

    Returns:
        cost_volume (Tensor): aggregated cost volume.

    """
    # NOTE: Not ideal but better than zero padding, since we average.
    # two DimenionPad specified from last to first the padding on the dimesion, so the first
    # two elements of the tuple specify the padding before / after the last dimension
    # while the third and forth elemements of the tuple specify the padding before / after
    # the second to last dimension

    # torch padding is strange, behaviors change for "constant" versus "reflect"
    # with "reflect having strange behavior".  For a 4D vector "reflect" must be
    # done over two (last and 2nd to last dimensions).  To support the desired
    # behavoir will have to shift dimensions.
    # desired, but must be done one at a time fourDimensionPad = (0, 0, 2, 2, 2, 2, 0, 0)
    # Cost volume arrives as 1 x H x W x C
    twoDimensionPad = (2, 2, 2, 2)
    # print("Cost Volume Shape : ", cost_volume.size())
    cost_volume.permute(3, 0, 1, 2)  # barrel shift right
    cost_volume = F.pad(cost_volume, twoDimensionPad, "reflect", 0)
    cost_volume.permute(1, 2, 3, 0)  # back to original order

    return F.avg_pool2d(cost_volume, kernel_size=5, stride=1)


def calc_cost_volume(left_features, right_features, mask=None):
    """
    Calculate the cost volume to generate predicted disparities.  Compute a
    batch matrix multiplication to compute inner-product over entire image and
    obtain a cost volume.

    Args:
        left_features (Tensor): left image features post forward pass through CNN.
        right_features (Tensor): right image features post forward pass through CNN.
        mask (, optional): mask

    Returns:
        inner_product (Tensor): output from feature matching model, size
        of tensor 1 x H x W x 201

    """
    inner_product, win_indices = [], []
    img_height, img_width = right_feature.shape[2], right_feature.shape[3]
    # print("right feature shape : ", right_feature.size())
    row_indices = torch.arange(0, img_width, dtype=torch.int64)

    for i in range(img_width):
        # print("left feature shape before squeeze : ", left_feature.size())
        left_column_features = torch.squeeze(left_feature[:, :, :, i])
        # print("left feature shape after squeeze  : ", left_column_features.size())
        start_win = max(0, i - settings.half_range)
        end_win = max(settings.disparity_range, settings.half_range + i + 1)
        start_win = start_win - max(0, end_win - img_width)  # was img_width.value
        end_win = min(img_width, end_win)

        right_win_features = torch.squeeze(right_feature[:, :, :, start_win:end_win])
        win_indices_column = torch.unsqueeze(row_indices[start_win:end_win], 0)
        # print("left feature shape      : ", left_column_features.size())
        # print("right win feature shape : ", right_win_features.size())
        inner_product_column = torch.einsum('ij,ijk->jk', left_column_features,
                                            right_win_features)
        inner_product_column = torch.unsqueeze(inner_product_column, 1)
        inner_product.append(inner_product_column)
        win_indices.append(win_indices_column)
    # print("inner_product len   : ", len(inner_product))
    # print("inner_product width : ", len(inner_product[0]))
    # print("win_indices len   : ", len(win_indices))
    # print("win_indices width : ", len(win_indices[0]))
    inner_product = torch.unsqueeze(torch.cat(inner_product, 1), 0)
    # print("inner_product after unsqueeze  : ", inner_product.size())
    win_indices = torch.cat(win_indices, 0).to(device)
    # print("win_indices shape : ", win_indices.size())
    return inner_product, win_indices


def inference(left_features, right_features, post_process):
    """Post process model output.

    Args:
        left_features (Tensor): left input cost volume.
        right_features (Tensor): right input cost volume.

    Returns:
        disp_prediction (Tensor): disparity prediction.

    """
    cost_volume, win_indices = calc_cost_volume(left_features, right_features)
    # print("Cost Volume Shape : ", cost_volume.size())
    img_height, img_width = cost_volume.shape[1], cost_volume.shape[2]  # Now 1 x C X H x W, was 1 x H x W x C
    if post_process:
        cost_volume = apply_cost_aggregation(cost_volume)
    cost_volume = torch.squeeze(cost_volume)
    # print("Cost Volume Shape (post squeeze): ", cost_volume.size())
    row_indices, _ = torch.meshgrid(torch.arange(0, img_width, dtype=torch.int64),
                                    torch.arange(0, img_height, dtype=torch.int64))

    disp_prediction_indices = torch.argmax(input=cost_volume, dim=-1)

    disp_prediction = []
    for i in range(img_width):
        column_disp_prediction = torch.gather(input=win_indices[i],
                                              index=disp_prediction_indices[:, i],
                                              dim=0)
        column_disp_prediction = torch.unsqueeze(column_disp_prediction, 1)
        disp_prediction.append(column_disp_prediction)

    # print("disp_prediction length (pre cat) : ", len(disp_prediction))
    disp_prediction = torch.cat(disp_prediction, 1)
    # print("disp_prediction shape (post cat): ", disp_prediction.size())
    # print("row indices                     : ", row_indices.size())
    disp_prediction = row_indices.permute(1, 0).to(device) - disp_prediction

    return disp_prediction


##########################################################################
# Main
##########################################################################

def main():

    # Python logging.
    LOGGER = logging.getLogger(__name__)
    exp_dir = join('experiments', '{}'.format(settings.exp_name))
    log_file = join(exp_dir, 'log.log')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(join(exp_dir, 'qualitative_samples'), exist_ok=True)
    setup_logging(log_path=log_file, log_level=settings.log_level, logger=LOGGER)

    random.seed(settings.seed)

    patch_locations_loaded = 'patch_locations' in locals() or 'patch_locations' in globals()
    if not (patch_locations_loaded) or patch_locations == None:
        if not os.path.exists(settings.patch_locations_path):
            print("New patch file being generated")
            find_and_store_patch_locations(settings)
        with open(settings.patch_locations_path, 'rb') as handle:
            print("Loading existing patch file")
            patch_locations = pickle.load(handle)
    else:
        print("Patch file already loaded")
    '''
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    '''
    # Assign GPU
    if not torch.cuda.is_available():
        print("GPU not available!")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    # Declare Siamese Network
    if settings.patch_size == 13:
        net = SiameseNetwork13().cuda()
        model = SiameseNetwork13().to(device)
        torchsummary.summary(model, input_size=[(3, 13, 13), (3, 13, 213)])
    else:
        net = SiameseNetwork().cuda()
        model = SiameseNetwork().to(device)
        torchsummary.summary(model, input_size=[(3, 37, 37), (3, 37, 237)])
    sys.stdout.flush()  # flush torchsummary.summary output

    if settings.phase == 'training' or settings.phase == 'both':
        training_dataset = SiameseDataset(settings, patch_locations['train'])
        # print("Batch Size : ", settings.batch_size)
        print("Loading training dataset")
        train_dataloader = DataLoader(training_dataset,
                                      shuffle=True,
                                      num_workers=8,
                                      batch_size=settings.batch_size)

        val_dataset = SiameseDataset(settings, patch_locations['val'])

        # print("Batch Size : ", settings.batch_size)
        print("Loading validation dataset")
        val_dataloader = DataLoader(val_dataset,
                                    shuffle=True,
                                    num_workers=2,
                                    batch_size=settings.batch_size)

        val_dataset_iterator = iter(val_dataloader)

        num_batches = len(train_dataloader)
        print("Number of ", settings.batch_size, "patch batches", num_batches)

        # Decalre Loss Function
        criterion = InnerProductLoss()
        # Declare Optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=settings.learning_rate)

        print("Start Training")
        # Train the model
        model = train()
        torch.save(model.state_dict(), settings.model_path)
        print("Model Saved Successfully")

    if settings.phase == 'testing' or settings.phase == 'both':
        print("Start Testing")
        # Load the saved model
        model.load_state_dict(torch.load(settings.model_path))
        model.eval()  # required for batch normalization to function correctly
        patch_locations_loaded = 'patch_locations' in locals() or 'patch_locations' in globals()
        if not (patch_locations_loaded) or patch_locations == None:
            with open(settings.patch_locations_path, 'rb') as handle:
                patch_locations = pickle.load(handle)

        test_image_indices = patch_locations['val']['ids']
        train_image_indices = patch_locations['train']['ids']

        counter = 0
        left_image_paths, right_image_paths, disparity_image_paths = load_image_paths(settings.data_path,
                                                                                      settings.left_img_folder,
                                                                                      settings.right_img_folder,
                                                                                      settings.disparity_folder)

        error_dict = {}
        for idx in test_image_indices:
            left_image_in = _load_image(left_image_paths[idx], settings.img_height, settings.img_width)
            right_image = _load_image(right_image_paths[idx], settings.img_height, settings.img_width)
            disparity_ground_truth = _load_disparity(disparity_image_paths[idx], settings.img_height, settings.img_width)

            # two DimenionPad specified from last to first the padding on the dimesion, so the first
            # two elements of the tuple specify the padding before / after the last dimension
            # while the third and forth elemements of the tuple specify the padding before / after
            # the second to last dimension
            twoDimensionPad = (
            settings.half_patch_size, settings.half_patch_size, settings.half_patch_size, settings.half_patch_size)
            left_image = F.pad(left_image_in, twoDimensionPad, "constant", 0)
            right_image = F.pad(right_image, twoDimensionPad, "constant", 0)
            left_image = torch.unsqueeze(left_image, 0)
            right_image = torch.unsqueeze(right_image, 0)
            # print("Left image size  : ", left_image.size())
            # print("Right image size : ", right_image.size())
            # left_feature, right_feature = model(left_image.to(device), right_image.to(device))
            left_feature, right_feature = model(left_image.to(device), right_image.to(device))
            # print("Left feature size  : ", left_feature.size())
            # print("Right feature size : ", right_feature.size())
            # print("Left Feature on Cuda: ", left_feature.get_device())
            disp_prediction = inference(left_feature, right_feature, post_process=True)
            error_dict[idx] = calc_error(disp_prediction.cpu(), disparity_ground_truth, idx)
            disp_image = prediction_to_image(disp_prediction.cpu())
            save_images([left_image_in.permute(1, 2, 0), disp_image], 1, ['left image', 'disparity'], settings.image_dir,
                        'disparity_{}.png'.format(idx))
            cv2.imwrite(join(settings.image_dir, (("00000" + str(idx))[-6:] + "_10.png")), np.array(disp_prediction.cpu()))
        if settings.test_all:
            for idx in train_image_indices:
                left_image_in = _load_image(left_image_paths[idx], settings.img_height, settings.img_width)
                right_image = _load_image(right_image_paths[idx], settings.img_height, settings.img_width)
                disparity_ground_truth = _load_disparity(disparity_image_paths[idx], settings.img_height,
                                                         settings.img_width)
                twoDimensionPad = (
                settings.half_patch_size, settings.half_patch_size, settings.half_patch_size, settings.half_patch_size)
                left_image = F.pad(left_image_in, twoDimensionPad, "constant", 0)
                right_image = F.pad(right_image, twoDimensionPad, "constant", 0)
                left_image = torch.unsqueeze(left_image, 0)
                right_image = torch.unsqueeze(right_image, 0)
                left_feature, right_feature = model(left_image.to(device), right_image.to(device))
                disp_prediction = inference(left_feature, right_feature, post_process=True)
                error_dict[idx] = calc_error(disp_prediction.cpu(), disparity_ground_truth, idx)
                disp_image = prediction_to_image(disp_prediction.cpu())
                save_images([left_image_in.permute(1, 2, 0), disp_image], 1, ['left image', 'disparity'],
                            settings.image_dir,
                            'disparity_{}.png'.format(idx))
                cv2.imwrite(join(settings.image_dir, (("00000" + str(idx))[-6:] + "_10.png")),
                            np.array(disp_prediction.cpu()))
        average_error = 0.0
        for idx in error_dict:
            average_error += error_dict[idx] / len(error_dict)
        print("Average Error : ", average_error)


if __name__ == "__main__":
    main()
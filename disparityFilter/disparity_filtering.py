'''  Pytorch implimentation of Disparity Filtering for Binocular Stereopsis via Image Matching
'''

import os
import sys
import glob
import pickle

from PIL import Image
import PIL.ImageOps
import numpy as np
import matplotlib.pyplot as plt

import logging
from os.path import join, isfile
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
import math
import torchsummary

##########################################################################
# Load Settings
##########################################################################

if False: #'google.colab' in str(get_ipython()):
    print("Running in colab mode")


    class Args:
        resume = False  # help='resume from checkpoint')
        data_path = '/content/kitti2015_plus'  # join('data', settings.dataset, settings.phase))
        exp_name = 'Urun6'
        result_dir = 'results'  # result directory
        log_level = 'INFO'  # choices = ['DEBUG', 'INFO'], help='log-level to use')
        batch_size = 4  # type=int, help='batch-size to use')
        seed = 3  # , type=int, help='random seed')
        patch_height = 316  # , type=int, help='patch height')
        patch_width = 316  # , type=int, help='patch width')
        num_patches = 10000  # , type=int, help='number of patches to train on')
        patch_overlap = 0  # , type=int, help='patch height')
        learning_rate = 0.0001  # , type=float, help='initial learning rate')
        reduction_factor = 25  # , type=int, help='ratio of the end learning rate to the starting learning rate')
        find_patch_locations = False  # , help='find and store patch locations')
        num_iterations = 1  # , type=int, help='number of training iterations')
        phase = 'both'  # , choices=['training', 'testing', 'both'], help='training or testing, if testing perform inference on test set.')
        eval = False  # , help='compute error on validation set.')
        test_all = True  # , help='run testing on all image pairs')
        max_batches = 200  # regardless of dataset size limit the number of batches
        shuffle_images = False  # Shuffle the selection of images fed into the patch generator)
        tensor_summary = False  # Enable the display of the tensor summary)


    settings = Args()
else:
    print("Running in script mode")
    parser = argparse.ArgumentParser(
        description='Re-implementation of Efficient Deep Learning for Stereo Matching')
    parser.add_argument('--resume', '-r', default=False, help='resume from checkpoint - not supported')
    parser.add_argument('--data-path', default='kitti2015_plus', type=str, help='root location of kitti_dataset')
    parser.add_argument('--exp-name', default='df_test1', type=str, help='name of experiment')
    parser.add_argument('--result-dir', default='results', type=str, help='results directory')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO'], help='log-level to use')
    parser.add_argument('--batch-size', default=4, type=int, help='batch-size to use')
    parser.add_argument('--seed', default=3, type=int, help='random seed')
    parser.add_argument('--patch-height', default=316, type=int, help='patch height')
    parser.add_argument('--patch-width', default=316, type=int, help='patch width')
    parser.add_argument('--num-patches', default=100000, type=int, help='number of training patches')
    parser.add_argument('--patch-overlap', default=0, type=int, help='overlap between patches')
    parser.add_argument('--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--reduction-factor', default=50, type=int,
                        help='ratio of the end learning rate to the starting learning rate')
    parser.add_argument('--find-patch-locations', default=False, help='find and store patch locations')
    parser.add_argument('--num_iterations', default=1, type=int, help='number of training iterations')
    parser.add_argument('--phase', default='training', choices=['training', 'testing', 'both'],
                        help='training or testing, if testing perform inference on test set.')
    parser.add_argument('--eval', default=False, help='compute error on validation set.')
    parser.add_argument('--test-all', default=False, help='run testing on all image pairs')
    parser.add_argument('--max-batches', default=100000, type=int,
                        help='regardless of dataset size limit the number of batches')
    parser.add_argument('--shuffle-images', default=False,
                        help='Shuffle the selection of images fed into the patch generator')
    parser.add_argument('--tensor-summary', default=False, help='print out tensor summary')

    settings = parser.parse_args()

# Settings, hyper-parameters.
settings.data_path = join(settings.data_path, 'training')
setattr(settings, 'exp_path', join(settings.result_dir, settings.exp_name))
setattr(settings, 'img_height', 370)
setattr(settings, 'img_width', 1224)
setattr(settings, 'num_train', 160)
setattr(settings, 'left_img_folder', 'image_2')
setattr(settings, 'right_img_folder', 'image_3')
setattr(settings, 'disparity_folder', 'disp_fnoc_0')
setattr(settings, 'ground_truth_folder', 'disp_noc_0')
setattr(settings, 'dmask_folder', 'disp_mask_0')
setattr(settings, 'disp_13x13_folder', 'disp_13x13')
setattr(settings, 'disp_37x37_folder', 'disp_37x37')
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

def load_image_paths(data_path, left_img_folder, disparity_folder, dmask_folder,
                     disp_13x13_folder, disp_37x37_folder):
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
    disparity_image_paths = sorted(glob.glob(join(data_path, disparity_folder, '*10.png')))
    dmask_image_paths = sorted(glob.glob(join(data_path, dmask_folder, '*10.png')))
    disp_13x13_image_paths = sorted(glob.glob(join(data_path, disp_13x13_folder, '*10.png')))
    disp_37x37_image_paths = sorted(glob.glob(join(data_path, disp_37x37_folder, '*10.png')))

    return left_image_paths, disparity_image_paths, dmask_image_paths, disp_13x13_image_paths, disp_37x37_image_paths


def _is_valid_location(sample_locations, img_width, img_height,
                       half_patch_size, half_range):
    """determines if the current location is valid, specifically that the patch
       is within the image.

    Args:
        sample locations (tuple): the co-ordinates for the center of the left
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


def _compute_valid_locations(settings, disparity_paths, disp_37x37_paths, sample_ids, img_height, img_width,
                             train=False):
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
    print("Computing Valid Locations")
    error_map = np.zeros((sample_ids[-1] + 1, img_height, img_width))
    disp_37x37_images = []
    for idx in sample_ids:
        if disparity_paths:
            disparity_image = _load_disparity(disparity_paths[idx], img_height, img_width)
        if disp_37x37_paths:
            disp_37x37_image = _load_disp_XxX(disp_37x37_paths[idx], img_height, img_width)
        mask = np.ceil(disparity_image / 256)
        masked_37x27_image = np.multiply(mask, disp_37x37_image)
        disp_error = np.abs(masked_37x27_image - disparity_image)
        three_pixel_error = ((disp_error > 3) * 1).astype(int)
        error_map[idx] = three_pixel_error

    valid_locations = np.zeros((settings.num_patches, 3))

    num_samples = len(sample_ids)
    num_valid_locations = np.zeros(num_samples)
    coverage_locations = []
    loc_idx = 0
    for idx in sample_ids:
        upper_left_x = 0
        # print("[",idx,"]", " three pixel error total ", np.sum(error_map[idx]))
        for x in range(math.ceil(img_width / settings.patch_width)):
            upper_left_y = 0
            for y in range(math.ceil(img_height / settings.patch_height)):
                if upper_left_y + settings.patch_height > img_height:
                    upper_left_y = img_height - settings.patch_height
                if upper_left_x + settings.patch_width > img_width:
                    upper_left_x = img_width - settings.patch_width
                num_errors = np.sum(error_map[idx][upper_left_y:upper_left_y + settings.patch_height,
                                    upper_left_x:upper_left_x + settings.patch_width])
                if num_errors > 3:
                    valid_locations[loc_idx] = (idx, upper_left_x, upper_left_y)
                    # print("Error total :", num_errors, idx, upper_left_x, upper_left_y)
                    loc_idx += 1
                upper_left_y += settings.patch_height
            upper_left_x += settings.patch_width

    print("Calculating Random Locations")
    random_start = loc_idx
    if train:
        num_random_samples = max(settings.num_patches - loc_idx, 0)
        max_x = max(img_width - settings.patch_width, 0)
        max_x = 116
        max_y = max(img_height - settings.patch_height, 0)
        while loc_idx < settings.num_patches:
            rand_idx = random.choice(sample_ids)
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            if np.sum(error_map[rand_idx][y:y + settings.patch_height, x:x + settings.patch_width]) > 3:
                valid_locations[loc_idx] = (rand_idx, x, y)
                loc_idx += 1
    else:  # ensure validation valid locations > 1% of the training validation locations
        num_copies = math.ceil(settings.num_patches / (100 * loc_idx))
        if num_copies > 1:
            for i in range(num_copies):
                for j in range(loc_idx):
                    valid_locations[i * num_copies + j + loc_idx] = valid_locations[j]
        valid_locations = valid_locations[0:num_copies * loc_idx]

    print("Total Number of valid locations: ", len(valid_locations))
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
    left_image_paths, disparity_paths, dmask_paths, disp_13x13_paths, disp_37x37_paths = \
        load_image_paths(settings.data_path,
                         settings.left_img_folder,
                         settings.disparity_folder,
                         settings.dmask_folder,
                         settings.disp_13x13_folder,
                         settings.disp_37x37_folder)
    sample_indices = list(range(len(left_image_paths)))
    if settings.shuffle_images:
        shuffle(sample_indices)
    train_ids = sample_indices[0:settings.num_train]
    val_ids = sample_indices[settings.num_train:]

    # Training set.
    valid_locations_train = _compute_valid_locations(settings,
                                                     disparity_paths,
                                                     disp_37x37_paths,
                                                     train_ids,
                                                     settings.img_height,
                                                     settings.img_width,
                                                     train=True)
    # print("3", valid_locations_train[1], valid_locations_train[11], valid_locations_train[12201])
    # Validation set.
    valid_locations_val = _compute_valid_locations(settings,
                                                   disparity_paths,
                                                   disp_37x37_paths,
                                                   val_ids,
                                                   settings.img_height,
                                                   settings.img_width,
                                                   train=False)

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

def _load_blended_disparity(ground_truth_path, mask_path, estimate_path, img_height, img_width):
    """Load disparity image as numpy array.

    Args:
        image_path (str): path to disparity image.
        img_height (int): desired height of output image (excess trimmed).
        img_width (int): desired width of output image (excess trimmed).

    Returns:
        disp_img (numpy.ndarray): disparity image array as tensor.

    """
    mask_img = np.array(Image.open(mask_path))
    mask_img = trim_image(mask_img, img_height, img_width)
    invert_mask = (mask_img == 0) * 1
    est_img = np.array(Image.open(estimate_path))
    est_img = trim_image(est_img, img_height, img_width)
    disp_img = np.array(Image.open(ground_truth_path)).astype('float64')
    disp_img = trim_image(disp_img, img_height, img_width)
    disp_img /= 256
    disp_img = disp_img + invert_mask * est_img

    return disp_img


def _load_disp_XxX(image_path, img_height, img_width):
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

    return disp_img


def _load_images(left_image_paths, disparity_paths, disp_13x13_paths, disp_37x37_paths, img_height, img_width):
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
    disparity_images = []
    disp_13x13_images = []
    disp_37x37_images = []
    num_images = len(left_image_paths)
    print("Num Images: ", num_images)
    for idx in range(num_images):
        left_images.append(_load_image(left_image_paths[idx], img_height, img_width))

        if disparity_paths:
            disparity_images.append(_load_disparity(disparity_paths[idx], img_height, img_width))

        if disp_13x13_paths:
            disp_13x13_images.append(_load_disp_XxX(disp_13x13_paths[idx], img_height, img_width))

        if disp_37x37_paths:
            disp_37x37_images.append(_load_disp_XxX(disp_37x37_paths[idx], img_height, img_width))

    return (left_images, np.array(disparity_images), np.array(disp_13x13_images), np.array(disp_37x37_images))


def get_concat_image_w_disparity(settings, idx, blended=True):
    """
    """
    left_image_paths, disparity_image_paths, dmask_image_paths, disp_13x13_image_paths, disp_37x37_image_paths = \
        load_image_paths(settings.data_path,
                         settings.left_img_folder,
                         settings.disparity_folder,
                         settings.dmask_folder,
                         settings.disp_13x13_folder,
                         settings.disp_37x37_folder)

    ground_truth_paths = sorted(glob.glob(join(settings.data_path, settings.ground_truth_folder, '*10.png')))
    img_height = settings.img_height
    img_width = settings.img_width
    left_image = _load_image(left_image_paths[idx], img_height, img_width)
    if blended:
        disparity_image = _load_blended_disparity(disparity_image_paths[idx], dmask_image_paths[idx], disp_37x37_image_paths[idx], img_height, img_width)
    else:
        disparity_image = _load_disparity(ground_truth_paths[idx], img_height, img_width)
    disp_13x13_image = _load_disp_XxX(disp_13x13_paths[idx], img_height, img_width)
    disp_37x37_image = _load_disp_XxX(disp_37x37_paths[idx], img_height, img_width)

    disp_13x13_tensor = torch.unsqueeze(torch.from_numpy(np.array(disp_13x13_image, dtype=np.float32)), 0)
    disp_37x37_tensor = torch.unsqueeze(torch.from_numpy(np.array(disp_37x37_image, dtype=np.float32)), 0)

    concat_image = torch.cat([left_image, disp_13x13_tensor, disp_37x37_tensor], dim=0)

    disparity_ground_truth = torch.from_numpy(np.array(disparity_image, dtype=np.float32))

    return left_image, concat_image, disparity_image


class DisparityDataset(Dataset):
    """Dataset class to provide training and validation data.

    When initialized, loads patch locations info from file, loads all left camera,
    13x13_disparity and 37x37_disparity images into memory for enabling fast loading.
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
        left_image_paths, disparity_paths, dmask_paths, disp_13x13_paths, disp_37x37_paths = \
            load_image_paths(settings.data_path,
                             settings.left_img_folder,
                             settings.disparity_folder,
                             settings.dmask_folder,
                             settings.disp_13x13_folder,
                             settings.disp_37x37_folder)

        self.left_images, self.disparity_images, self.disp_13x13_images, self.disp_37x37_images = \
            _load_images(left_image_paths,
                         disparity_paths,
                         disp_13x13_paths,
                         disp_37x37_paths,
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
        """Creates the stacked input volume of R, G, B, 13x13disparity, 37x37_disparity

        Args:
          sample_info (list): the encoded sample patch location information of
                              index, left column, row

        Returns:
          input_patch (Tensor): sub-image input patch
          labels (numpy array): the array used as "ground truth"
        """

        idx = sample_info[0]
        x = sample_info[1]
        y = sample_info[2]

        left_image = self.left_images[idx]
        disparity_image = self.disparity_images[idx]
        disp_13x13_image = self.disp_13x13_images[idx]
        disp_37x37_image = self.disp_37x37_images[idx]

        x2 = x + settings.patch_width
        y2 = y + settings.patch_height

        left_patch = left_image[:, y:y2, x:x2]
        disparity_patch = disparity_image[y:y2, x:x2]
        if np.sum(disparity_patch) == 0:
            print("[", idx, "] x: ", x, " y: ", y, " has zero disparity")
        disp_13x13_patch = torch.unsqueeze(torch.from_numpy(np.array(disp_13x13_image[y:y2, x:x2], dtype=np.float32)), 0)
        disp_37x37_patch = torch.unsqueeze(torch.from_numpy(np.array(disp_37x37_image[y:y2, x:x2], dtype=np.float32)), 0)

        tran = transforms.ToTensor()
        input_patch = torch.cat([left_patch, disp_13x13_patch, disp_37x37_patch], dim=0)

        labels = disparity_patch

        return input_patch, labels

    def __getitem__(self, index):
        """Generates one sample of data to itterate training on.  __getitem__
           can be called by using the index of the assigend variable.  I.e.
           dataset = SiameseDataset(self, settings, patch_locations)
           dataset[0] will call __getitem__ with an index of 0
        Args:
          index (int): the index into the array of patch sets.

        Returns:
          input_patch (Tensor): sub-image input patch
          labels (Tensor): the array used as "ground truth"
        """

        # Loading the image
        if index > self.length - 1:
            print("Index is too large : ", index, "Dataset length : ", self.length)
        sample_info = np.zeros((3,), dtype=int)
        # Convert location information from floats into ints
        sample_info[0] = int(self.patch_locations['valid_locations'][index][0])
        sample_info[1] = int(self.patch_locations['valid_locations'][index][1])
        sample_info[2] = int(self.patch_locations['valid_locations'][index][2])
        input_patch, labels = self._pytorch_parse_function(sample_info)

        # Apply image transformations (not currently used)
        if self.transform is not None:
            input_patch = self.transform(left_patch)
        return input_patch, torch.from_numpy(np.array(labels, dtype=np.float32))

    def get_concat_image_w_disparity(self, idx):
        """
        """

        left_image = self.left_images[idx]
        disparity_image = self.disparity_images[idx]
        disp_13x13_image = self.disp_13x13_images[idx]
        disp_37x37_image = self.disp_37x37_images[idx]

        disp_13x13_tensor = torch.from_numpy(np.array(disp_13x13_image, dtype=np.float32))
        disp_37x37_tensor = torch.from_numpy(np.array(disp_37x37_image, dtype=np.float32))

        concat_image = torch.cat([left_image, disp_13x13_tensor, disp_37x37_tensor], dim=0)

        labels = disparity_patch

        return concat_image, labels


##########################################################################
# Models
##########################################################################

class FilterNet(nn.Module):

    def build_block(self, in_channels, out_channels, kernel_size, padding, upsample=False, relu=True):
        block = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                      bias=True),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),

        ]
        if upsample:
            block.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        return nn.Sequential(*block)

    def __init__(self):
        super().__init__()

        self.down1 = self.build_block(in_channels=5, out_channels=128, kernel_size=3, padding=1)
        self.down2 = self.build_block(in_channels=128, out_channels=32, kernel_size=1, padding=0)
        self.down3 = self.build_block(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.down4 = self.build_block(in_channels=32, out_channels=128, kernel_size=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.down5 = self.build_block(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.down6 = self.build_block(in_channels=128, out_channels=32, kernel_size=1, padding=0)
        self.down7 = self.build_block(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.down8 = self.build_block(in_channels=32, out_channels=128, kernel_size=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.down9 = self.build_block(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.down10 = self.build_block(in_channels=128, out_channels=32, kernel_size=1, padding=0)
        self.down11 = self.build_block(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.down12 = self.build_block(in_channels=32, out_channels=128, kernel_size=1, padding=0, upsample=True)

        self.up1 = self.build_block(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.up2 = self.build_block(in_channels=256, out_channels=64, kernel_size=1, padding=0)
        self.up3 = self.build_block(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.up4 = self.build_block(in_channels=64, out_channels=256, kernel_size=1, padding=0, upsample=True)

        self.up5 = self.build_block(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.up6 = self.build_block(in_channels=256, out_channels=64, kernel_size=1, padding=0)
        self.up7 = self.build_block(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.up8 = self.build_block(in_channels=64, out_channels=256, kernel_size=1, padding=0)
        self.final_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        c1 = self.down4(x)
        x = self.pool1(c1)
        x = self.down5(x)
        x = self.down6(x)
        x = self.down7(x)
        c2 = self.down8(x)
        x = self.pool2(c2)
        x = self.down9(x)
        x = self.down10(x)
        x = self.down11(x)
        x = self.down12(x)

        x = torch.cat([x, c2], dim=1)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = torch.cat([x, c1], dim=1)
        x = self.up5(x)
        x = self.up6(x)
        x = self.up7(x)
        x = self.up8(x)
        x = self.final_conv(x)
        return x


class UNet(nn.Module):

    def build_block(self, in_channels, out_channels, kernel_size, padding, upsample=False, relu=True):
        block = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                      bias=True),
            #nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        ]
        if upsample:
            block.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        return nn.Sequential(*block)

    def __init__(self):
        super().__init__()

        self.down1 = self.build_block(in_channels=1,   out_channels=64, kernel_size=3, padding=0)
        self.down2 = self.build_block(in_channels=64,  out_channels=64, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.down3 = self.build_block(in_channels=64,  out_channels=128, kernel_size=3, padding=0)
        self.down4 = self.build_block(in_channels=128, out_channels=128, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.down5 = self.build_block(in_channels=128, out_channels=256, kernel_size=3, padding=0)
        self.down6 = self.build_block(in_channels=256, out_channels=256, kernel_size=3, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.down7 = self.build_block(in_channels=256, out_channels=512, kernel_size=3, padding=0)
        self.down8 = self.build_block(in_channels=512, out_channels=512, kernel_size=3, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.down9  = self.build_block(in_channels=512,  out_channels=1024, kernel_size=3, padding=0)
        self.down10 = self.build_block(in_channels=1024, out_channels=512,  kernel_size=3, padding=0, upsample=True)

        self.up1 = self.build_block(in_channels=1024, out_channels=512, kernel_size=3, padding=0)
        self.up2 = self.build_block(in_channels=512,  out_channels=256, kernel_size=3, padding=0, upsample=True)

        self.up3 = self.build_block(in_channels=512, out_channels=256, kernel_size=3, padding=0)
        self.up4 = self.build_block(in_channels=256, out_channels=128, kernel_size=3, padding=0, upsample=True)

        self.up5 = self.build_block(in_channels=256, out_channels=128, kernel_size=3, padding=0)
        self.up6 = self.build_block(in_channels=128, out_channels=64,  kernel_size=3, padding=0, upsample=True)

        self.up7 = self.build_block(in_channels=128, out_channels=64,  kernel_size=3, padding=0)
        self.up8 = self.build_block(in_channels=64,  out_channels=64,  kernel_size=3, padding=0)

        self.final_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0, bias=True)

    def forward(self, x):

        x  = self.down1(x)
        c1 = self.down2(x)
        x  = self.pool1(c1)

        x  = self.down3(x)
        c2 = self.down4(x)
        x  = self.pool2(c2)

        x  = self.down5(x)
        c3 = self.down6(x)
        x  = self.pool3(c3)

        x  = self.down7(x)
        c4 = self.down8(x)
        x  = self.pool4(c4)

        x  = self.down9(x)
        x  = self.down10(x)

        x  = torch.cat([x, c4[:,:,4:-4,4:-4]], dim=1)
        x  = self.up1(x)
        x  = self.up2(x)

        x  = torch.cat([x, c3[:,:,16:-16,16:-16]], dim=1)
        x  = self.up3(x)
        x  = self.up4(x)

        x  = torch.cat([x, c2[:,:,40:-40,40:-40]], dim=1)
        x  = self.up5(x)
        x  = self.up6(x)

        x  = torch.cat([x, c1[:,:,88:-88,88:-88]], dim=1)
        x  = self.up7(x)
        x  = self.up8(x)

        x = self.final_conv(x)
        return x

    def output_size(self, input_size):
        '''  Calculate output image size based upon input size and network configuration.
        #FIXME - constants should be moved to __init__ and should be parameters which
        configure the shape of the UNet for hyperparameter search.
        '''
        kernel_size = 3
        cnns_per_stage = 2
        depth = 4
        scale_factor_per_stage = 2

        f = (kernel_size - 1) * cnns_per_stage * depth
        y, x = input_size
        for _ in range(depth-1):
            x = (x - (kernel_size - 1) * cnns_per_stage) / scale_factor_per_stage
            y = (y - (kernel_size - 1) * cnns_per_stage) / scale_factor_per_stage
            f = (f + (kernel_size - 1) * cnns_per_stage) * scale_factor_per_stage
        if x % 2 == 1:
            x_smaller = int(x)
            x_larger = x_smaller + 1
            for _ in range(depth-1):
                x_smaller = (x_smaller + (kernel_size - 1) * cnns_per_stage) * scale_factor_per_stage
                x_larger = (x_larger + (kernel_size - 1) * cnns_per_stage) * scale_factor_per_stage
            print("Invalid Input width, valid sizes are ", x_smaller, "and", x_larger)
        if y % 2 == 1:
            y_smaller = int(y)
            y_larger = y_smaller + 1
            for _ in range(depth-1):
                y_smaller = (y_smaller + (kernel_size - 1) * cnns_per_stage) * scale_factor_per_stage
                y_larger = (y_larger + (kernel_size - 1) * cnns_per_stage) * scale_factor_per_stage
            print("Invalid Input width, valid sizes are ", y_smaller, "and", y_larger)
        y, x = input_size

        return (y-f, x-f)

##########################################################################
# Loss Function
##########################################################################

class ImageDeltaLoss(torch.nn.Module):
    """
       Calculats the MSE loss between the predicted disparity over the entire
       output volume against the corresponding ground truth.
    """

    def __init__(self):
        super(ImageDeltaLoss, self).__init__()

    def forward(self, output_volume, input_volume, labels):
        """Calculate the loss describe above.

        Args:
          output_volume (Tensor): output of from filter network
          input_volume (Tensor): the input volume fed to the network, used as
                                 a feed forward term in some experiments.
          lables (Tensor): the "ground truth" of the expected disparity


        Returns:
          Loss (scaler)
        """

        output_disparity = torch.squeeze(output_volume[:, -1, :, :])
        net_disparity = output_disparity
        loss = F.mse_loss(net_disparity, labels[:,92:-92,92:-92])

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

def train(model):
    counter = []
    loss_history = []
    lr = settings.learning_rate

    print("Number of iterrations : ", settings.num_iterations)
    for epoch in range(settings.num_iterations):
        # print ("Epoch : ", epoch)
        for i, data in enumerate(train_dataloader, 0):
            input_volume, labels = data
            input_volume = torch.unsqueeze(input_volume[:,-1,:,:], 1)
            if i % 9999999 == 0:
                net_disparity = torch.squeeze(input_volume[:, -1, :, :])
                masked_net_disparity = net_disparity * torch.ceil(labels / 256).float()
                print("Input Disparity : ", torch.sum(torch.squeeze(masked_net_disparity[0, :, :])))
                print("GT    Disparity : ", torch.sum(torch.squeeze(labels[0, :, :])))
                disparity_error = torch.abs(masked_net_disparity - labels)
                three_pixel_disparity = F.threshold(disparity_error, 3, 0)
                print("Input Error : ", torch.sum(three_pixel_disparity))

            input_volume, labels = input_volume.cuda(), labels.cuda()
            optimizer.zero_grad()
            output_volume = model(input_volume)
            loss_image_delta = criterion(output_volume, input_volume, labels)
            loss_image_delta.backward()
            optimizer.step()
            if i % 100 == 0:
                # Note validation set must be >= %1 of training set for iterator to not break when it runs out of validation data
                print("{}, Epoch: {}, Batch: {}, Learning Rate: {}, Training loss: {}, Validation loss: {}".format(
                    datetime.datetime.now(tz=pytz.utc), epoch, i, lr, loss_image_delta.item(),
                    'placeholder'))
                loss_history.append(loss_image_delta.item())
                lr = adjust_learning_rate(optimizer, int(i / 100), int(num_batches / 100))
            if i == settings.max_batches:
                break
    return model


##########################################################################
# Post Processing Functions
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
    # untrimmed_prediction = disparity_prediction.detach().numpy()
    masked_prediction_valid = disparity_prediction * valid_gt_pixels
    num_valid_gt_pixels = valid_gt_pixels.sum()

    # Use 3-pixel error metric for now.
    num_error_pixels = (np.abs(masked_prediction_valid - disparity_ground_truth) > 3).sum()
    error += num_error_pixels / num_valid_gt_pixels

    print('{}, Error: {:04f}, for image index {}'.format(datetime.datetime.now(tz=pytz.utc), error, idx))

    return error


##########################################################################
# Main
##########################################################################

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
        print("New patch file being generatd")
        find_and_store_patch_locations(settings)
    with open(settings.patch_locations_path, 'rb') as handle:
        print("Loading existing patch file")
        patch_locations = pickle.load(handle)
else:
    print("Patch file already loaded")

if not torch.cuda.is_available():
    print("GPU not available!")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
# Declare Network for training
model = UNet()
# Calculate output dimentions before moving structure to GPU
output_height, output_width = model.output_size((settings.patch_height, settings.patch_width))
# Create Model Instance for
model = model.to(device)
# Print out Model Summary
if settings.tensor_summary == True:
    torchsummary.summary(model, input_size=(5, 48, 48))
    sys.stdout.flush()  # flush torchsummary.summary output

if settings.phase == 'training' or settings.phase == 'both':
    training_dataset = DisparityDataset(settings, patch_locations['train'])
    print("Loading training dataset")
    train_dataloader = DataLoader(training_dataset,
                                  shuffle=True,
                                  num_workers=1,
                                  batch_size=settings.batch_size)

    val_dataset = DisparityDataset(settings, patch_locations['val'])

    # print("Batch Size : ", settings.batch_size)
    print("Loading validation dataset")
    val_dataloader = DataLoader(val_dataset,
                                shuffle=True,
                                num_workers=1,
                                batch_size=settings.batch_size)

    val_dataset_iterator = iter(val_dataloader)

    num_batches = len(train_dataloader)
    print("Number of ", settings.batch_size, "patch batches", num_batches)

    # Decalre Loss Function
    criterion = ImageDeltaLoss()
    # Declare Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)

    print("Start Training")
    # Train the model
    train(model)
    torch.save(model.state_dict(), settings.model_path)
    print("Model Saved Successfully")

if settings.phase == 'testing' or settings.phase == 'both':
    print("Start Testing")
    # Load the saved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = UNet().to(get_device)
    model.load_state_dict(torch.load(settings.model_path))
    model.eval()  # required for batch normalization to function correctly

    with open(settings.patch_locations_path, 'rb') as handle:
        patch_locations = pickle.load(handle)
    test_image_indices = patch_locations['val']['ids']
    train_image_indices = patch_locations['train']['ids']

    counter = 0
    left_image_paths, disparity_paths, dmask_paths, disp_13x13_paths, disp_37x37_paths = \
        load_image_paths(settings.data_path,
                         settings.left_img_folder,
                         settings.disparity_folder,
                         settings.dmask_folder,
                         settings.disp_13x13_folder,
                         settings.disp_37x37_folder)


    error_dict = {}
    print("Testing Center")
    for idx in test_image_indices:
        left_image, input_volume, disparity_ground_truth = get_concat_image_w_disparity(settings, idx, blended=False)
        input_volume = torch.unsqueeze(input_volume[-1,:,:], 0)

        upper_left_y = settings.img_height - settings.patch_height
        lower_right_y = upper_left_y + settings.patch_height
        upper_left_x = int((settings.img_width - settings.patch_width) / 2)
        lower_right_x = upper_left_x + settings.patch_width
        input_volume = input_volume[:,upper_left_y:lower_right_y, upper_left_x:lower_right_x]

        upper_left_y = upper_left_y + int((settings.patch_height - output_height) / 2)
        lower_right_y = upper_left_y + output_height
        upper_left_x = upper_left_x + int((settings.patch_width - output_width) / 2)
        lower_right_x = upper_left_x + output_width
        disparity_ground_truth = disparity_ground_truth[upper_left_y:lower_right_y, upper_left_x:lower_right_x]

        input_volume = input_volume.cuda()
        input_volume = torch.unsqueeze(input_volume, 0)
        output_volume = model(input_volume.to(device))
        disparity_prediction = torch.squeeze(output_volume)
        disparity_prediction = disparity_prediction.cpu().detach().numpy()

        error_dict[idx] = calc_error(disparity_prediction, disparity_ground_truth, idx)
        disp_image = prediction_to_image(disparity_prediction)
        save_images([left_image.permute(1, 2, 0), disp_image], 1, ['left image', 'disparity'], settings.image_dir,
                    'disparity_{}.png'.format(idx))
        cv2.imwrite(join(settings.image_dir, (("00000" + str(idx))[-6:] + "_10.png")), disparity_prediction)

    average_error = 0.0
    for idx in error_dict:
        average_error += error_dict[idx] / len(error_dict)
    print("Average Test Error : ", average_error)


    if settings.test_all:
        for idx in train_image_indices:
            left_image, input_volume, disparity_ground_truth = get_concat_image_w_disparity(settings, idx, blended=False)
            input_volume = torch.unsqueeze(input_volume[-1,:,:], 0)

            upper_left_y = settings.img_height - settings.patch_height
            lower_right_y = upper_left_y + settings.patch_height
            upper_left_x = int((settings.img_width - settings.patch_width) / 2)
            lower_right_x = upper_left_x + settings.patch_width
            input_volume = input_volume[:,upper_left_y:lower_right_y, upper_left_x:lower_right_x]

            output_patch_size = 132
            upper_left_y = upper_left_y + int((settings.patch_width - output_patch_size) / 2)
            lower_right_y = upper_left_y + output_patch_size
            upper_left_x = upper_left_x + int((settings.patch_width - output_patch_size) / 2)
            lower_right_x = upper_left_x + output_patch_size
            disparity_ground_truth = disparity_ground_truth[upper_left_y:lower_right_y, upper_left_x:lower_right_x]

            input_volume = input_volume.cuda()
            input_volume = torch.unsqueeze(input_volume, 0)
            output_volume = model(input_volume.to(device))
            output_disparity = torch.squeeze(output_volume)
            input_disparity = torch.squeeze(input_volume[:, -1, :, :])
            disparity_prediction = output_disparity #.add(input_disparity[92:-92,92:-92])

            disparity_prediction = disparity_prediction.cpu().detach().numpy()
            error_dict[idx] = calc_error(disparity_prediction, disparity_ground_truth, idx)

            disp_image = prediction_to_image(disparity_prediction)
            save_images([left_image.permute(1, 2, 0), disp_image], 1, ['left image', 'disparity'], settings.image_dir,
                    'disparity_{}.png'.format(idx))
            cv2.imwrite(join(settings.image_dir, (("00000"+str(idx))[-6:]+"_10.png")), disparity_prediction)


    average_error = 0.0
    for idx in error_dict:
        average_error += error_dict[idx] / len(error_dict)
    print("Average Error : ", average_error)

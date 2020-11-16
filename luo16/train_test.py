'''  PyTorch implementation of Efficient Deep Learning for Stereo Matching by  Wenjie Luo, Alexander G. Schwing, & Raquel Urtasun
'''

import os
import sys
import stat
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

# Project code imports
from util.img_utils import *
from dataset.siamese_dataset import SiameseDataset
from net.siamese_network import SiameseNetwork
from net.siamese_network_13 import SiameseNetwork13
from loss.inner_product_loss import InnerProductLoss

##########################################################################
# Utilities
##########################################################################

LOG_FORMAT = '%(asctime)-15s %(levelname)-5s %(name)-15s - %(message)s'


def setup_logging(log_path=None, log_level='DEBUG', logger=None, fmt=LOG_FORMAT):
    """Prepare logging for the provided logger.

    Args:
        log_path (str, optional): full path to the desired log file
        debug (bool, optional): log in verbose mode or not
        logger (logging.Logger, optional): logger to setup logging upon,
            if it's None, root logger will be used
        fmt (str, optional): format for the logging message

    Returns:
        logger: logging object
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
        logger.info('Log file is ' + str(log_path))

    return logger
# /setup_logging()

##########################################################################
# Input Pre Procesing
##########################################################################

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

    # logger.info("Number of images : {}".format(len(sample_ids)))
    for i, idx in enumerate(sample_ids):
        disp_img = np.array(Image.open(disparity_image_paths[idx])).astype('float64')
        # NOTE: We want images of same size for efficient loading.
        disp_img = trim_image(disp_img, img_height, img_width)
        disp_img /= 256
        max_disp_img = np.max(disp_img)
        num_valid_locations[i] = (disp_img != 0).sum()
        # logger.info("{} Max disp: {} Num valid locations : {}".format(disparity_image_paths[idx], max_disp_img, num_valid_locations[i]))

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
    logger.info("Total Number of valid locations: {}".format(valid_count))
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
    # make read-only to prevent clobber
    os.chmod(settings.patch_locations_path, stat.S_IREAD)


##########################################################################
# Adaptive Learning Rate
##########################################################################

def adjust_learning_rate(settings, optimizer, step, num_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    step_size = settings.reduction_factor ** (1.0 / num_steps)
    lr = settings.learning_rate * ((1.0 / step_size) ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


##########################################################################
# Training
##########################################################################

def train(settings, num_batches, train_dataloader, optimizer, net, criterion, val_dataset_iterator, tensorboard_writer):
    counter = []
    loss_history = []
    lr = settings.learning_rate

    logger.info("Number of iterations : {}".format(settings.num_iterations))
    cum_batch_counter = 0
    for epoch in range(settings.num_iterations):
        # logger.info("Epoch : {}".format(epoch))
        for i, data in enumerate(train_dataloader, 0):
            # logger.info("Batch : {}".format(i))
            left_patch, right_patch, labels = data
            # logger.info("Data Size : {}".format(left_patch.size()))
            left_patch, right_patch, labels = left_patch.cuda(), right_patch.cuda(), labels.cuda()
            optimizer.zero_grad()
            left_feature, right_feature = net(left_patch, right_patch)
            loss_inner_product = criterion(left_feature, right_feature, labels)
            loss_inner_product.backward()
            optimizer.step()
            cum_batch_counter += 1
            tensorboard_writer.add_scalar('training_loss', loss_inner_product.item(), global_step=cum_batch_counter)
            tensorboard_writer.add_scalar('learning_rate', lr)
            if i % 1000 == 0:
                # Note validation set must be >= %1 of training set for iterator to not break when it runs out of validation data
                left_patch, right_patch, labels = next(val_dataset_iterator)
                left_patch, right_patch, labels = left_patch.cuda(), right_patch.cuda(), labels.cuda()
                optimizer.zero_grad()
                left_feature, right_feature = net(left_patch, right_patch)
                val_loss_inner_product = criterion(left_feature, right_feature, labels)
                tensorboard_writer.add_scalar('validation_loss', val_loss_inner_product.item(), cum_batch_counter)
                logger.info("{}, Epoch: {}, Batch: {}, Learning Rate: {}, Training loss: {}, Validation loss: {}".format(
                    datetime.datetime.now(tz=pytz.utc), epoch, i, lr, loss_inner_product.item(),
                    val_loss_inner_product.item()))
                loss_history.append(loss_inner_product.item())
                lr = adjust_learning_rate(settings, optimizer, int(i / 100), int(num_batches / 100))
            if i == settings.max_batches:
                break
    return net


##########################################################################
# Inference Functions
##########################################################################

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

    logger.info('Avg 3-pix error metric = {:04f}, for image index {}'.format(error, idx))

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
    # logger.info("Cost Volume Shape : {}".format(cost_volume.size()))
    cost_volume.permute(3, 0, 1, 2)  # barrel shift right
    cost_volume = F.pad(cost_volume, twoDimensionPad, "reflect", 0)
    cost_volume.permute(1, 2, 3, 0)  # back to original order

    return F.avg_pool2d(cost_volume, kernel_size=5, stride=1)


def calc_cost_volume(settings, left_features, right_features, mask=None):
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
    img_height, img_width = right_features.shape[2], right_features.shape[3]
    # logger.info("right feature shape : {}".format(right_feature.size()))
    row_indices = torch.arange(0, img_width, dtype=torch.int64)

    for i in range(img_width):
        # logger.info("left feature shape before squeeze : {}".format(left_feature.size()))
        left_column_features = torch.squeeze(left_features[:, :, :, i])
        # logger.info("left feature shape after squeeze  : {}".format(left_column_features.size()))
        start_win = max(0, i - settings.half_range)
        end_win = max(settings.disparity_range, settings.half_range + i + 1)
        start_win = start_win - max(0, end_win - img_width)  # was img_width.value
        end_win = min(img_width, end_win)

        right_win_features = torch.squeeze(right_features[:, :, :, start_win:end_win])
        win_indices_column = torch.unsqueeze(row_indices[start_win:end_win], 0)
        # logger.info("left feature shape      : {}".format(left_column_features.size()))
        # logger.info("right win feature shape : {}".format(right_win_features.size()))
        inner_product_column = torch.einsum('ij,ijk->jk', left_column_features,
                                            right_win_features)
        inner_product_column = torch.unsqueeze(inner_product_column, 1)
        inner_product.append(inner_product_column)
        win_indices.append(win_indices_column)
    # logger.info("inner_product len   : {}".format(len(inner_product)))
    # logger.info("inner_product width : {}".format(len(inner_product[0])))
    # logger.info("win_indices len   : {}".format(len(win_indices)))
    # logger.info("win_indices width : {}".format(len(win_indices[0])))
    inner_product = torch.unsqueeze(torch.cat(inner_product, 1), 0)
    # logger.info("inner_product after unsqueeze  : {}".format(inner_product.size()))
    win_indices = torch.cat(win_indices, 0).to(device)
    # logger.info("win_indices shape : {}".format(win_indices.size()))
    return inner_product, win_indices


def inference(settings, left_features, right_features, post_process):
    """Post process model output.

    Args:
        left_features (Tensor): left input cost volume.
        right_features (Tensor): right input cost volume.

    Returns:
        disp_prediction (Tensor): disparity prediction.

    """
    cost_volume, win_indices = calc_cost_volume(settings, left_features, right_features)
    # logger.info("Cost Volume Shape : {}".format(cost_volume.size()))
    img_height, img_width = cost_volume.shape[1], cost_volume.shape[2]  # Now 1 x C X H x W, was 1 x H x W x C
    if post_process:
        cost_volume = apply_cost_aggregation(cost_volume)
    cost_volume = torch.squeeze(cost_volume)
    # logger.info("Cost Volume Shape (post squeeze): {}".format(cost_volume.size()))
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

    # logger.info("disp_prediction length (pre cat) : ", len(disp_prediction))
    disp_prediction = torch.cat(disp_prediction, 1)
    # logger.info("disp_prediction shape (post cat): ", disp_prediction.size())
    # logger.info("row indices                     : ", row_indices.size())
    disp_prediction = row_indices.permute(1, 0).to(device) - disp_prediction

    return disp_prediction

def setup_filesystem(settings):
    os.makedirs(settings.exp_path, exist_ok=True)
    os.makedirs(settings.image_dir, exist_ok=True)
    settings_filename = os.path.join(settings.exp_path, 'settings.log')
    with open(settings_filename, 'w') as settings_file:
        settings_file.write(str(settings))

# /setup_filesystem()

def process_cmdline_args():
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
        parser.add_argument('--result-dir', default='/cs230-datasets/proj/results', type=str, help='results directory')
        parser.add_argument('--experiments-dir', default='experiments', type=str, help='experiments directory')
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
    # /if-else

    # Settings, hyper-parameters.
    settings.data_path = os.path.join(settings.data_path, 'training')
    setattr(settings, 'exp_path', os.path.join(settings.result_dir, settings.exp_name))
    setattr(settings, 'img_height', 370)
    setattr(settings, 'img_width', 1224)
    setattr(settings, 'half_patch_size', (settings.patch_size // 2))
    setattr(settings, 'half_range', settings.disparity_range // 2)
    setattr(settings, 'num_train', 160)
    setattr(settings, 'model_name', os.path.join('model_', str(settings.patch_size)))

    setattr(settings, 'left_img_folder', 'image_2')
    setattr(settings, 'right_img_folder', 'image_3')
    setattr(settings, 'disparity_folder', 'disp_noc_0')
    setattr(settings, 'num_val', 40)
    setattr(settings, 'num_input_channels', 3)
    setattr(settings, 'image_dir', os.path.join(settings.exp_path, 'images'))
    setattr(settings, 'patch_locations_path', os.path.join(settings.exp_path, 'patch_locations.pkl'))
    setattr(settings, 'model_path', os.path.join(settings.exp_path, 'model.pt'))

    return settings
# /process_cmdline_args()

##########################################################################
# Main
##########################################################################

device = 'unknown'
logger = None

def main():

    settings = process_cmdline_args()    
    setup_filesystem(settings)

    # Python logging.
    LOGGER = logging.getLogger(__name__)
    log_file = os.path.join(settings.exp_path, 'log.log')
    global logger
    logger = setup_logging(log_path=log_file, log_level=settings.log_level, logger=LOGGER)
    #- logger.info('PYTHONPATH = ' + str(sys.path))
    random.seed(settings.seed)

    patch_locations_loaded = 'patch_locations' in locals() or 'patch_locations' in globals()
    if not (patch_locations_loaded) or patch_locations == None:
        if not os.path.exists(settings.patch_locations_path):
            logger.info("New patch file being generated")
            find_and_store_patch_locations(settings)
        with open(settings.patch_locations_path, 'rb') as handle:
            logger.info("Loading existing patch file " + settings.patch_locations_path)
            patch_locations = pickle.load(handle)
    else:
        logger.info("Patch file already loaded")
    '''
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    '''
    # Assign GPU
    if not torch.cuda.is_available():
        logger.warning("GPU not available!")
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device {}'.format(device))
    torch.backends.cudnn.benchmark = True

    # torchsummary, below, writes to stdout so redirect
    stdout_original = sys.stdout
    sys.stdout = open(os.path.join(settings.exp_path, 'model.log'), 'w')
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
    sys.stdout = stdout_original

    tensorboard_writer =\
        SummaryWriter(log_dir=os.path.join(settings.result_dir, 'tensorboard', settings.exp_name))

    # ----- TRAINING -----

    if settings.phase == 'training' or settings.phase == 'both':
        training_dataset = SiameseDataset(settings, patch_locations['train'])
        # logger.info("Batch Size : ", settings.batch_size)
        logger.info('training_dataset size = {}'.format(len(training_dataset)))
        logger.info("Loading training dataset")
        train_dataloader = DataLoader(training_dataset,
                                      shuffle=True,
                                      num_workers=8,
                                      batch_size=settings.batch_size)

        val_dataset = SiameseDataset(settings, patch_locations['val'])
        logger.info('training_dataset size = {}'.format(len(training_dataset)))
        # logger.info("Batch Size : ", settings.batch_size)
        logger.info("Loading validation dataset")
        val_dataloader = DataLoader(val_dataset,
                                    shuffle=True,
                                    num_workers=2,
                                    batch_size=settings.batch_size)

        val_dataset_iterator = iter(val_dataloader)

        num_batches = len(train_dataloader)
        logger.info("Number of {} patch batches {}".format(settings.batch_size, num_batches))

        # Decalre Loss Function
        criterion = InnerProductLoss()
        # Declare Optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=settings.learning_rate)

        logger.info("Start Training")
        # Train the model
        model = train(settings, num_batches, train_dataloader, optimizer, net, criterion, val_dataset_iterator, tensorboard_writer)
        torch.save(model.state_dict(), settings.model_path)
        logger.info("Model Saved Successfully")
    # /if 

    # ----- TESTING -----

    if settings.phase == 'testing' or settings.phase == 'both':
        logger.info("Start Testing")
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
            left_image_in = load_image(left_image_paths[idx], settings.img_height, settings.img_width)
            right_image = load_image(right_image_paths[idx], settings.img_height, settings.img_width)
            disparity_ground_truth = load_disparity(disparity_image_paths[idx], settings.img_height, settings.img_width)

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
            # logger.info("Left image size  : {}".format(left_image.size()))
            # logger.info("Right image size : {}".format(right_image.size()))
            # left_feature, right_feature = model(left_image.to(device), right_image.to(device))
            left_feature, right_feature = model(left_image.to(device), right_image.to(device))
            # logger.info("Left feature size  : {}".format(left_feature.size()))
            # logger.info("Right feature size : {}".format(right_feature.size()))
            # logger.info("Left Feature on Cuda: {}".format(left_feature.get_device()))
            disp_prediction = inference(settings, left_feature, right_feature, post_process=True)
            error_dict[idx] = calc_error(disp_prediction.cpu(), disparity_ground_truth, idx)
            disp_image = prediction_to_image(disp_prediction.cpu())
            save_images([left_image_in.permute(1, 2, 0), disp_image], 1, ['left image', 'disparity'], settings.image_dir,
                        'disparity_{}.png'.format(idx))
            cv2.imwrite(os.path.join(settings.image_dir, (("00000" + str(idx))[-6:] + "_10.png")), np.array(disp_prediction.cpu()))
        # /for idx

        if settings.test_all:
            for idx in train_image_indices:
                left_image_in = load_image(left_image_paths[idx], settings.img_height, settings.img_width)
                right_image = load_image(right_image_paths[idx], settings.img_height, settings.img_width)
                disparity_ground_truth = load_disparity(disparity_image_paths[idx], settings.img_height,
                                                        settings.img_width)
                twoDimensionPad = (
                settings.half_patch_size, settings.half_patch_size, settings.half_patch_size, settings.half_patch_size)
                left_image = F.pad(left_image_in, twoDimensionPad, "constant", 0)
                right_image = F.pad(right_image, twoDimensionPad, "constant", 0)
                left_image = torch.unsqueeze(left_image, 0)
                right_image = torch.unsqueeze(right_image, 0)
                left_feature, right_feature = model(left_image.to(device), right_image.to(device))
                disp_prediction = inference(settings, left_feature, right_feature, post_process=True)
                error_dict[idx] = calc_error(disp_prediction.cpu(), disparity_ground_truth, idx)
                disp_image = prediction_to_image(disp_prediction.cpu())
                save_images([left_image_in.permute(1, 2, 0), disp_image], 1, ['left image', 'disparity'],
                            settings.image_dir,
                            'disparity_{}.png'.format(idx))
                cv2.imwrite(os.path.join(settings.image_dir, (("00000" + str(idx))[-6:] + "_10.png")),
                            np.array(disp_prediction.cpu()))
            # /for idx
        # /if settings.test_all

        average_error = 0.0
        for idx in error_dict:
            average_error += error_dict[idx] / len(error_dict)
        logger.info("Average Error : {}".format(average_error))
        tensorboard_writer.add_scalar('average_disparity_error', average_error, global_step=idx)

    # Freeze results (for experiment) dir to prevent accidental clobber
    os.chmod(settings.exp_path, stat.S_IREAD)

# /main()

if __name__ == "__main__":
    main()

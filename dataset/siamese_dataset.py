import torch
from torch.utils.data import Dataset
from util.img_utils import *

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
            load_images(left_image_paths,
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

        labels = get_labels(self._settings.disparity_range, self._settings.half_range)

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

# /class SiameseDataset

import numpy as np

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

# /trim_image()


def load_image_paths(data_path, left_img_folder, right_img_folder, disparity_folder):
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

# /load_image_paths()

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

# /_load_image()


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

# /_load_disparity()


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

    print('_load_images: Loaded', len(left_images), 'images')
    return (left_images, right_images, np.array(disparity_images))

# /_load_images()

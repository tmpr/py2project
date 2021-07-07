import matplotlib.pyplot as plt
import torch
from random import choice

from typing import Tuple

import numpy as np


def visualize_input_tensor(input_tensor):
    rows, cols, axes, = 2, 4, []
    figs = input_tensor.shape[0]
    fig = plt.figure(figsize=(20, 6))

    for a in range(figs):
        b = input_tensor[a]
        axes.append(fig.add_subplot(rows, cols, a+1))
        subplot_title = ("Layer " + str(a))
        axes[-1].set_title(subplot_title)
        plt.imshow(b)
    plt.show()


def visualize_results(batch, batch_out):
    for input_tensor, out in zip(batch, batch_out):
        fig = plt.figure(figsize=(20, 15))
        axes = []

        axes.append(fig.add_subplot(1, 4, 1))
        axes[-1].set_title("First layer of input")
        axes[-1].axes.xaxis.set_visible(False)
        axes[-1].axes.yaxis.set_visible(False)
        plt.imshow(input_tensor[0][0])

        axes.append(fig.add_subplot(1, 4, 2))
        axes[-1].set_title("Mask")
        axes[-1].axes.xaxis.set_visible(False)
        axes[-1].axes.yaxis.set_visible(False)
        plt.imshow(input_tensor[0][1])

        axes.append(fig.add_subplot(1, 4, 3))
        axes[-1].set_title("Reconstructed image")
        axes[-1].axes.xaxis.set_visible(False)
        axes[-1].axes.yaxis.set_visible(False)
        plt.imshow(input_tensor[0][0] +
                   out.squeeze().cpu().detach() * input_tensor[0][1])

        axes.append(fig.add_subplot(1, 4, 4))
        axes[-1].set_title("Actual network output")
        axes[-1].axes.xaxis.set_visible(False)
        axes[-1].axes.yaxis.set_visible(False)
        plt.imshow(out.squeeze().cpu().detach())

        plt.show()


def denormalize(tensor, mean, std):
    tensor *= std.cuda()
    tensor += mean.cuda()
    return tensor


def crop_mse(original, out, mask, mse):
    """Return mean squared error over cropped out error, 
    which is indicated by the mask.

    Args:
        original (torch.tensor): Original tensor which has not been cropped.
        out (torch.tensor): Prediction of a neural net.
        mask (torch.tensor): Tensor with cropped out part == 1, else == 0.
        mse (nn.MSELoss): Instantiated loss function.

    Returns:
        tensor: Calculated loss.
    """
    return mse(out * mask, original.cuda() * mask)


def disect(image_array: np.array,
           crop_size: Tuple[int, int],
           crop_center: Tuple[int, int],
           border_distance=True) -> tuple:
    """Crop part out of image.

    Args:
        image_array (np.array): Some 2D numpy array
        crop_size (Tuple[int, int]): Dimensions of to-be-cropped rectangle.
        crop_center (Tuple[int, int]): Center of to-be-cropped rectangle.

    Raises:
        ValueError: If rectangle is closer than 20 pixels from any margin 
                    and if crop-size or crop-center contain even numbers.

    Returns:
        tuple: Manipulated image, meta-tuple, target array.
    """

    image_array = image_array.copy()

    _discect_check_args(crop_size, crop_center, image_array)

    y,  x = crop_center
    dy, dx = crop_size
    i_height, i_width = image_array.shape

    left, right = x - dx//2, x + dx//2
    bottom, top = y - dy//2, y + dy//2

    if any(x <= 20 for x in (left, bottom, i_height - top, 
                            i_width - right)) and border_distance:
        raise ValueError('Cropped out area must be more ' +
                         'than 20 pixels away from border'
                         + str(image_array.shape))

    target_array = image_array[bottom:top + 1, left:right + 1].copy()
    crop_array = np.zeros_like(image_array)

    crop_array[bottom:top + 1, left:right + 1] = 1

    # Fill the cropped part with the mean of the non-cropped pixels right away.
    mean = np.mean([array_val for array_row, mask_row in zip(image_array, crop_array)
                    for array_val, mask_val in zip(array_row, mask_row)
                    if not mask_val])

    image_array[bottom:top + 1, left:right + 1] = mean

    return image_array, crop_array, target_array


def _discect_check_args(crop_size, crop_center, image_array):
    if not isinstance(image_array, np.ndarray):
        raise ValueError
    elif len(image_array.shape) != 2:
        raise ValueError('image_array not 2-dimensional.')
    elif len(crop_size) != 2:
        raise ValueError('crop_size does not have exactly 2 elements.')
    elif len(crop_center) != 2:
        raise ValueError('crop_center does not have exactly 2 elements.')
    elif not (crop_size[0] * crop_size[1]) % 2:
        raise ValueError('crop_size contains even elements.')


def crop_dimensions(height: int, width: int, border: int = 30):
    """Return crop dimesions that obilige to the rules of ex4."""
    try:
        center = (choice(range(border, width - border)),
                  choice(range(border, height - border)))
    except IndexError:
        return crop_dimensions(height, width, border - 1)

    x, y = center

    try:
        size = (choice(range(1, min(x - 19, width - 19 - x), 2)),
                choice(range(1, min(y - 19, height - 19 - y), 2))
                )
    except IndexError:
        size = (1, 1)

    return center, size


def custom_pad(array: np.array, size: int) -> np.array:
    """Pad array with zeros s.t. it becomes a square of size `size`."""
    square = np.zeros((size, size), dtype=array.dtype)
    square[:array.shape[0], :array.shape[1]] = array
    return square


def selu_weighted(module: torch.nn.Module) -> torch.nn.Module:
    """Initialize weights according to Selu theory:

    The weights distributed such that the mean = 0 and the variance
    = 1/`input_units`"""
    module.weight.data.normal_(0.0,
                               np.sqrt(1. / np.prod(module.weight.shape[1:])))
    return module

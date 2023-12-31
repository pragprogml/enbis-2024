import logging
import random

import cv2
import numpy as np

from src.libs.utils import IMAGE_H, IMAGE_W

# Set seed for reproducibility
np.random.seed(42)

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def resize_and_equalize(
    image: np.ndarray,
) -> np.ndarray:
    """
    Resizes and equalizes the histogram of an image.

    This function first resizes the image to a fixed size and then converts it to a grayscale image.
    Afterwards, it applies histogram equalization to enhance the contrast of the grayscale image.

    Parameters:
    image (numpy.ndarray): The input image in BGR format.

    Returns:
    numpy.ndarray: The equalized grayscale image.
    """
    resized_image = cv2.resize(image, (IMAGE_W, IMAGE_H), cv2.INTER_CUBIC)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    return equalized_image


def augment_image(
    image: np.ndarray,
    horizontal_distortion_limit: float,
    min_brightness: float,
    max_brightness: float,
    max_rotation_angle: float,
    zoom_range: float,
) -> np.ndarray:
    """
    Applies various augmentation techniques to an image including horizontal shift, brightness adjustment,
    rotation, and zoom.

    Parameters:
    image (numpy.ndarray): The input image to be augmented.
    horizontal_distortion_limit (float): The limit for random horizontal shift as a fraction of image width.
    min_brightness (float): The lower bound for random brightness adjustment factor.
    max_brightness (float): The upper bound for random brightness adjustment factor.
    max_rotation_angle (float): The maximum rotation angle in degrees for random rotation.
    zoom_range (float): The minimum fraction of image size to be taken for zoom effect.

    Returns:
    numpy.ndarray: The augmented image.
    """
    # Horizontal Shift
    shift_ratio = random.uniform(-horizontal_distortion_limit, horizontal_distortion_limit)
    height, width = image.shape[:2]
    shift_amount = width * shift_ratio
    if shift_ratio > 0:
        image = image[:, : int(width - shift_amount)]
    if shift_ratio < 0:
        image = image[:, int(-1 * shift_amount) :]  # noqa: E203

    image = cv2.resize(image, (width, height), cv2.INTER_CUBIC)

    # Brightness Adjustment
    brightness_value = random.uniform(min_brightness, max_brightness)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV)
    hsv_image = np.array(hsv_image, dtype=np.float64)
    hsv_image[:, :, 1] = hsv_image[:, :, 1] * brightness_value
    hsv_image[:, :, 1][hsv_image[:, :, 1] > 255] = 255
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * brightness_value
    hsv_image[:, :, 2][hsv_image[:, :, 2] > 255] = 255
    hsv_image = np.array(hsv_image, dtype=np.uint8)
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Rotation
    rotation_angle = int(random.uniform(-max_rotation_angle, max_rotation_angle))
    rotation_matrix = cv2.getRotationMatrix2D((int(width / 2), int(height / 2)), rotation_angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # Zoom
    zoom_factor = random.uniform(zoom_range, 1)
    height_taken = int(zoom_factor * height)
    width_taken = int(zoom_factor * width)
    height_start = random.randint(0, height - height_taken)
    width_start = random.randint(0, width - width_taken)
    image = image[height_start : height_start + height_taken, width_start : width_start + width_taken]  # noqa: E203
    image = cv2.resize(image, (width, height), cv2.INTER_CUBIC)

    # Prepare image
    equalized_image = resize_and_equalize(image)

    return equalized_image

Module libs.preprocessing
=========================

Functions
---------

`augment_image(image: numpy.ndarray, horizontal_distortion_limit: float, min_brightness: float, max_brightness: float, max_rotation_angle: float, zoom_range: float) ‑> numpy.ndarray`
:   Applies various augmentation techniques to an image including horizontal shift, brightness adjustment,
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

`resize_and_equalize(image: numpy.ndarray) ‑> numpy.ndarray`
:   Resizes and equalizes the histogram of an image.
    
    This function first resizes the image to a fixed size and then converts it to a grayscale image.
    Afterwards, it applies histogram equalization to enhance the contrast of the grayscale image.
    
    Parameters:
    image (numpy.ndarray): The input image in BGR format.
    
    Returns:
    numpy.ndarray: The equalized grayscale image.
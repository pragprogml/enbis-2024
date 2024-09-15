Module libs.utils
=================

Functions
---------

`calculate_intersection_over_union(prediction_box: Tuple[int, int, int, int], ground_truth_box: Tuple[int, int, int, int]) ‑> float`
:   Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
    prediction_box (Tuple[int, int, int, int]): The bounding box of the predicted object.
        It is defined by (x_min, y_min, x_max, y_max).
    ground_truth_box (Tuple[int, int, int, int]): The bounding box of the ground truth object.
        It is defined by (x_min, y_min, x_max, y_max).
    
    Returns:
    float: The IoU ratio as a float value. It is the area of overlap between the two bounding
        boxes divided by the area of union.

`merge_test_with_sampled_validation_images(root_path: str, config_path: str, num_sampled_images: int) ‑> list`
:   Samples a specified number of images from a dataset's validation set and combines them
    with all images from the test set.
    
    Parameters:
    root_path (str): The root directory path where the dataset is located.
    config_path (str): The path to the configuration file that contains paths for test and
                       validation datasets within the root directory.
    num_sampled_images (int): The number of images to sample from the validation set.
    
    Returns:
    list: A list of file paths, combining all test images and the sampled validation images.
    
    Raises:
    ValueError: If the number of images to sample is negative or exceeds the available images in the validation set.
    FileNotFoundError: If the configuration file, test, or validation directories are not found.

`perform_object_detection_and_plot_results(model, img_path: str) ‑> numpy.ndarray`
:   Performs object detection on an image using a specified model.
    
    Parameters:
    model: The object detection model to be used for detecting objects in the image.
           Expected to have a method that takes an image as input and returns detection results.
    img_path (str): Path to the image file on which object detection will be performed.
    
    Returns:
    detect_img: An image with detection results plotted. The image is in RGB format.
    
    Raises:
    FileNotFoundError: If the specified image file does not exist.
    Exception: For errors related to image loading, processing, or detection.

`predict(input_image: str, model_file_path: str) ‑> Tuple[float, float, float]`
:   Predicts objects in an image using the YOLO model.
    
    This function loads a YOLO model from a given file path, and then uses the model
    to predict objects in the provided image. It logs and returns the coordinates
    of the bounding box and confidence level of the first detected object.
    
    Parameters:
    input_image: The image in which objects are to be detected.
    model_file_path: Path to the file where the YOLO model is stored.
    
    Returns:
    A tuple (x_min, y_min, confidence) representing the coordinates of the top-left
    corner of the bounding box and the confidence level of the first detected object.
    Returns (0, 0, 0) if no objects are detected.

`set_device_config() ‑> str`
:   Sets the device configuration for training based on the availability of hardware accelerators.
    
    Args:
    config (dict): A configuration dictionary where the device configuration will be set.
    
    Returns:
    None: The function modifies the 'config' dictionary in place.
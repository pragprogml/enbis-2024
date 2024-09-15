import logging
import os
from random import sample
from typing import Tuple

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

IMAGE_H = 256
IMAGE_W = 306


def predict(
    input_image: str,
    model_file_path: str,
) -> Tuple[float, float, float]:
    """
    Predicts objects in an image using the YOLO model.

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
    """

    detection_model = YOLO(model_file_path)
    try:
        detection_results = detection_model(input_image)
    except Exception as error:
        logging.error(f"Error predicting image: {error}")
        return []

    first_result_boxes = detection_results[0].boxes.xyxy.tolist()
    first_result_confidences = detection_results[0].boxes.conf.tolist()

    if len(first_result_boxes) == 0:
        return 0, 0, 0
    else:
        first_box = first_result_boxes[0]
        logging.info(f"Boxes: {first_box[0], first_box[1], first_box[2], first_box[3]}")
        logging.info(f"Confidences: {first_result_confidences[0]}")

        return first_box[0], first_box[1], first_result_confidences[0]


def merge_test_with_sampled_validation_images(
    root_path: str,
    config_path: str,
    num_sampled_images: int,
) -> list:
    """
    Samples a specified number of images from a dataset's validation set and combines them
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
    """

    if num_sampled_images < 0:
        raise ValueError("Number of images to sample must be non-negative")

    try:
        with open(config_path) as conf_file:
            config = yaml.safe_load(conf_file)
    except FileNotFoundError:
        raise FileNotFoundError("Configuration file not found")

    # Preparing paths for test and validation datasets
    test_images_dir = os.path.join(root_path, config["data_split"]["test_path"])
    val_images_dir = os.path.join(root_path, config["data_split"]["val_path"])

    # Gathering all test and validation image paths
    try:
        all_test_image_paths = [
            os.path.join(test_images_dir, file) for file in os.listdir(test_images_dir) if file.endswith(".png")
        ]
        all_val_image_paths = [
            os.path.join(val_images_dir, file) for file in os.listdir(val_images_dir) if file.endswith(".png")
        ]
    except FileNotFoundError:
        raise FileNotFoundError("Test or Validation directory not found")

    # Sampling random images from the validation set
    try:
        randomly_sampled_val_images = sample(all_val_image_paths, num_sampled_images)
    except ValueError:
        raise ValueError("Number of images to sample exceeds available images in validation set")

    # Combining test and sampled validation images
    combined_sampled_images = all_test_image_paths + randomly_sampled_val_images

    return combined_sampled_images


def perform_object_detection_and_plot_results(
    model,
    img_path: str,
) -> np.ndarray:
    """
    Performs object detection on an image using a specified model.

    Parameters:
    model: The object detection model to be used for detecting objects in the image.
           Expected to have a method that takes an image as input and returns detection results.
    img_path (str): Path to the image file on which object detection will be performed.

    Returns:
    detect_img: An image with detection results plotted. The image is in RGB format.

    Raises:
    FileNotFoundError: If the specified image file does not exist.
    Exception: For errors related to image loading, processing, or detection.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image from path: {img_path}")

        detect_result = model(img)
        if not detect_result:
            raise RuntimeError("No detection results returned by the model")

        boxes = detect_result[0].boxes.xyxy.tolist()
        confidences = detect_result[0].boxes.conf.tolist()
        logging.info(f"Boxes: {boxes}")
        logging.info(f"Confidences: {confidences}")

        detect_img = detect_result[0].plot(probs=True, labels=False, line_width=1)

        detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
        return detect_img, boxes, confidences
    except cv2.error as cv2e:
        raise Exception(f"OpenCV error during processing: {str(cv2e)}")
    except Exception as e:
        raise Exception(f"Error in object detection: {str(e)}")


def calculate_intersection_over_union(
    prediction_box: Tuple[int, int, int, int],
    ground_truth_box: Tuple[int, int, int, int],
) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
    prediction_box (Tuple[int, int, int, int]): The bounding box of the predicted object.
        It is defined by (x_min, y_min, x_max, y_max).
    ground_truth_box (Tuple[int, int, int, int]): The bounding box of the ground truth object.
        It is defined by (x_min, y_min, x_max, y_max).

    Returns:
    float: The IoU ratio as a float value. It is the area of overlap between the two bounding
        boxes divided by the area of union.
    """

    # Determine the (x, y)-coordinates of the intersection rectangle
    x_min_inter = max(prediction_box[0], ground_truth_box[0])
    y_min_inter = max(prediction_box[1], ground_truth_box[1])
    x_max_inter = min(prediction_box[2], ground_truth_box[2])
    y_max_inter = min(prediction_box[3], ground_truth_box[3])

    # Compute the area of intersection rectangle
    intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(0, y_max_inter - y_min_inter + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    prediction_box_area = (prediction_box[2] - prediction_box[0] + 1) * (prediction_box[3] - prediction_box[1] + 1)
    ground_truth_box_area = (ground_truth_box[2] - ground_truth_box[0] + 1) * (
        ground_truth_box[3] - ground_truth_box[1] + 1
    )

    # Compute the intersection over union
    iou = intersection_area / float(prediction_box_area + ground_truth_box_area - intersection_area)

    return iou


def set_device_config() -> str:
    """
    Sets the device configuration for training based on the availability of hardware accelerators.

    Args:
    config (dict): A configuration dictionary where the device configuration will be set.

    Returns:
    None: The function modifies the 'config' dictionary in place.
    """

    if torch.cuda.is_available():
        logging.info("CUDA is available. Running on NVIDIA GPU.")
        torch.cuda.empty_cache()
        return "cuda"
    elif torch.backends.mps.is_available():
        logging.info("MPS is available. Running on Apple GPU.")
        return "mps"
    else:
        logging.info("No GPU found. Using CPU.")
        return "cpu"

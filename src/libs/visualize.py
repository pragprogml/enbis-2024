import logging
import os
from random import sample

import cv2
from matplotlib import pyplot as plt
from numpy import ceil
from seaborn import barplot

from src.libs.utils import perform_object_detection_and_plot_results


def display_labeled_images(
    image_directory,
    label_directory,
    max_images_to_display=32,
):
    """
    Displays a grid of images annotated with bounding boxes based on corresponding label files.

    Parameters:
    image_directory (str): Path to the directory containing image files. This directory
                           should exist and contain image files.
    label_directory (str): Path to the directory containing label files. Each label file
                           should have the same base name as its corresponding image file
                           and should be in `.txt` format. The directory should exist.
    max_images_to_display (int, optional): The maximum number of images to display in the grid.
                                           The actual number of images displayed may be less
                                           if there are fewer images available. Defaults to 32.

    Label Format:
    The label files are expected to contain bounding box information in the YOLO format:
    `class_id x_center y_center width height`, where each value is space-separated.
    Each line in a label file corresponds to one bounding box. Only labels with exactly
    five numerical values are considered valid.

    Notes:
    - The function randomly samples images from the image directory. If fewer images are
      available than `max_images_to_display`, it will only display the available images.
    - Images for which the corresponding label files are not found or are in an invalid
      format are skipped.

    Example Usage:
    display_labeled_images("/path/to/images", "/path/to/labels", 16)

    Raises:
    FileNotFoundError: If the specified image or label directory does not exist.
    ValueError: If no image files are found in the given image directory.
    """

    if not os.path.exists(image_directory):
        raise FileNotFoundError(f"Image directory not found: {image_directory}")
    if not os.path.exists(label_directory):
        raise FileNotFoundError(f"Label directory not found: {label_directory}")

    all_image_files = os.listdir(image_directory)
    num_images_available = len(all_image_files)

    if num_images_available == 0:
        raise ValueError("No image files found in the image directory")

    num_images_to_display = min(max_images_to_display, num_images_available)
    selected_image_files = sample(all_image_files, num_images_to_display)

    num_rows = int(ceil(num_images_to_display / 4))
    num_cols = min(num_images_to_display, 4)

    figure, axes = plt.subplots(num_rows, num_cols, figsize=(16, 16))

    for i, image_file in enumerate(selected_image_files):
        logging.info(f"Processing image: {image_file}")
        row = i // num_cols
        col = i % num_cols

        current_image_path = os.path.join(image_directory, image_file)
        image = cv2.imread(current_image_path)
        if image is None:
            logging.info(f"Failed to load image: {image_file}")
            continue

        associated_label_file = os.path.splitext(image_file)[0] + ".txt"
        current_label_path = os.path.join(label_directory, associated_label_file)
        try:
            with open(current_label_path, "r") as f:
                labels = f.read().strip().split("\n")
                logging.info(f"Found {len(labels)} labels in file: {associated_label_file}")
        except FileNotFoundError:
            logging.info(f"Label file not found: {associated_label_file}")
            continue

        for label in labels:
            parts = label.split()
            if len(parts) != 5:
                logging.info(f"Invalid label format in file: {associated_label_file}")
                continue
            class_id, x_center, y_center, width, height = map(float, parts)
            x_min = int((x_center - width / 2) * image.shape[1])
            y_min = int((y_center - height / 2) * image.shape[0])
            x_max = int((x_center + width / 2) * image.shape[1])
            y_max = int((y_center + height / 2) * image.shape[0])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 127, 255), 4)

        axes[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[row, col].grid(False)
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


def evaluate_model_on_images(
    detection_model,
    image_directory_path,
    num_images_to_sample=32,
    save_to=None,
):
    """
    Evaluates a detection and localization model on a specified number of images from a given directory.

    Parameters:
    detection_model: The object detection model to be applied to the images.
    image_directory_path (str): Path to the directory containing images to be processed.
    num_images_to_sample (int, optional): The number of images to randomly sample from the directory. Defaults to 32.
    save_to (str, optional): Path where the resulting image grid will be saved. If not provided, the grid is
    displayed on the screen.

    Returns:
    None

    Note:
    The function assumes that all files in the specified directory are image files.
    """

    if not os.path.exists(image_directory_path):
        raise FileNotFoundError(f"Directory not found: {image_directory_path}")

    all_image_files = [file for file in os.listdir(image_directory_path) if file.endswith(".png")]
    num_images_available = len(all_image_files)

    if num_images_available == 0:
        raise ValueError("No image files found in the directory")

    num_images_to_sample = min(num_images_to_sample, num_images_available)
    sampled_image_files = sample(all_image_files, num_images_to_sample)

    num_rows = int(ceil(num_images_to_sample / 4))
    num_cols = min(num_images_to_sample, 4)

    figure, subplots = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(16, 16))

    for i, image_filename in enumerate(sampled_image_files):
        logging.info(f"Processing image: {image_filename}")
        row = i // num_cols
        col = i % num_cols

        image_file_path = os.path.join(image_directory_path, image_filename)
        try:
            detected_image, boxes, confidence = perform_object_detection_and_plot_results(
                detection_model, image_file_path
            )
        except Exception as e:
            logging.info(f"Error processing {image_filename}: {e}")
            continue

        subplots[row, col].imshow(detected_image)
        subplots[row, col].axis("off")

    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()


def display_metrics(
    evaluation_metrics,
    save_to=None,
):
    """
    Displays a bar plot for YOLO model evaluation metrics.

    This function creates a bar plot for the given evaluation metrics of a YOLO model,
    including mAP50-95, mAP50, and mAP75. It allows the option to save the plot to a file.

    Parameters:
    evaluation_metrics: An object containing the evaluation metrics (mAP50-95, mAP50,Precision,Recall) for a YOLO model.
    save_to (str, optional): Where the bar plot will be saved. If not provided, the plot is displayed on the screen.

    Returns:
    None
    """

    bar_plot_axes = barplot(
        x=["mAP50-95", "mAP50", "Precision", "Recall"],
        y=[
            evaluation_metrics.box.map,
            evaluation_metrics.box.map50,
            evaluation_metrics.box.mp,
            evaluation_metrics.box.mr,
        ],
    )

    bar_plot_axes.set_title("YOLOv8 Evaluation Metrics")
    bar_plot_axes.set_xlabel("Metric")
    bar_plot_axes.set_ylabel("Value")

    current_figure = plt.gcf()
    current_figure.set_size_inches(8, 6)

    for patch in bar_plot_axes.patches:
        bar_plot_axes.annotate(
            "{:.3f}".format(patch.get_height()),
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
        )

    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()

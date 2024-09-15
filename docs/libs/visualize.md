Module libs.visualize
=====================

Functions
---------

`display_labeled_images(image_directory, label_directory, max_images_to_display=32)`
:   Displays a grid of images annotated with bounding boxes based on corresponding label files.
    
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

`display_metrics(evaluation_metrics, save_to=None)`
:   Displays a bar plot for YOLO model evaluation metrics.
    
    This function creates a bar plot for the given evaluation metrics of a YOLO model,
    including mAP50-95, mAP50, and mAP75. It allows the option to save the plot to a file.
    
    Parameters:
    evaluation_metrics: An object containing the evaluation metrics (mAP50-95, mAP50,Precision,Recall) for a YOLO model.
    save_to (str, optional): Where the bar plot will be saved. If not provided, the plot is displayed on the screen.
    
    Returns:
    None

`evaluate_model_on_images(detection_model, image_directory_path, num_images_to_sample=32, save_to=None)`
:   Evaluates a detection and localization model on a specified number of images from a given directory.
    
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
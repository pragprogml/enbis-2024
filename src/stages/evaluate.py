import argparse
import logging
import os

import yaml
from ultralytics import YOLO

from src.libs.utils import set_device_config
from src.libs.visualize import display_metrics, evaluate_model_on_images

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def evaluate(
    config_path: str,
) -> None:
    """
    Evaluate model based on configuration.

    Args:
        config_path (str): Path to the configuration file.
    """

    try:
        with open(config_path) as conf_file:
            config = yaml.safe_load(conf_file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        return
    except yaml.YAMLError as err:
        logging.error(f"Error parsing YAML file: {err}")
        return

    # Ensure all required configuration keys are present
    required_keys = ["train", "evaluate"]
    if not all(key in config for key in required_keys):
        logging.error("Missing required keys in the configuration.")
        return

    # Initialize the model
    model = YOLO(
        config["evaluate"]["best_model"],
    )

    # Save evaluation metrics and images
    try:
        reports_folder = config["evaluate"]["reports_dir"]
        metrics_image_path = os.path.join(reports_folder, config["evaluate"]["metrics_image"])
        pred_images_path = os.path.join(reports_folder, config["evaluate"]["predicted_images"])

        evaluate_model_on_images(
            detection_model=model,
            image_directory_path=config["data_split"]["test_path"],
            num_images_to_sample=12,
            save_to=pred_images_path,
        )

        metrics = model.val(device=set_device_config(), project=config["train"]["project"])
        display_metrics(evaluation_metrics=metrics, save_to=metrics_image_path)

    except KeyError as e:
        logging.error(f"Missing configuration for {e}")
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    evaluate(config_path=args.config)

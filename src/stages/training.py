import argparse
import logging
import os
import shutil

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO, settings

from src.libs.utils import set_device_config

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def training_config(
    dir: str,
) -> None:
    """
    Updates the global YOLO settings.

    Parameters:
    dir (Text): A string representing the base directory path. This function appends subdirectories
                like 'data', 'weights', and 'runs' to this base path for organizing datasets, model
                weights, and run logs, respectively.

    Returns:
    None: The function returns nothing. It updates the global 'settings' variable with the new paths
          and tool configurations.
    """

    settings.update({"datasets_dir": dir + "/data"})
    settings.update({"weights_dir": dir + "/weights"})
    settings.update({"runs_dir": dir + "/runs"})
    settings.update({"wandb": True})
    settings.update({"mlflow": False})


def training(
    config_path: str,
) -> None:
    """
    Trains a YOLO model using configuration settings from a specified YAML file.

    Parameters:
    config_path (str): A string specifying the path to the YAML configuration file. This file
                       contains all necessary parameters for training the model, such as data paths,
                       training options, and model specifications.

    Returns:
    None: This function does not return anything. It trains the model and exports it for future use.
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
    required_keys = ["pretrained", "train"]
    if not all(key in config for key in required_keys):
        logging.error("Missing required keys in the configuration.")
        return

    try:
        # Loading a pretrained model
        model = YOLO(config["pretrained"]["model"])

        # Training the model
        model.train(
            data=config["train"]["data"],
            epochs=config["train"]["epochs"],
            seed=config["train"]["seed"],
            batch=config["train"]["batch"],
            workers=config["train"]["workers"],
            device=set_device_config(),
            project=config["train"]["project"],
        )

        logging.info("Training complete.")

        # exporting in .onnx format
        onnx_path = model.export(format="onnx")  # noqa: F841

        run_path = os.path.dirname(onnx_path)
        logging.info(f".pt model copied from {run_path} to {config['train']['best_model']}")
        shutil.copyfile(os.path.join(run_path, "best.pt"), config["train"]["best_model"])

        logging.info(f".onnx model copied from {run_path} to models/")
        shutil.copyfile(onnx_path, "models/best_model.onnx")

    except Exception as e:
        logging.error(f"Error during model training: {e}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    load_dotenv(".envrc")
    ROOT_DIR = os.getenv("ROOT_DIR")

    training_config(ROOT_DIR)
    training(config_path=args.config)

Module stages.training
======================

Functions
---------

`training(config_path: str) ‑> None`
:   Trains a YOLO model using configuration settings from a specified YAML file.
    
    Parameters:
    config_path (str): A string specifying the path to the YAML configuration file. This file
                       contains all necessary parameters for training the model, such as data paths,
                       training options, and model specifications.
    
    Returns:
    None: This function does not return anything. It trains the model and exports it for future use.

`training_config(dir: str) ‑> None`
:   Updates the global YOLO settings.
    
    Parameters:
    dir (Text): A string representing the base directory path. This function appends subdirectories
                like 'data', 'weights', and 'runs' to this base path for organizing datasets, model
                weights, and run logs, respectively.
    
    Returns:
    None: The function returns nothing. It updates the global 'settings' variable with the new paths
          and tool configurations.
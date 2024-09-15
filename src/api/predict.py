import io
import logging
import os
import sys
from typing import BinaryIO

import numpy as np
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from ultralytics import YOLO

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

load_dotenv(".envrc")

PARAMS = os.getenv("PARAMS")
MODEL = os.getenv("MODEL")

if MODEL is None:
    print("Warning: The environment variable MODEL is not set.")

if PARAMS is None:
    print("Warning: The environment variable PARAMS is not set.")
else:
    try:
        with open(PARAMS) as conf_file:
            config = yaml.safe_load(conf_file)
            MODEL = config["evaluate"]["best_model"]
            logger.info(f"Configuration loaded from {PARAMS} with model: {MODEL}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found : {PARAMS}")

logger.info(f"Loading model: {MODEL}")
model = YOLO(MODEL)

description = """
The Anatomy of a Machine Learning Pipeline (API)
"""

app = FastAPI(
    title="The Anatomy of a Machine Learning Pipeline (API)",
    description=description,
    version="0.0.1",
    contact={
        "name": "PPML",
        "url": "https://github.com/pragprogml",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/license/mit/",
    },
)

Instrumentator().instrument(app).expose(app)

METRIC = Counter(
    "inference_predictions_total_plug",
    "Number of times the prediction api has been requested, and a plug was detected.",
    labelnames=("plug",),
)

METRIC = Counter(
    "inference_predictions_total_no_plug",
    "Number of times the prediction api has been requested, but no plug was detected.",
    labelnames=("no_plug",),
)


def read_image_from_bytes(
    file_bytes: BinaryIO,
) -> Image:
    """
    Loads an image from a byte stream.

    Args:
    file_bytes (BinaryIO): The byte stream of the image.

    Returns:
    Image: The loaded image object.

    Raises:
    IOError: If the image cannot be opened.
    ValueError: If the input is not valid image data.
    """
    try:
        with io.BytesIO(file_bytes) as buffer:
            image = Image.open(buffer)
            image.load()  # Load image data to avoid 'I/O operation on closed file' error
            return image
    except IOError:
        raise IOError("Unable to open the image. The file may be corrupted or the format may not be supported.")
    except ValueError:
        raise ValueError("Invalid image data. Please provide valid image bytes.")


@app.get("/healthcheck", tags=["system"])
def healthcheck():
    """
    Health Check Endpoint.

    This endpoint is used to check the health of the application. It returns a JSON response
    indicating the status of the application. This is commonly used in monitoring the application
    to ensure it's up and running.

    Returns:
        JSONResponse: A JSON response with the status of the application.
                    Example: {"status": "up"}
    """
    return JSONResponse(content=jsonable_encoder({"status": "up"}))


@app.post("/predict/image", tags=["inference"])
async def predict(
    file: UploadFile = File(...),
):
    """
    Predict objects in an image using the YOLO model.

    This endpoint accepts an image file, processes it using the YOLO (You Only Look Once) model,
    and returns the bounding box coordinates of the detected objects along with their confidence scores.

    Args:
        file (UploadFile, required): The image file to be analyzed. Supported formats are JPG and PNG.

    Raises:
        ValueError: If the uploaded image is not in JPG or PNG format.
        Exception: For any unexpected errors during processing.

    Returns:
        JSONResponse: The response contains the coordinates (x_center, y_center, width, height)
                      of the detected object's bounding box and its confidence score.
                      If no object is detected, returns zeros for all coordinates and confidence.
                      In case of an error, returns an appropriate error message with the status code.

    Example Response:
        - On successful detection: {"x_center": 120, "y_center": 340, "width": 80, "height": 150, "confidence": 0.95}
        - If no object is detected: {"x_center": 0, "y_center": 0, "width": 0, "height": 0, "confidence": 0}
        - On error: {"error": "Error message"}
    """

    try:
        file_extension = file.filename.split(".")[-1]
        is_supported_extension = file_extension in ("jpg", "png")

        if not is_supported_extension:
            raise ValueError("Image must be in jpg or png format!")

        uploaded_image = read_image_from_bytes(await file.read())

        detection_results = model.predict(uploaded_image)

        detected_boxes = detection_results[0].boxes.xyxy.tolist()
        detected_confidences = detection_results[0].boxes.conf.tolist()

        if not detected_boxes:
            METRIC.labels("no-plug").inc()
            return JSONResponse(
                content=jsonable_encoder({"x_center": 0, "y_center": 0, "width": 0, "height": 0, "confidence": 0})
            )

        primary_detected_box = detected_boxes[0]
        primary_confidence = detected_confidences[0]

        logger.info(f"Detected bounding Box: {primary_detected_box}")
        logger.info(f"Confidence: {primary_confidence}")

        bbox = {
            "x_center": np.round(primary_detected_box[0], 2),
            "y_center": np.round(primary_detected_box[1], 2),
            "width": np.round(primary_detected_box[2], 2),
            "height": np.round(primary_detected_box[3], 2),
            "confidence": np.round(primary_confidence, 2),
        }

        METRIC.labels("plug").inc()
        return JSONResponse(content=jsonable_encoder(bbox))

    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

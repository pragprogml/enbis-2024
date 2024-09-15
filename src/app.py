import argparse
import logging
import os

import gradio as gr
from dotenv import load_dotenv

from libs.utils import IMAGE_H, IMAGE_W, merge_test_with_sampled_validation_images, predict

logger = logging.getLogger()
logger.setLevel(logging.INFO)

load_dotenv(".envrc")
ROOT_DIR = os.getenv("ROOT_DIR")
DEMO_PORT = os.getenv("DEMO_PORT")

args_parser = argparse.ArgumentParser()
args_parser.add_argument("--config", dest="config", required=False, default="params.yaml")
args_parser.add_argument("--model", dest="model", required=False, default="models/best_model.pt")
args = args_parser.parse_args()

with gr.Blocks(theme=gr.themes.Default()) as demo:
    with gr.Row():
        gr.Markdown("# The Anatomy of a Machine Learning Pipeline (demo)")

    def predict_and_annotate_section(img):
        # check if args.model exists
        if not os.path.exists(args.model):
            logging.error(f"Model file not found at {args.model}")
            return [(img, []), "Error: Model file not found"]

        predicted_x_coord, predicted_y_coord, predicted_confidence = predict(img, args.model)
        logging.info(
            f"Predicted X: {predicted_x_coord}, Predicted Y: {predicted_y_coord}, Predicted Confidence: {predicted_confidence}"  # noqa: E501
        )

        annotation_sections = []

        if predicted_confidence > 0.8:
            w = 0.45 * IMAGE_W
            h = 0.25 * IMAGE_H
            logging.info(f"X: {predicted_x_coord}, Y: {predicted_y_coord}, W: {w}, H: {h}")
            annotation_sections.append(
                (
                    (
                        int(predicted_x_coord),
                        int(predicted_y_coord),
                        int(predicted_x_coord + w),
                        int(predicted_y_coord + h),
                    ),
                    "Plug",
                )
            )
            return [(img, annotation_sections), "Plug"]
        else:
            return [(img, annotation_sections), "No Plug"]

    with gr.Row():
        with gr.Column():
            with gr.Row():
                gr.Image(
                    "media/bicocca-logo.png",
                    label=None,
                    container=False,
                    show_download_button=False,
                    show_share_button=False,
                )
            with gr.Row():
                input_image = gr.Image(
                    show_download_button=False,
                    show_share_button=False,
                    label="Image",
                    height=IMAGE_H,
                    width=IMAGE_W,
                    sources=[],
                )
            with gr.Row():
                predict_button = gr.Button("Predict")

        with gr.Column():
            with gr.Row():
                prediction = gr.Label(label="Classification")
            with gr.Row():
                annotated_output_image = gr.AnnotatedImage(
                    show_label=True, show_legend=False, height=IMAGE_H, width=IMAGE_W, label="Localization"
                )

    with gr.Row():
        combined_test_validation_images = merge_test_with_sampled_validation_images(
            ROOT_DIR, os.path.join(ROOT_DIR, args.config), 16
        )
        image_examples_display = gr.Examples(
            combined_test_validation_images,
            input_image,
            examples_per_page=len(combined_test_validation_images),
            label="Test Images",
        )

    predict_button.click(predict_and_annotate_section, [input_image], [annotated_output_image, prediction])

if __name__ == "__main__":
    demo.launch(share=False, server_port=int(DEMO_PORT), inbrowser=True)

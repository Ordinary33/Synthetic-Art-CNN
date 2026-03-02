import gradio as gr
import requests
import io
from PIL import Image

from src.logger_config import setup_logger

logger = setup_logger(name="Interface_logger")
API_BASE_URL = "http://127.0.0.1:8000"


def get_available_models():
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            return response.json().get("models", ["No models found."])
    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to API.")

    return ["API OFFLINE"]


def classify_image(image: Image.Image, selected_model: str):
    if image is None:
        return "Please upload an image", "0.00%"

    if "API OFFLINE" in selected_model:
        return "API is offline", "0.00%"

    img_byt_arr = io.BytesIO()
    image.save(img_byt_arr, format="JPEG")
    img_bytes = img_byt_arr.getvalue()

    files = {"file": ("upload.jpg", img_bytes, "image/jpeg")}
    data = {"model_name": selected_model}

    try:
        response = requests.post(f"{API_BASE_URL}/classify", files=files, data=data)

        if response.status_code != 200:
            return f"API Error: {response.text}", "0.00%"

        result = response.json()
        label = result["label"]
        confidence = f"{result['confidence'] * 100: .2f}%"

        return label, confidence

    except requests.exceptions.ConnectionError:
        return "Network Error: Could not reach API", "0.00%"


available_models = get_available_models()

with gr.Blocks(title="AI Image Detector", theme=gr.themes.Soft()) as ui:
    gr.Markdown("<h1 style='text-align: center;'>🎨 AI Image Detector</h1>")
    gr.Markdown(
        "<p style='text-align: center; font-size: 110%; color: gray;'>Select a custom model and upload an image in JPG format</p>"
    )

    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=available_models,
                value=available_models[0] if available_models else None,
                label="Select CNN model",
                interactive=True,
            )
            input_image = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Classify Image", variant="primary")

        with gr.Column():
            with gr.Row():
                output_label = gr.Textbox(label="Image Label", text_align="center")
                output_confidence = gr.Textbox(
                    label="Model Confidence", text_align="center"
                )

    submit_btn.click(
        fn=classify_image,
        inputs=[input_image, model_dropdown],
        outputs=[output_label, output_confidence],
    )

if __name__ == "__main__":
    ui.launch()

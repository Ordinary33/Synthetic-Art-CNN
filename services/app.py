from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from pathlib import Path

from src.logger_config import setup_logger
from src.classify import AIDetector

app = FastAPI()
logger = setup_logger(name="API_logger")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

current_model_name = None
detector_engine = None


@app.get("/")
def check():
    return {"message": "API is running"}


@app.get("/models")
async def get_all_models():
    """Fetches all onnx format model from the artifacts directory"""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"

    models = [f.name for f in MODEL_DIR.glob("*.onnx")]
    if not models:
        return {"models": ["No models found."]}
    return {"models": models}


@app.post("/classify")
async def classify_image(file: UploadFile = File(...), model_name: str = Form(...)):
    """Receives and image and model name, and returns the label and confidence"""
    global current_model_name, detector_engine

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    if detector_engine is None or current_model_name != model_name:
        if model_name.startswith("No models"):
            raise HTTPException(status_code=400, detail="No valid model selected.")

        logger.info(f"API is loading model: {model_name}...")
        try:
            detector_engine = AIDetector(model_name)
            current_model_name = model_name
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load engine: {str(e)}"
            )

        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Could not read image file: {str(e)}"
            )

        try:
            result = detector_engine.classify(image)
            return {
                "label": result["label"],
                "confidence": result["confidence"],
                "model_used": current_model_name,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

app = FastAPI()

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
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"

    models = [f.name for f in MODEL_DIR.glob("*.onnx")]
    if not models:
        return {"models": ["No models found."]}
    return {"models": models}

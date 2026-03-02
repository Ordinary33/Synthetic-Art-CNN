import onnxruntime as ort
import numpy as np
from PIL import Image
from pathlib import Path

from src.preprocess import preprocess_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "artifacts"


class AIDetector:
    def __init__(self, model: str):
        self.classes = ["AI-Generated", "Non-AI-Generated"]
        self.model_path = MODEL_PATH / model
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def classify(self, image: Image.Image):
        input_tensor = preprocess_image(image)
        output = self.session.run(None, {self.input_name: input_tensor})
        raw_scores = output[0][0]
        predicted_idx = int(np.argmax(raw_scores))
        shifted_scores = raw_scores - np.max(raw_scores)
        exp_scores = np.exp(shifted_scores)
        probabilities = exp_scores / np.sum(exp_scores)
        confidence = float(probabilities[predicted_idx])

        return {"label": self.classes[predicted_idx], "confidence": confidence}

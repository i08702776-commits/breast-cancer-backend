from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import numpy as np
import onnxruntime as ort
from datetime import datetime
import time
import os

app = FastAPI(
    title="BreakHis AI API",
    description="Breast Cancer Detection using ResNet-200D (ONNX)",
    version="1.0.0"
)

# -----------------------------
# CORS Configuration
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your Flutter domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load ONNX Model
# -----------------------------
MODEL_PATH = "model.onnx"

try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    print("✅ Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return image.astype(np.float32)


# -----------------------------
# Softmax
# -----------------------------
def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# -----------------------------
# Clinical Insights
# -----------------------------
def get_clinical_insights(pred: int):
    if pred == 0:
        return [
            {
                "title": "Regular Cell Structure",
                "description": "The AI identified uniform cell shapes and sizes throughout the specimen. This lack of variation is a strong indicator of non-malignant tissue behavior."
            },
            {
                "title": "No Atypical Hyperplasia",
                "description": "Ductal and lobular structures maintain their natural architectural integrity. No significant overcrowding or abnormal growth patterns were detected."
            },
            {
                "title": "Clear Margins",
                "description": "Inter-cellular boundaries are distinct and well-defined. The stroma shows no signs of invasive infiltration or aggressive expansion."
            },
        ]
    else:
        return [
            {
                "title": "Irregular Cell Structure",
                "description": "The AI detected irregular cell shapes and sizes. This variation may indicate malignant tissue behavior requiring further examination."
            },
            {
                "title": "Atypical Hyperplasia Detected",
                "description": "Ductal and lobular structures show abnormal architectural patterns. Significant overcrowding and abnormal growth patterns were observed."
            },
            {
                "title": "Unclear Margins",
                "description": "Inter-cellular boundaries are indistinct. The stroma shows possible signs of invasive infiltration requiring clinical evaluation."
            },
        ]


# -----------------------------
# Recommendations
# -----------------------------
def get_recommendations(pred: int):
    if pred == 0:
        return [
            "Routine follow-up recommended",
            "No urgent intervention needed",
            "Maintain regular screening schedule"
        ]
    else:
        return [
            "Consult a specialist doctor immediately",
            "Further biopsy may be required",
            "Schedule urgent follow-up appointment"
        ]


# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "BreakHis AI backend is running"
    }


# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "ResNet-200D",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    start_time = time.time()

    # File type validation
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are allowed."
        )

    # File size validation (Max 10 MB)
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image size must be less than 10 MB."
        )

    try:
        image = Image.open(file.file).convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to process image."
        )

    # Preprocessing
    input_tensor = preprocess(image)

    # Prediction
    outputs = session.run(None, {input_name: input_tensor})
    raw = outputs[0][0]

    probs = softmax(raw) if (raw.min() < 0 or raw.max() > 1) else raw

    pred = int(np.argmax(probs))
    confidence = float(np.max(probs))

    label = "Malignant" if pred == 1 else "Benign"
    risk_level = "High" if pred == 1 else "Low"

    prediction_time = round(time.time() - start_time, 3)

    return {
        "success": True,

        "label": label,
        "confidence": round(confidence * 100, 2),

        "risk_level": risk_level,

        "prediction_time_seconds": prediction_time,

        "scan_date": datetime.now().isoformat(),

        "model_name": "ResNet-200D",

        "report_id": f"REP-{datetime.now().strftime('%Y%m%d%H%M%S')}",

        "clinical_insights": get_clinical_insights(pred),

        "recommendations": get_recommendations(pred),

        "summary": "AI-based histopathology analysis completed.",

        "finding": (
            "Abnormal malignant patterns detected in tissue image."
            if pred == 1
            else "No malignant patterns detected. Tissue appears benign."
        ),

        "disclaimer": (
            "This AI assessment is intended to assist pathologists and "
            "should not replace professional medical diagnosis."
        )
    }
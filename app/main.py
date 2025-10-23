from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
from pathlib import Path
from model_loader import predict_image

# === FastAPI setup ===
app = FastAPI(title="Chest X-Ray Pneumonia Detector")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mount static and templates
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# === Routes ===
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render homepage with upload form"""
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    """Handle image upload and model prediction"""
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction
    predicted_class, confidence = predict_image(file_path)

    # Build result dictionary
    result = {
        "filename": file.filename,
        "class": predicted_class,
        "confidence": f"{confidence * 100:.2f}%",
        "image_url": f"/uploads/{file.filename}",
    }

    return templates.TemplateResponse("index.html", {"request": request, "result": result})
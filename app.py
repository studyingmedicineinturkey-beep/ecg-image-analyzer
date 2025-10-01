# app.py  — minimal sanity-check API
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2 as cv
import platform, sys

# Optional PDF rasterization
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    fitz = None
    HAS_PYMUPDF = False

app = FastAPI(title="ECG Image Analyzer (minimal sanity check)")

def pdf_first_page_to_bgr(pdf_bytes: bytes):
    if not HAS_PYMUPDF:
        return None
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0:
        return None
    page = doc.load_page(0)
    mat = fitz.Matrix(2.0, 2.0)  # 2x scale
    pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    return bgr

@app.get("/")
def home():
    return {"ok": True, "hint": "Use POST /analyze_image (PNG/JPG/PDF). Docs at /docs"}

@app.get("/debug")
def debug():
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": getattr(np, "__version__", "n/a"),
        "opencv": getattr(cv, "__version__", "n/a"),
        "pymupdf": HAS_PYMUPDF,
    }
    return info

@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    import traceback
    try:
        data = await file.read()
        name = (file.filename or "").lower()
        ctype = (file.content_type or "").lower()

        # 1) try as image
        arr = np.frombuffer(data, np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_COLOR)

        # 2) if not an image and looks like PDF, rasterize
        if img is None and (name.endswith(".pdf") or "pdf" in ctype):
            img = pdf_first_page_to_bgr(data)

        if img is None:
            return JSONResponse({"error": "Not an image (or PDF unsupported)"}, status_code=400)

        h, w = img.shape[:2]

        # Return a tiny mock payload + decoded image size
        facts = {
            "measurements": {"hr": 75, "pr_ms": 160, "qrs_ms": 95, "qt_ms": 380, "qtc_ms": 410},
            "labels": [{"code": "UNK", "prob": 0.1}],
            "quality": {"noise": "unknown"}
        }
        return JSONResponse({"image": {"width": w, "height": h}, "facts": facts})
    except Exception as e:
        return JSONResponse(
            {
                "error": "analyze_failed",
                "detail": str(e),
                "hint": "Open /debug and /logs; then share 'detail' if you need help"
            },
            status_code=500,
        )

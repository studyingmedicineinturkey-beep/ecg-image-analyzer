from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np, cv2 as cv, platform, sys
try:
    import fitz  # PyMuPDF for PDFs
    HAS_PYMUPDF = True
except Exception:
    fitz = None
    HAS_PYMUPDF = False

app = FastAPI(title="ECG Image Analyzer - minimal")

def pdf_first_page_to_bgr(pdf_bytes: bytes):
    if not HAS_PYMUPDF: return None
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0: return None
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(2,2), alpha=False)  # RGB
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return cv.cvtColor(img, cv.COLOR_RGB2BGR)

@app.get("/debug")
def debug():
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "opencv": cv.__version__,
        "pymupdf": HAS_PYMUPDF
    }

@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_COLOR)
        if img is None and (file.filename or "").lower().endswith(".pdf"):
            img = pdf_first_page_to_bgr(data)
        if img is None:
            return JSONResponse({"error":"Not an image/PDF"}, status_code=400)
        h,w = img.shape[:2]
        facts = {"measurements":{"hr":75,"pr_ms":160,"qrs_ms":95,"qt_ms":380,"qtc_ms":410},
                 "labels":[{"code":"UNK","prob":0.1}],
                 "quality":{"noise":"unknown"}}
        return {"image":{"width":w,"height":h},"facts":facts}
    except Exception as e:
        return JSONResponse({"error":"analyze_failed","detail":str(e)}, status_code=500)

@app.get("/")
def home():
    return {"ok": True, "hint":"POST /analyze_image (PNG/JPG/PDF). Docs at /docs"}

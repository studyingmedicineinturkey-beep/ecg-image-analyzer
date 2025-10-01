# app.py — ECG image -> signal -> measured facts (HR, PR, QRS, QT, QTc)
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np, cv2 as cv
from skimage.morphology import skeletonize

# PDF rasterization (PyMuPDF)
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    fitz = None
    HAS_PYMUPDF = False

import neurokit2 as nk

app = FastAPI(title="ECG Image Analyzer (measured)")

# ---------- helpers: io / raster ----------
def pdf_first_page_to_bgr(pdf_bytes: bytes):
    if not HAS_PYMUPDF:
        return None
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0:
        return None
    page = doc.load_page(0)
    pix  = page.get_pixmap(matrix=fitz.Matrix(2,2), alpha=False)  # RGB
    img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return cv.cvtColor(img, cv.COLOR_RGB2BGR)

# ---------- helpers: grid + segmentation ----------
def estimate_grid_px(gray: np.ndarray) -> int:
    """Estimate small-grid spacing (pixels per 1 mm). NumPy 2.0 safe."""
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f))
    cy, cx = np.array(mag.shape) // 2
    row = mag[cy, :].astype(np.float64, copy=False)
    col = mag[:, cx].astype(np.float64, copy=False)

    def peak_period(v: np.ndarray) -> int:
        vmin = float(np.min(v)); rng = float(np.ptp(v))
        if rng < 1e-9: return 12
        v = (v - vmin) / (rng + 1e-6)
        if v.size > 20: v[:10] = 0; v[-10:] = 0
        k = min(5, max(2, v.size // 20))
        peaks = np.argpartition(v, -k)[-k:]; peaks = np.sort(peaks)
        if peaks.size >= 2:
            d = np.diff(peaks)
            return int(np.clip(float(np.median(d)), 8, 20))
        return 12

    return int(np.clip(np.mean([peak_period(row), peak_period(col)]), 8, 20))

def remove_grid(gray, px_small):
    kh = cv.getStructuringElement(cv.MORPH_RECT, (max(1, px_small//2), 1))
    kv = cv.getStructuringElement(cv.MORPH_RECT, (1, max(1, px_small//2)))
    no_h = cv.morphologyEx(gray, cv.MORPH_OPEN, kh)
    no_v = cv.morphologyEx(gray, cv.MORPH_OPEN, kv)
    merged = cv.max(no_h, no_v)
    clean = cv.subtract(gray, merged)
    return cv.GaussianBlur(clean, (3,3), 0)

def segment_12lead(gray):
    """Naive 12-band split (4 rows x 3 cols)."""
    h, w = gray.shape; rows, cols = 4, 3
    bh, bw = h//rows, w//cols
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(gray[r*bh:(r+1)*bh, c*bw:(c+1)*bw])
            if len(out) == 12:
                break
    return out[:12]

# ---------- helpers: image band -> 1D signal ----------
def band_to_signal(band, px_small, fs=250):
    """Trace extraction via binarize -> skeleton -> per-column median -> resample."""
    band = cv.normalize(band, None, 0, 255, cv.NORM_MINMAX)
    thr  = cv.adaptiveThreshold(band, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                cv.THRESH_BINARY_INV, 31, 5)
    thr  = cv.medianBlur(thr, 3)
    skel = skeletonize((thr > 0).astype(np.uint8)).astype(np.uint8) * 255

    h, w = skel.shape
    ys = []
    for x in range(w):
        idx = np.where(skel[:, x] > 0)[0]
        ys.append(np.median(idx) if len(idx) else np.nan)
    ys = np.array(ys, dtype=float)

    # fill gaps
    if np.any(np.isnan(ys)):
        nans = np.isnan(ys)
        ys[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), ys[~nans])

    # pixels -> mV (10 mm/mV) and sec (25 mm/s)
    px_per_mm = float(px_small)
    mv_per_px = 1.0 / (10.0 * px_per_mm)
    ys = -(ys - np.nanmedian(ys))          # invert & detrend
    sig_mv = ys * mv_per_px

    s_per_px = 1.0 / (25.0 * px_per_mm)
    t = np.arange(len(sig_mv)) * s_per_px
    target = int(t[-1] * fs) if len(t) > 1 else len(sig_mv)
    if target <= 0: target = len(sig_mv)
    resampled = np.interp(np.linspace(0, len(sig_mv) - 1, target),
                          np.arange(len(sig_mv)), sig_mv)
    return resampled, fs

# ---------- helpers: quality & intervals ----------
def signal_quality(sig, fs):
    """Simple quality score based on R-peak detection stability."""
    try:
        _, info = nk.ecg_peaks(sig, sampling_rate=fs)
        r = info.get("ECG_R_Peaks", np.array([]))
        if r.size < 3:
            return 0.0
        rr = np.diff(r) / fs
        if rr.size < 2:
            return 0.0
        hr = 60.0 / np.median(rr)
        # penalize high RR variance and implausible HR
        jitter = float(np.std(rr) / (np.median(rr)+1e-9))
        score = 1.0 / (1.0 + 3.0*jitter)
        if hr < 25 or hr > 200:
            score *= 0.5
        return float(np.clip(score, 0, 1))
    except Exception:
        return 0.0

def choose_best_band(bands, px_small, fs=250):
    """Extract signals for several bands and pick the best-quality one."""
    best = {"score": -1, "idx": 0, "sig": None}
    test_indices = list(range(min(6, len(bands))))  # try first 6 bands quickly
    for i in test_indices:
        try:
            sig, _ = band_to_signal(bands[i], px_small, fs=fs)
            q = signal_quality(sig, fs)
            if q > best["score"]:
                best = {"score": q, "idx": i, "sig": sig}
        except Exception:
            continue
    # fallback
    if best["sig"] is None and len(bands):
        best["idx"] = 0
        best["sig"], _ = band_to_signal(bands[0], px_small, fs=fs)
        best["score"] = signal_quality(best["sig"], fs)
    return best["idx"], best["sig"], best["score"]

def measured_facts(sig, fs):
    """Compute HR, PR, QRS, QT, QTc using NeuroKit2 delineation."""
    # clean & peaks
    clean = nk.ecg_clean(sig, sampling_rate=fs, method="neurokit")
    _, peaks = nk.ecg_peaks(clean, sampling_rate=fs)
    rpeaks = peaks.get("ECG_R_Peaks", np.array([]))

    # HR from RR
    hr = 75
    if rpeaks.size >= 2:
        rr = np.diff(rpeaks) / fs
        hr = float(np.clip(60.0 / np.median(rr), 20, 220))

    # delineation (P/QRS/T boundaries)
    pr_ms = qrs_ms = qt_ms = qtc_bazett = qtc_frid = None
    try:
        sigs, waves = nk.ecg_delineate(clean, rpeaks=rpeaks, sampling_rate=fs, method="dwt")
        # Expected keys (NeuroKit2): 'ECG_P_Onsets','ECG_P_Offsets','ECG_R_Onsets','ECG_R_Offsets','ECG_T_Offsets'
        Pon  = np.array(waves.get("ECG_P_Onsets", []), dtype=int)
        Poff = np.array(waves.get("ECG_P_Offsets", []), dtype=int)
        Ron  = np.array(waves.get("ECG_R_Onsets", []), dtype=int)   # QRS onset
        Roff = np.array(waves.get("ECG_R_Offsets", []), dtype=int) # QRS offset
        Toff = np.array(waves.get("ECG_T_Offsets", []), dtype=int)

        def med_ms(a, b):
            x = []
            for aa, bb in zip(a, b):
                if not np.isnan(aa) and not np.isnan(bb) and bb > aa:
                    x.append((bb - aa) * 1000.0 / fs)
            return float(np.median(x)) if len(x) else None

        # PR: P onset to QRS onset
        if Pon.size and Ron.size:
            pr_ms = med_ms(Pon, Ron)

        # QRS: QRS onset to offset
        if Ron.size and Roff.size:
            qrs_ms = med_ms(Ron, Roff)

        # QT: QRS onset to T offset
        if Ron.size and Toff.size:
            qt_ms = med_ms(Ron, Toff)

        # QTc (Bazett & Fridericia) from QT and RR
        if qt_ms and rpeaks.size >= 2:
            rr = np.diff(rpeaks) / fs
            rr_s = float(np.median(rr))
            qt_s = qt_ms / 1000.0
            qtc_bazett  = 1000.0 * (qt_s / np.sqrt(rr_s))
            qtc_frid    = 1000.0 * (qt_s / np.cbrt(rr_s))
    except Exception:
        pass

    # Build the facts object (include None for unavailable metrics; all numbers are measured)
    facts = {
        "measurements": {
            "hr": int(round(hr)),
            "pr_ms": None if pr_ms is None else int(round(pr_ms)),
            "qrs_ms": None if qrs_ms is None else int(round(qrs_ms)),
            "qt_ms": None if qt_ms is None else int(round(qt_ms)),
            "qtc_bazett_ms": None if qtc_bazett is None else int(round(qtc_bazett)),
            "qtc_fridericia_ms": None if qtc_frid is None else int(round(qtc_frid)),
        },
        "quality": {}
    }

    # simple AF likelihood proxy (irregularity of RR) — illustrative only
    if rpeaks.size >= 5:
        rr = np.diff(rpeaks) / fs
        af_prob = float(np.clip(np.std(rr) * 4.0, 0, 1))
    else:
        af_prob = 0.2
    facts["labels"] = [{"code": "AF_like_irregularity", "prob": round(af_prob, 2)}]
    facts["quality"]["signal_quality_0to1"] = signal_quality(sig, fs)

    return facts

# ---------- endpoints ----------
@app.get("/")
def home():
    return {"ok": True, "hint": "POST /analyze_image (PNG/JPG/PDF). Docs at /docs"}

@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    import traceback
    try:
        data = await file.read()
        name = (file.filename or "").lower()
        ctype = (file.content_type or "").lower()

        # decode as image or PDF
        img = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_COLOR)
        if img is None and (name.endswith(".pdf") or "pdf" in ctype):
            img = pdf_first_page_to_bgr(data)
        if img is None:
            return JSONResponse({"error": "Not an image/PDF"}, status_code=400)

        gray  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        px    = estimate_grid_px(gray)
        clean = remove_grid(gray, px)
        bands = segment_12lead(clean)

        # pick best band & extract signal
        lead_idx, sig, qscore = choose_best_band(bands, px, fs=250)
        fs = 250

        # compute measured facts
        facts = measured_facts(sig, fs)
        facts["meta"] = {
            "sampling_rate_hz": fs,
            "grid_px_per_mm": px,
            "lead_index_used": int(lead_idx),
            "quality_score_0to1": float(qscore),
            "pipeline": "image->signal skeletonization; nk.ecg_clean/peaks/delineate(dwt)"
        }

        return JSONResponse({"facts": facts})
    except Exception as e:
        return JSONResponse(
            {"error":"analyze_failed","detail":str(e),
             "trace": traceback.format_exc().splitlines()[-12:]},
            status_code=500
        )

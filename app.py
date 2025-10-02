# app.py — Professional path: WFDB ingestion + (kept) image fallback
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np, io, zipfile, os, tempfile

# ---- image stack (kept from your MVP; shortened here) ----
import cv2 as cv
from skimage.morphology import skeletonize
try:
    import fitz
    HAS_PYMUPDF = True
except Exception:
    fitz = None
    HAS_PYMUPDF = False

import neurokit2 as nk
import wfdb  # NEW

app = FastAPI(title="ECG Analyzer (WFDB-first + image fallback)")

# ---------- Common helpers ----------
def facts_from_signal(sig, fs):
    """Compute HR/PR/QRS/QT/QTc with robust fallbacks on a 1-D signal."""
    # cleaner 1 (always)
    clean1 = nk.ecg_clean(sig, sampling_rate=fs, method="neurokit")
    peaks1 = nk.ecg_peaks(clean1, sampling_rate=fs)[1].get("ECG_R_Peaks", np.array([]))

    # HR (fallback to autocorr)
    def hr_autocorr(x):
        x = x - np.mean(x)
        ac = np.correlate(x, x, mode='full')[len(x)-1:]
        lag_min = int(0.3*fs); lag_max = int(2.5*fs)
        if lag_max <= lag_min or lag_max >= len(ac): return 75.0
        idx = lag_min + np.argmax(ac[lag_min:lag_max]); rr = idx / fs
        return float(np.clip(60.0/rr, 20, 220))

    if getattr(peaks1, "size", 0) >= 2:
        rr = np.diff(peaks1) / fs
        hr = float(np.clip(60.0 / np.median(rr), 20, 220))
    else:
        hr = hr_autocorr(clean1)

    pr_ms = qrs_ms = qt_ms = qtc_b = qtc_f = None
    for method in ("dwt", "peak"):
        try:
            _, waves = nk.ecg_delineate(clean1, rpeaks=peaks1, sampling_rate=fs, method=method)
            Pon  = np.array(waves.get("ECG_P_Onsets", []), dtype=float)
            Ron  = np.array(waves.get("ECG_R_Onsets", []), dtype=float)
            Roff = np.array(waves.get("ECG_R_Offsets", []), dtype=float)
            Toff = np.array(waves.get("ECG_T_Offsets", []), dtype=float)

            def med_ms(a, b):
                vals = [(bb-aa)*1000.0/fs for aa,bb in zip(a,b) if not np.isnan(aa) and not np.isnan(bb) and bb>aa]
                return float(np.median(vals)) if vals else None

            if pr_ms is None and Pon.size and Ron.size:   pr_ms  = med_ms(Pon, Ron)
            if qrs_ms is None and Ron.size and Roff.size: qrs_ms = med_ms(Ron, Roff)
            if qt_ms is None and Ron.size and Toff.size:  qt_ms  = med_ms(Ron, Toff)

            if qt_ms and getattr(peaks1, "size", 0) >= 2:
                rr_s = float(np.median(np.diff(peaks1)/fs))
                qt_s = qt_ms/1000.0
                qtc_b = 1000.0 * (qt_s / np.sqrt(rr_s))
                qtc_f = 1000.0 * (qt_s / np.cbrt(rr_s))
        except Exception:
            continue

    # quality proxy
    qscore = 0.0
    try:
        info = nk.ecg_peaks(clean1, sampling_rate=fs)[1]
        r = info.get("ECG_R_Peaks", np.array([]))
        if getattr(r, "size", 0) >= 3:
            rr = np.diff(r) / fs
            if rr.size >= 2:
                jitter = float(np.std(rr) / (np.median(rr)+1e-9))
                qscore = float(np.clip(1.0 / (1.0 + 3.0*jitter), 0, 1))
    except Exception:
        pass

    # AF-like proxy (illustrative only)
    af_prob = 0.2
    if getattr(peaks1, "size", 0) >= 5:
        rr = np.diff(peaks1) / fs
        af_prob = float(np.clip(np.std(rr) * 4.0, 0, 1))

    return {
        "measurements": {
            "hr": int(round(hr)),
            "pr_ms": None if pr_ms is None else int(round(pr_ms)),
            "qrs_ms": None if qrs_ms is None else int(round(qrs_ms)),
            "qt_ms": None if qt_ms is None else int(round(qt_ms)),
            "qtc_bazett_ms": None if qtc_b is None else int(round(qtc_b)),
            "qtc_fridericia_ms": None if qtc_f is None else int(round(qtc_f)),
        },
        "labels": [{"code": "AF_like_irregularity", "prob": round(af_prob, 2)}],
        "quality": {"signal_quality_0to1": qscore}
    }

# ---------- WFDB ingestion ----------
@app.post("/analyze_wfdb")
async def analyze_wfdb(file: UploadFile = File(...)):
    """
    Accepts a ZIP containing a WFDB record: <name>.hea + <name>.dat (+ optional .atr).
    Example test files: PTB-XL or MIT-BIH (convert to WFDB if needed).
    """
    import traceback
    try:
        data = await file.read()
        with tempfile.TemporaryDirectory() as d:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                z.extractall(d)
            # find a header (.hea)
            hea = [f for f in os.listdir(d) if f.lower().endswith(".hea")]
            if not hea:
                return JSONResponse({"error": "No .hea file found in ZIP"}, status_code=400)
            rec_name = os.path.splitext(hea[0])[0]
            rec = wfdb.rdrecord(os.path.join(d, rec_name))
            # choose a lead: if 12-lead, prefer II or V5; else channel 0
            ch_names = [str(x) for x in (rec.sig_name or [])]
            prefer = None
            for cand in ("II","V5","V2","V1"):
                if cand in ch_names:
                    prefer = ch_names.index(cand); break
            ch_idx = prefer if prefer is not None else 0
            sig = rec.p_signal[:, ch_idx].astype(float)
            fs = int(rec.fs)
            facts = facts_from_signal(sig, fs)
            facts["meta"] = {
                "source": "wfdb",
                "record": rec_name,
                "lead_name": ch_names[ch_idx] if ch_names else f"ch{ch_idx}",
                "sampling_rate_hz": fs,
                "units": rec.units[ch_idx] if getattr(rec, "units", None) else "mV"
            }
            return JSONResponse({"facts": facts})
    except Exception as e:
        return JSONResponse({"error": "analyze_failed", "detail": str(e)}, status_code=500)

# ---------- Image route (kept: simplified – still acceptable as beta) ----------
def pdf_first_page_to_bgr(pdf_bytes: bytes):
    if not HAS_PYMUPDF: return None
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0: return None
    page = doc.load_page(0)
    pix  = page.get_pixmap(matrix=fitz.Matrix(2,2), alpha=False)
    img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return cv.cvtColor(img, cv.COLOR_RGB2BGR)

def enhance(gray):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    e = clahe.apply(gray)
    return cv.bilateralFilter(e, d=5, sigmaColor=50, sigmaSpace=5)

def estimate_grid_px(gray: np.ndarray) -> int:
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f))
    cy, cx = np.array(mag.shape)//2
    row = mag[cy, :].astype(np.float64, copy=False)
    col = mag[:, cx].astype(np.float64, copy=False)
    def pp(v):
        vmin = float(np.min(v)); rng = float(np.ptp(v))
        if rng < 1e-9: return 12
        v = (v - vmin)/(rng+1e-6)
        if v.size>20: v[:10]=0; v[-10:]=0
        k = min(5, max(2, v.size//20))
        peaks = np.argpartition(v, -k)[-k:]; peaks = np.sort(peaks)
        if peaks.size>=2: return int(np.clip(float(np.median(np.diff(peaks))), 8, 20))
        return 12
    return int(np.clip(np.mean([pp(row), pp(col)]), 8, 20))

def remove_grid(gray, px_small):
    kh = cv.getStructuringElement(cv.MORPH_RECT, (max(1, px_small//2), 1))
    kv = cv.getStructuringElement(cv.MORPH_RECT, (1, max(1, px_small//2)))
    merged = cv.max(cv.morphologyEx(gray, cv.MORPH_OPEN, kh),
                    cv.morphologyEx(gray, cv.MORPH_OPEN, kv))
    return cv.GaussianBlur(cv.subtract(gray, merged), (3,3), 0)

def segment_12lead(gray):
    h, w = gray.shape; bh, bw = h//4, w//3
    out = []
    for r in range(4):
        for c in range(3):
            out.append(gray[r*bh:(r+1)*bh, c*bw:(c+1)*bw])
            if len(out)==12: break
    return out[:12]

def band_to_signal(band, px_small, fs=250):
    band = cv.normalize(band, None, 0, 255, cv.NORM_MINMAX)
    thr1 = cv.adaptiveThreshold(band,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,31,5)
    blur = cv.GaussianBlur(band,(5,5),0)
    _, thr2 = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    thr = cv.max(thr1, thr2)
    thr = cv.morphologyEx(thr, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT,(3,3)),1)
    thr = cv.medianBlur(thr, 3)
    skel = skeletonize((thr>0).astype(np.uint8)).astype(np.uint8)*255
    h,w = skel.shape; ys=[]
    for x in range(w):
        idx = np.where(skel[:,x]>0)[0]
        ys.append(float(np.median(idx)) if idx.size else np.nan)
    ys = np.array(ys, dtype=float)
    if np.any(np.isnan(ys)):
        nans = np.isnan(ys)
        if np.any(~nans): ys[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), ys[~nans])
        else: ys[:] = 0.0
    pxmm = float(px_small); mv_per_px = 1.0/(10.0*pxmm)
    ys = -(ys - np.nanmedian(ys)); sig_mv = ys * mv_per_px
    s_per_px = 1.0/(25.0*pxmm); t = np.arange(len(sig_mv))*s_per_px
    target = max(1500, (int(t[-1]*fs) if len(t)>1 else len(sig_mv)))  # ensure length
    res = np.interp(np.linspace(0, len(sig_mv)-1, target), np.arange(len(sig_mv)), sig_mv)
    return res, fs

def choose_best_band(bands, px_small, fs=250):
    best={"score":-1,"idx":0,"sig":None}
    for i in range(min(12,len(bands))):
        try:
            s,_ = band_to_signal(bands[i], px_small, fs)
            var=float(np.var(s))
            q=0.0
            try:
                r= nk.ecg_peaks(nk.ecg_clean(s,fs), sampling_rate=fs)[1].get("ECG_R_Peaks", np.array([]))
                if getattr(r,"size",0)>=3:
                    rr=np.diff(r)/fs
                    if rr.size>=2:
                        jitter=float(np.std(rr)/(np.median(rr)+1e-9))
                        q=float(np.clip(1.0/(1.0+3.0*jitter),0,1))
            except: pass
            score = 0.6*q + 0.4*(np.tanh(var*5))
            if score>best["score"]: best={"score":score,"idx":i,"sig":s}
        except: continue
    if best["sig"] is None and len(bands):
        best["sig"], _ = band_to_signal(bands[0], px_small, fs)
    return best["idx"], best["sig"], best["score"]

@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    import traceback
    try:
        data = await file.read()
        name = (file.filename or "").lower()
        ctype = (file.content_type or "").lower()
        img = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_COLOR)
        if img is None and (name.endswith(".pdf") or "pdf" in ctype):
            if not HAS_PYMUPDF: return JSONResponse({"error":"PDF not supported on server"}, 400)
            doc = fitz.open(stream=data, filetype="pdf")
            pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2,2), alpha=False)
            img = cv.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.h,pix.w,3), cv.COLOR_RGB2BGR)
        if img is None: return JSONResponse({"error":"Not an image/PDF"}, 400)

        gray = enhance(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
        px = estimate_grid_px(gray)
        bands = segment_12lead(remove_grid(gray, px))
        idx, sig, qscore = choose_best_band(bands, px, fs=250)
        facts = facts_from_signal(sig, fs=250)
        facts["meta"] = {"source":"image", "lead_index_used": int(idx), "grid_px_per_mm": int(px), "quality_score_0to1": float(qscore)}
        return JSONResponse({"facts": facts})
    except Exception as e:
        return JSONResponse({"error":"analyze_failed","detail":str(e)}, status_code=500)

@app.get("/")
def root():
    return {"ok": True, "docs": "/docs", "routes": ["/analyze_wfdb (ZIP of WFDB .hea/.dat)", "/analyze_image (PNG/JPG/PDF)"]}

# app.py — ECG Analyzer (WFDB-first + improved image fallback)
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

import numpy as np
import io, zipfile, os, tempfile

# ---- image stack ----
import cv2 as cv
from skimage.morphology import skeletonize

# PDF support (optional)
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    fitz = None
    HAS_PYMUPDF = False

# physiology / signals
import neurokit2 as nk
import wfdb
from scipy import signal as sp_signal

app = FastAPI(title="ECG Analyzer (WFDB-first + improved image)")

# --------------------------------------------------------------------
# ---------------------- Common signal helpers -----------------------
# --------------------------------------------------------------------
def facts_from_signal(sig: np.ndarray, fs: int):
    """Compute HR/PR/QRS/QT/QTc with robust fallbacks on a 1-D signal."""
    sig = np.asarray(sig, dtype=float)
    if sig.ndim != 1:
        sig = sig.ravel()

    # Clean + peaks
    clean = nk.ecg_clean(sig, sampling_rate=fs, method="neurokit")
    info = nk.ecg_peaks(clean, sampling_rate=fs)[1]
    r = info.get("ECG_R_Peaks", np.array([]))

    # HR (fallback to autocorrelation if needed)
    hr = _hr_from_rr(r, fs)
    if hr is None:
        hr = _hr_autocorr(clean, fs)

    pr_ms = qrs_ms = qt_ms = qtc_b = qtc_f = None
    # Try two delineation methods
    for method in ("dwt", "peak"):
        try:
            _, waves = nk.ecg_delineate(clean, rpeaks=r, sampling_rate=fs, method=method)
            Pon  = _to_f(waves.get("ECG_P_Onsets", []))
            Ron  = _to_f(waves.get("ECG_R_Onsets", []))
            Roff = _to_f(waves.get("ECG_R_Offsets", []))
            Toff = _to_f(waves.get("ECG_T_Offsets", []))

            if pr_ms is None and Pon.size and Ron.size:
                pr_ms  = _median_ms(Pon, Ron, fs)
            if qrs_ms is None and Ron.size and Roff.size:
                qrs_ms = _median_ms(Ron, Roff, fs)
            if qt_ms is None and Ron.size and Toff.size:
                qt_ms  = _median_ms(Ron, Toff, fs)

            if qt_ms and r.size >= 2:
                rr_s = float(np.median(np.diff(r)) / fs)
                qt_s = qt_ms/1000.0
                qtc_b = 1000.0 * (qt_s / np.sqrt(rr_s))
                qtc_f = 1000.0 * (qt_s / np.cbrt(rr_s))
        except Exception:
            continue

    # quality proxy from RR jitter
    qscore = 0.0
    try:
        if r.size >= 3:
            rr = np.diff(r) / fs
            if rr.size >= 2:
                jitter = float(np.std(rr) / (np.median(rr) + 1e-9))
                qscore = float(np.clip(1.0 / (1.0 + 3.0 * jitter), 0, 1))
    except Exception:
        pass

    # toy AF-like index (illustrative only)
    af_prob = 0.2
    if r.size >= 5:
        rr = np.diff(r) / fs
        af_prob = float(np.clip(np.std(rr) * 4.0, 0, 1))

    return {
        "measurements": {
            "hr": int(round(hr)) if hr is not None else None,
            "pr_ms": None if pr_ms is None else int(round(pr_ms)),
            "qrs_ms": None if qrs_ms is None else int(round(qrs_ms)),
            "qt_ms": None if qt_ms is None else int(round(qt_ms)),
            "qtc_bazett_ms": None if qtc_b is None else int(round(qtc_b)),
            "qtc_fridericia_ms": None if qtc_f is None else int(round(qtc_f)),
        },
        "labels": [{"code": "AF_like_irregularity", "prob": round(af_prob, 2)}],
        "quality": {"signal_quality_0to1": qscore},
    }


def _to_f(x):
    return np.array(x, dtype=float) if hasattr(x, "__len__") else np.array([], dtype=float)


def _median_ms(a, b, fs):
    vals = [(bb - aa) * 1000.0 / fs for aa, bb in zip(a, b) if not np.isnan(aa) and not np.isnan(bb) and bb > aa]
    return float(np.median(vals)) if len(vals) else None


def _hr_from_rr(r, fs):
    r = np.asarray(r)
    if r.size < 2:
        return None
    rr = np.diff(r) / fs
    if rr.size == 0 or np.median(rr) <= 0:
        return None
    return float(np.clip(60.0 / np.median(rr), 20, 220))


def _hr_autocorr(x, fs):
    x = x - np.mean(x)
    ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
    lag_min, lag_max = int(0.32 * fs), int(2.5 * fs)  # ~24–187 bpm
    if lag_max <= lag_min or lag_max >= len(ac):
        return None
    idx = lag_min + int(np.argmax(ac[lag_min:lag_max]))
    rr = idx / fs
    return float(np.clip(60.0 / rr, 20, 220))


def _quality_rr(sig, fs):
    """Return (quality 0..1, HR by RR or None)."""
    try:
        clean = nk.ecg_clean(sig, sampling_rate=fs)
        r = nk.ecg_peaks(clean, sampling_rate=fs)[1].get("ECG_R_Peaks", np.array([]))
        if r.size < 3:
            return 0.0, None
        rr = np.diff(r) / fs
        if rr.size < 2:
            return 0.0, None
        jitter = float(np.std(rr) / (np.median(rr) + 1e-9))
        q = float(np.clip(1.0 / (1.0 + 3.0 * jitter), 0, 1))
        hr = float(np.clip(60.0 / np.median(rr), 20, 220))
        return q, hr
    except Exception:
        return 0.0, None


# --------------------------------------------------------------------
# ------------------------- WFDB ingestion ---------------------------
# --------------------------------------------------------------------
@app.post("/analyze_wfdb")
async def analyze_wfdb(file: UploadFile = File(...)):
    """
    Upload a ZIP with WFDB <name>.hea + <name>.dat (+ optional .atr).
    """
    try:
        data = await file.read()
        with tempfile.TemporaryDirectory() as d:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                z.extractall(d)

            hea = [f for f in os.listdir(d) if f.lower().endswith(".hea")]
            if not hea:
                return JSONResponse({"error": "No .hea file found in ZIP"}, status_code=400)

            rec_name = os.path.splitext(hea[0])[0]
            rec = wfdb.rdrecord(os.path.join(d, rec_name))

            ch_names = [str(x) for x in (rec.sig_name or [])]
            prefer = None
            for cand in ("II", "V5", "V2", "V1"):
                if cand in ch_names:
                    prefer = ch_names.index(cand)
                    break
            ch_idx = prefer if prefer is not None else 0

            sig = rec.p_signal[:, ch_idx].astype(float)
            fs = int(rec.fs)

            facts = facts_from_signal(sig, fs)
            facts["meta"] = {
                "source": "wfdb",
                "record": rec_name,
                "lead_name": ch_names[ch_idx] if ch_names else f"ch{ch_idx}",
                "sampling_rate_hz": fs,
                "units": rec.units[ch_idx] if getattr(rec, "units", None) else "mV",
            }
            return JSONResponse({"facts": facts})
    except Exception as e:
        return JSONResponse({"error": "analyze_failed", "detail": str(e)}, status_code=500)


# --------------------------------------------------------------------
# -------------------- Image pipeline helpers ------------------------
# --------------------------------------------------------------------
def pdf_first_page_to_bgr(pdf_bytes: bytes):
    if not HAS_PYMUPDF:
        return None
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0:
        return None
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return cv.cvtColor(img, cv.COLOR_RGB2BGR)


def enhance(gray: np.ndarray):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    e = clahe.apply(gray)
    return cv.bilateralFilter(e, d=5, sigmaColor=50, sigmaSpace=5)


def estimate_grid_px(gray: np.ndarray) -> int:
    """Return small-square pixels; NumPy 2.0 safe."""
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f))
    cy, cx = np.array(mag.shape) // 2
    row = mag[cy, :].astype(np.float64, copy=False)
    col = mag[:, cx].astype(np.float64, copy=False)

    def pp(v):
        vmin = float(np.min(v))
        rng = float(np.ptp(v))  # np.ptp, not ndarray.ptp (NumPy 2.0 safe)
        if rng < 1e-9:
            return 12
        v = (v - vmin) / (rng + 1e-6)
        if v.size > 20:
            v[:10] = 0
            v[-10:] = 0
        k = min(5, max(2, v.size // 20))
        peaks = np.argpartition(v, -k)[-k:]
        peaks = np.sort(peaks)
        if peaks.size >= 2:
            return int(np.clip(float(np.median(np.diff(peaks))), 8, 20))
        return 12

    return int(np.clip(np.mean([pp(row), pp(col)]), 8, 20))


def remove_grid(gray, px_small):
    kh = cv.getStructuringElement(cv.MORPH_RECT, (max(1, px_small // 2), 1))
    kv = cv.getStructuringElement(cv.MORPH_RECT, (1, max(1, px_small // 2)))
    merged = cv.max(cv.morphologyEx(gray, cv.MORPH_OPEN, kh),
                    cv.morphologyEx(gray, cv.MORPH_OPEN, kv))
    return cv.GaussianBlur(cv.subtract(gray, merged), (3, 3), 0)


def segment_12lead(gray):
    h, w = gray.shape
    bh, bw = h // 4, w // 3
    out = []
    for r in range(4):
        for c in range(3):
            out.append(gray[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw])
            if len(out) == 12:
                break
    return out[:12]


def band_to_signal(band, px_small, fs=250):
    band = cv.normalize(band, None, 0, 255, cv.NORM_MINMAX)
    thr1 = cv.adaptiveThreshold(band, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv.THRESH_BINARY_INV, 31, 5)
    blur = cv.GaussianBlur(band, (5, 5), 0)
    _, thr2 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    thr = cv.max(thr1, thr2)
    thr = cv.morphologyEx(thr, cv.MORPH_CLOSE,
                          cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), 1)
    thr = cv.medianBlur(thr, 3)
    skel = skeletonize((thr > 0).astype(np.uint8)).astype(np.uint8) * 255

    h, w = skel.shape
    ys = []
    for x in range(w):
        idx = np.where(skel[:, x] > 0)[0]
        ys.append(float(np.median(idx)) if idx.size else np.nan)
    ys = np.array(ys, dtype=float)

    if np.any(np.isnan(ys)):
        nans = np.isnan(ys)
        if np.any(~nans):
            ys[nans] = np.interp(np.flatnonzero(nans),
                                 np.flatnonzero(~nans), ys[~nans])
        else:
            ys[:] = 0.0

    # scale to mV
    mv_per_px = 1.0 / (10.0 * float(px_small))  # 10 mm/mV, px/mm
    ys = -(ys - np.nanmedian(ys))
    sig_mv = ys * mv_per_px

    # resample to timebase 25 mm/s
    s_per_px = 1.0 / (25.0 * float(px_small))
    t = np.arange(len(sig_mv)) * s_per_px
    target = max(1500, int(t[-1] * fs) if len(t) > 1 else len(sig_mv))
    res = np.interp(np.linspace(0, len(sig_mv) - 1, target),
                    np.arange(len(sig_mv)), sig_mv)
    return res, fs


def choose_best_band(bands, px_small, fs=250):
    """Score by RR quality + variance; return (idx, sig, score)."""
    best = {"score": -1, "idx": 0, "sig": None}
    for i in range(min(12, len(bands))):
        try:
            s, _ = band_to_signal(bands[i], px_small, fs)
            var = float(np.var(s))
            q, _ = _quality_rr(s, fs)
            score = 0.6 * q + 0.4 * (np.tanh(var * 5))
            if score > best["score"]:
                best = {"score": score, "idx": i, "sig": s}
        except Exception:
            continue
    if best["sig"] is None and len(bands):
        best["sig"], _ = band_to_signal(bands[0], px_small, fs)
    return best["idx"], best["sig"], best["score"]


def pick_rhythm_strip(bands, px_small, fs=250):
    """Prefer bottom rhythm strip (index 11 in 4x3 layout) but fall back intelligently."""
    pref_idx = min(11, len(bands) - 1)
    sig_pref, _ = band_to_signal(bands[pref_idx], px_small, fs)
    q_pref, hr_pref = _quality_rr(sig_pref, fs)
    hr_pref_ac = _hr_autocorr(sig_pref, fs)

    auto_idx, sig_auto, q_auto = choose_best_band(bands, px_small, fs)
    q_auto2, hr_auto = _quality_rr(sig_auto, fs)
    hr_auto_ac = _hr_autocorr(sig_auto, fs)

    q_auto = max(q_auto, q_auto2)

    prefer_rhythm = False
    if hr_pref_ac and hr_auto_ac:
        # if rhythm says slow (<90) but auto says fast (>100), favor rhythm
        if (hr_pref_ac < 90 and hr_auto_ac > 100) or (q_pref >= q_auto + 0.1):
            prefer_rhythm = True
    elif q_pref >= q_auto + 0.1:
        prefer_rhythm = True

    if prefer_rhythm:
        return pref_idx, sig_pref, q_pref
    else:
        return auto_idx, sig_auto, q_auto


# --------------------------------------------------------------------
# ---------------------------- Image route ---------------------------
# --------------------------------------------------------------------
@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        data = await file.read()
        name = (file.filename or "").lower()
        ctype = (file.content_type or "").lower()

        # PNG/JPG first
        img = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_COLOR)

        # PDF fallback
        if img is None and (name.endswith(".pdf") or "pdf" in ctype):
            if not HAS_PYMUPDF:
                return JSONResponse({"error": "PDF not supported on server"}, status_code=400)
            img = pdf_first_page_to_bgr(data)

        if img is None:
            return JSONResponse({"error": "Not an image/PDF"}, status_code=400)

        gray = enhance(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
        px = estimate_grid_px(gray)
        bands = segment_12lead(remove_grid(gray, px))

        # prefer rhythm strip with sanity checks
        idx, sig, qscore = pick_rhythm_strip(bands, px, fs=250)

        # compute two HRs and reconcile
        hr1 = _hr_autocorr(sig, fs=250) or 75.0
        facts_tmp = facts_from_signal(sig, fs=250)
        hr2 = facts_tmp["measurements"]["hr"] or hr1
        hr_used = (min(hr1, hr2) if (hr1 and hr2) else (hr1 or hr2 or 75.0))

        # If outcome is tachy but rhythm-strip autocorr is slow, force rhythm-strip
        try:
            pref_idx = min(11, len(bands) - 1)
            sig_pref, _ = band_to_signal(bands[pref_idx], px, fs=250)
            hr_pref = _hr_autocorr(sig_pref, fs=250)
            if hr_used > 100 and hr_pref and hr_pref < 80:
                idx, sig, qscore = pref_idx, sig_pref, _quality_rr(sig_pref, fs=250)[0]
                facts_tmp = facts_from_signal(sig, fs=250)
        except Exception:
            pass

        facts = facts_tmp
        facts["meta"] = {
            "source": "image",
            "lead_index_used": int(idx),
            "grid_px_per_mm": int(px),
            "quality_score_0to1": float(qscore),
        }
        return JSONResponse({"facts": facts})

    except Exception as e:
        return JSONResponse({"error": "analyze_failed", "detail": str(e)}, status_code=500)


# --------------------------------------------------------------------
# ------------------------------ Root --------------------------------
# --------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "docs": "/docs",
        "routes": [
            "/analyze_wfdb (ZIP of WFDB .hea/.dat)",
            "/analyze_image (PNG/JPG/PDF)",
        ],
    }

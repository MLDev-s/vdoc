# C:\repo_python\vdoc\utils_reglas.py
# -*- coding: utf-8 -*-
"""
Reglas heurísticas (sin IA) para:
- Papel impreso real
- Fotocopias B/N y color
- Arrugas y pliegues
- Detección de pantalla/teléfono en escena
"""
import cv2
import numpy as np
from PIL import Image

# ----------------------------- Parámetros ajustables -----------------------------
TH_COLORFULNESS_BN = 15.0
TH_SAT_MEAN_BN = 0.08
TH_LUM_BINARITY = 0.72

TH_HALFTONE_PEAKS = 8
TH_HALFTONE_RING_RMIN = 0.06
TH_HALFTONE_RING_RMAX = 0.35
TH_HALFTONE_PEAK_REL = 7.5

# Arrugas
TH_WRINKLE_LINES = 500
TH_WRINKLE_VARLAP_PCT = 0.35
TH_WRINKLE_VARLAP_T = 60.0

# Pliegue/doblez fuerte
TH_FOLD_LEN_RATIO = 0.58   # % del lado menor que debe cubrir la línea
TH_FOLD_ANGLE_TOL = 18     # tolerancia en grados para H o V
TH_FOLD_MARGIN = 0.08      # margen para descartar líneas pegadas al borde
# ---------------------------------------------------------------------------------


# ----------------------------- Utilidades básicas -----------------------------
def _to_bgr(image_pil: Image.Image) -> np.ndarray:
    arr = np.array(image_pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _biggest_contour(bw: np.ndarray):
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def _quad_ar_from_cnt(cnt) -> float | None:
    rect = cv2.minAreaRect(cnt)
    (w, h) = rect[1]
    if w == 0 or h == 0:
        return None
    ar = max(w, h) / (min(w, h) + 1e-6)
    return float(ar)


def _band_means(gray: np.ndarray, cnt, t_in=3, t_out=4) -> tuple[float, float]:
    mask = np.zeros_like(gray, np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=1)
    dist = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 3) - cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    band_in = (dist >= 0) & (dist <= t_in)
    band_out = (dist < 0) & (dist >= -t_out)
    mi = float(np.mean(gray[band_in])) if np.any(band_in) else 0.0
    mo = float(np.mean(gray[band_out])) if np.any(band_out) else 0.0
    return mi, mo


def _colorfulness_hasler_susstrunk(img_bgr: np.ndarray) -> float:
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)
    return np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)


def _saturation_mean(img_bgr: np.ndarray) -> tuple[float, float]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    return float(np.mean(s)), float(np.std(s))
# -----------------------------------------------------------------------------


# ----------------------------- Señales extra --------------------------------
def _laminado_gloss_score(gray: np.ndarray) -> float:
    """% de píxeles muy brillantes con bordes duros (plástico/laminado)."""
    thr = max(220, np.percentile(gray, 97))
    mask = (gray >= thr)
    if not np.any(mask):
        return 0.0
    e = cv2.Canny(gray, 80, 160)
    score = float(np.sum(e & mask)) / (gray.size + 1e-6)
    return score * 100.0


def _score_chip_dorado(bgr: np.ndarray) -> float:
    """Puntaje de 'chip dorado' (documentos inteligentes)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((h >= 10) & (h <= 45) & (s >= 60) & (v >= 80)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    H, W = bgr.shape[:2]
    area_img = float(H * W)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c) / (area_img + 1e-6)
    x, y, w, h = cv2.boundingRect(c)
    rect_area = (w * h) / (area_img + 1e-6)
    rectness = (area / (rect_area + 1e-6))
    return 100.0 * area * max(0.0, min(1.0, rectness))
# -----------------------------------------------------------------------------


# ----------------------------- Reglas principales ----------------------------
def es_papel_impreso(pil_image: Image.Image) -> tuple[bool, dict]:
    """True si el contorno principal se comporta como tarjeta/documento físico."""
    bgr = _to_bgr(pil_image)

    H, W = bgr.shape[:2]
    scale = 1024 / max(H, W) if max(H, W) > 1024 else 1.0
    small = cv2.resize(bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)

    gray_full = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.bilateralFilter(gray_full, 7, 50, 50)
    edges = cv2.Canny(gray_blur, 60, 180)
    edges = cv2.dilate(edges, None, 1)

    cnt = _biggest_contour(edges)
    if cnt is None:
        return False, {"reason": "sin_contorno"}

    area_rel = cv2.contourArea(cnt) / float(edges.size)
    if area_rel < 0.15:
        return False, {"reason": "doc_pequenho", "area_rel": area_rel}

    peri = cv2.arcLength(cnt, True)
    approx1 = cv2.approxPolyDP(cnt, 0.01 * peri, True)
    approx2 = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    esquinas_afiladas = (len(approx1) == 4 and len(approx2) == 4)

    mi, mo = _band_means(gray_full, cnt, t_in=3, t_out=5)
    sin_sombra_espesor = (mo >= mi - 1.5)

    band_in = np.zeros_like(gray_full, np.uint8)
    cv2.drawContours(band_in, [cnt], -1, 255, thickness=4)
    band_in = cv2.erode(band_in, None, iterations=1).astype(bool)

    lap_var = float(cv2.Laplacian(gray_full, cv2.CV_64F).var())
    textura_papel = (lap_var >= 180.0 and np.mean(gray_full[band_in]) > 50)

    sob = cv2.Sobel(gray_full, cv2.CV_32F, 1, 1, ksize=3)
    mask_border = np.zeros_like(gray_full, np.uint8)
    cv2.drawContours(mask_border, [cnt], -1, 255, thickness=3)
    borde_vals = np.abs(sob)[mask_border.astype(bool)]
    borde_duro = (float(np.mean(borde_vals)) >= 18.0)

    ar_minrect = _quad_ar_from_cnt(cnt)

    # Señales de plástico/laminado o chip
    gloss = _laminado_gloss_score(gray_full)
    chip_score = _score_chip_dorado(bgr)
    laminado_o_chip = (gloss >= 0.12) or (chip_score >= 0.25)

    señales = int(esquinas_afiladas) + int(sin_sombra_espesor) + int(textura_papel) + int(borde_duro)
    es_papel = (textura_papel and ((esquinas_afiladas and sin_sombra_espesor) or (señales >= 3))) \
               and (not laminado_o_chip)

    dbg = {
        "area_rel": round(float(area_rel), 3),
        "approx1_vertices": int(len(approx1)),
        "approx2_vertices": int(len(approx2)),
        "mi_interior": round(float(mi), 2),
        "mo_exterior": round(float(mo), 2),
        "lap_var": round(float(lap_var), 2),
        "ar_minrect": ar_minrect,
        "esquinas_afiladas": bool(esquinas_afiladas),
        "sin_sombra_espesor": bool(sin_sombra_espesor),
        "textura_papel": bool(textura_papel),
        "borde_duro": bool(borde_duro),
        "señales": int(señales),
        "laminado_gloss_pct": round(float(gloss), 3),
        "chip_score": round(float(chip_score), 3),
        "laminado_o_chip": bool(laminado_o_chip),
    }
    return es_papel, dbg


def _fft_moire_score(gray: np.ndarray) -> float:
    h, w = gray.shape
    s = min(h, w)
    cy, cx = h // 2, w // 2
    r = s // 2
    crop = gray[cy - r: cy + r, cx - r: cx + r]
    crop = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
    f = np.fft.fftshift(np.fft.fft2(crop.astype(np.float32)))
    mag = np.log1p(np.abs(f))
    yy, xx = np.indices(mag.shape)
    rr = np.sqrt((yy - 256) ** 2 + (xx - 256) ** 2) / 256.0
    mask_mid = (rr > 0.08) & (rr < 0.35)
    mid_vals = mag[mask_mid]
    m, s = float(np.mean(mid_vals)), float(np.std(mid_vals))
    thr = m + 3.0 * s
    peaks = (mid_vals > thr).sum()
    score = float(peaks) / 400.0
    return max(0.0, min(score, 1.0))


def _find_bright_rect_with_dark_bezel(bgr: np.ndarray) -> tuple[bool, dict]:
    """Rectángulo brillante con bisel oscuro y uniforme (pantalla)."""
    small = bgr
    max_side = 1280
    h, w = small.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = small.shape[:2]

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=120,
                            minLineLength=int(min(h, w) * 0.35),
                            maxLineGap=18)
    if lines is None or len(lines) < 2:
        return False, {"h_lines": 0, "v_lines": 0}

    horizontales, verticales = [], []
    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = l
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(ang) < 12 or abs(abs(ang) - 180) < 12:
            horizontales.append(l)
        if abs(abs(ang) - 90) < 12:
            verticales.append(l)

    if len(horizontales) < 1 or len(verticales) < 1:
        return False, {"h_lines": len(horizontales), "v_lines": len(verticales)}

    xs, ys = [], []
    for x1, y1, x2, y2 in horizontales + verticales:
        xs += [x1, x2]
        ys += [y1, y2]
    x_min, x_max = max(0, min(xs)), min(w - 1, max(xs))
    y_min, y_max = max(0, min(ys)), min(h - 1, max(ys))
    bw, bh = x_max - x_min, y_max - y_min
    area_rel = (bw * bh) / float(h * w + 1e-6)

    if bw <= 0 or bh <= 0 or area_rel < 0.18:
        return False, {"box": [int(x_min), int(y_min), int(x_max), int(y_max)], "area_rel": round(area_rel, 3)}

    ar = bw / float(bh + 1e-6)
    phone_like = (1.45 <= ar <= 2.70) or (0.37 <= ar <= 0.69)

    pad = int(0.06 * min(bw, bh))
    inner = gray[y_min + pad: y_max - pad, x_min + pad: x_max - pad]
    bezel_top = gray[y_min: y_min + max(1, pad // 2), x_min: x_max]
    bezel_bot = gray[y_max - max(1, pad // 2): y_max, x_min: x_max]
    bezel_left = gray[y_min: y_max, x_min: x_min + max(1, pad // 2)]
    bezel_right = gray[y_min: y_max, x_max - max(1, pad // 2): x_max]

    inner_mean = float(np.mean(inner)) if inner.size else 0.0
    bezel_stack = np.concatenate([bezel_top, bezel_bot, bezel_left, bezel_right])
    bezel_mean = float(np.mean(bezel_stack)) if bezel_stack.size else inner_mean
    bezel_std = float(np.std(bezel_stack)) if bezel_stack.size else 0.0

    strong_bezel = (inner_mean - bezel_mean) > 8.0 and bezel_std < 24.0

    dbg = {
        "box": [int(x_min), int(y_min), int(x_max), int(y_max)],
        "area_rel": round(area_rel, 3),
        "ar_box": round(float(ar), 3),
        "inner_mean": round(inner_mean, 2),
        "bezel_mean": round(bezel_mean, 2),
        "bezel_std": round(bezel_std, 2),
        "h_lines": len(horizontales),
        "v_lines": len(verticales),
        "phone_like_ar": phone_like,
        "strong_bezel": strong_bezel,
    }
    return (phone_like and strong_bezel), dbg


def _detect_device_bezel_global(bgr: np.ndarray) -> tuple[bool, dict]:
    """Bisel oscuro alrededor de toda la imagen (móvil ocupando gran parte)."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    m = max(6, int(0.035 * min(h, w)))
    top = gray[:m, :]
    bot = gray[h - m:, :]
    lef = gray[:, :m]
    rig = gray[:, w - m:]

    inner = gray[m:h - m, m:w - m] if (h > 2 * m and w > 2 * m) else gray
    if inner.size == 0:
        return False, {"reason": "small_image"}

    bezel = np.concatenate([top.flatten(), bot.flatten(), lef.flatten(), rig.flatten()])
    inner_mean = float(np.mean(inner))
    bezel_mean = float(np.mean(bezel))
    bezel_std = float(np.std(bezel))

    bezel_ok = (inner_mean - bezel_mean) > 7.0 and bezel_std < 27.0

    thr = cv2.adaptiveThreshold(inner, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, -6)
    thr = cv2.medianBlur(thr, 5)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ok = False
    dbg = {
        "bezel_mean": round(bezel_mean, 2),
        "inner_mean": round(inner_mean, 2),
        "bezel_std": round(bezel_std, 2),
        "bezel_ok": bool(bezel_ok),
        "screen_box": None,
        "area_rel": 0.0,
        "ar": None,
        "phone_like": False
    }
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, ww, hh = cv2.boundingRect(c)
        x1, y1, x2, y2 = x + m, y + m, x + m + ww, y + m + hh
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        area_rel = (bw * bh) / float(h * w)
        ar = bw / float(bh)
        phone_like = (1.45 <= ar <= 2.70) or (0.37 <= ar <= 0.69)

        dbg.update({
            "screen_box": [int(x1), int(y1), int(x2), int(y2)],
            "area_rel": round(float(area_rel), 3),
            "ar": round(float(ar), 3),
            "phone_like": bool(phone_like),
        })
        ok = bezel_ok and phone_like and (area_rel >= 0.18)

    return ok, dbg


def _detect_edge_margins_phone4(bgr: np.ndarray) -> tuple[bool, dict]:
    """
    Bandas oscuras, uniformes y simétricas en los 4 bordes.
    Dispara si hay (i) 3+ lados buenos, (ii) lados opuestos buenos o (iii) 2 adyacentes.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    m = max(6, int(0.05 * min(h, w)))

    inner = gray[m:h - m, m:w - m] if (h > 2 * m and w > 2 * m) else gray
    inner_mean = float(np.mean(inner))
    inner_std = float(np.std(inner))

    bands = {"L": gray[:, :m], "R": gray[:, w - m:], "T": gray[:m, :], "B": gray[h - m:, :]}
    stats, flags = {}, {}
    for k, a in bands.items():
        mean_k, std_k = float(np.mean(a)), float(np.std(a))
        stats[k] = {"mean": round(mean_k, 2), "std": round(std_k, 2)}
        darker_ok = (inner_mean - mean_k) > max(8.0, 0.25 * inner_std)
        uniform_ok = std_k < 28.0
        flags[k] = bool(darker_ok and uniform_ok)

    num_ok = sum(1 for v in flags.values() if v)
    opp_ok = (flags["L"] and flags["R"]) or (flags["T"] and flags["B"])
    adj_ok = (flags["L"] and flags["T"]) or (flags["T"] and flags["R"]) or (flags["R"] and flags["B"]) or (flags["B"] and flags["L"])
    decision = (num_ok >= 3) or opp_ok or adj_ok
    dbg = {"m": int(m), "inner_mean": round(inner_mean, 2), "inner_std": round(inner_std, 2),
           "flags": flags, "stats": stats, "num_ok": int(num_ok), "opp_ok": bool(opp_ok),
           "adj_ok": bool(adj_ok), "ok": bool(decision)}
    return decision, dbg


def _screen_periodicity_score(gray: np.ndarray) -> float:
    """Razón pico/mediana de picos estrechos a alta frecuencia (rejilla subpíxel)."""
    g = cv2.resize(gray, (768, 768), interpolation=cv2.INTER_AREA).astype(np.float32)
    g -= np.mean(g)
    win = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
    gw = g * win
    F = np.fft.fftshift(np.fft.fft2(gw))
    mag = np.log1p(np.abs(F))

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dy, dx = (Y - cy), (X - cx)
    R = np.sqrt(dy ** 2 + dx ** 2) / (min(cy, cx) + 1e-6)
    theta = (np.degrees(np.arctan2(dy, dx)) % 180.0)

    ring = (R > 0.22) & (R < 0.48)
    ang_mask = (theta < 8) | (theta > 172) | ((theta > 82) & (theta < 98))
    band = ring & ang_mask
    vals = mag[band]
    if vals.size == 0:
        return 0.0

    med = float(np.median(vals))
    top = float(np.percentile(vals, 99.7))
    ratio = (top + 1e-6) / (med + 1e-6)
    return float(ratio)


def es_fotocopia_bn(image_pil: Image.Image) -> tuple[bool, dict]:
    img = _to_bgr(image_pil)
    cf = _colorfulness_hasler_susstrunk(img)
    s_mean, s_std = _saturation_mean(img)
    binarity, bin_dbg = _luminance_binarity(img)
    is_bn = (cf < TH_COLORFULNESS_BN) and (s_mean < TH_SAT_MEAN_BN) and (binarity > TH_LUM_BINARITY)
    dbg = {"colorfulness": float(cf), "sat_mean": float(s_mean), "sat_std": float(s_std),
           "luminance_binarity": float(binarity), **{f"lum_{k}": v for k, v in bin_dbg.items()}}
    return bool(is_bn), dbg


def _channel_misregistration_score(img_bgr: np.ndarray) -> float:
    """Desfase de bordes entre canales; alto = más probable copia a color."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    ch_edges = [cv2.Canny(ch, 60, 180).astype(np.float32) for ch in cv2.split(img_bgr)]
    base = edges.astype(np.float32)
    eps = 1e-6
    scores = []
    for ce in ch_edges:
        num = np.sum(base * ce)
        den = np.sqrt((np.sum(base ** 2) + eps) * (np.sum(ce ** 2) + eps))
        scores.append(num / (den + eps))
    return float(1.0 - np.mean(scores))


def es_fotocopia_color(image_pil: Image.Image) -> tuple[bool, dict]:
    img = _to_bgr(image_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    peaks, fft_dbg = _find_halftone_fft_peaks(gray)
    is_halftone = len(peaks) >= TH_HALFTONE_PEAKS

    fallback = False
    misreg = 0.0
    if not is_halftone:
        misreg = _channel_misregistration_score(img)
        fallback = (misreg >= 0.18)

    dbg = {"fft_peaks": int(len(peaks)), **fft_dbg,
           "rmin": TH_HALFTONE_RING_RMIN, "rmax": TH_HALFTONE_RING_RMAX,
           "peak_gain_rel": TH_HALFTONE_PEAK_REL,
           "misregistration": float(misreg), "fallback_used": bool(fallback)}
    return bool(is_halftone or fallback), dbg


def _find_halftone_fft_peaks(gray: np.ndarray,
                             rmin=TH_HALFTONE_RING_RMIN,
                             rmax=TH_HALFTONE_RING_RMAX,
                             rel_gain=TH_HALFTONE_PEAK_REL):
    g = cv2.resize(gray, (1024, 1024), interpolation=cv2.INTER_AREA)
    f = np.fft.fftshift(np.fft.fft2(g.astype(np.float32)))
    mag = np.log1p(np.abs(f))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    Rn = R / (np.sqrt(cy ** 2 + cx ** 2))
    ring = (Rn >= rmin) & (Rn <= rmax)
    ring_vals = mag[ring]
    ring_med = np.median(ring_vals)
    peaks = np.argwhere((mag > (ring_med * rel_gain)) & ring)
    return peaks, {"ring_med": float(ring_med), "n_peaks": int(len(peaks))}


def _wrinkle_features(gray: np.ndarray):
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 60, 180, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=28, maxLineGap=4)
    n_lines = 0 if lines is None else len(lines)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    win = 32
    high_var_windows = 0
    total_windows = 0
    for y in range(0, h - win + 1, win):
        for x in range(0, w - win + 1, win):
            patch = lap[y:y + win, x:x + win]
            v = float(np.var(patch))
            total_windows += 1
            if v > TH_WRINKLE_VARLAP_T:
                high_var_windows += 1
    pct = 0.0 if total_windows == 0 else high_var_windows / total_windows
    return {"n_lines": int(n_lines), "pct_high_var_lap": float(pct), "total_windows": int(total_windows)}, edges, lines


def es_papel_arrugado(image_pil: Image.Image) -> tuple[bool, dict]:
    img = _to_bgr(image_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    wf, edges, lines = _wrinkle_features(gray)
    is_wrinkled = (wf["n_lines"] >= TH_WRINKLE_LINES) or (wf["pct_high_var_lap"] >= TH_WRINKLE_VARLAP_PCT)
    dbg = {"wrinkle_lines": wf["n_lines"], "wrinkle_pct_high_var": wf["pct_high_var_lap"], "windows": wf["total_windows"]}
    return bool(is_wrinkled), dbg


def es_doblez_fuerte(image_pil: Image.Image) -> tuple[bool, dict]:
    """
    True si hay un pliegue largo que cruza gran parte de la ROI.
    Usa Canny + HoughLinesP y descarta bordes de la tarjeta (líneas en el marco).
    """
    img = _to_bgr(image_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    hp = cv2.GaussianBlur(gray, (0, 0), 1.2)
    realce = cv2.addWeighted(gray, 1.6, hp, -0.6, 0)
    edges = cv2.Canny(realce, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=int(min(h, w) * TH_FOLD_LEN_RATIO),
                            maxLineGap=12)
    found = False
    best = None
    if lines is not None:
        mx = int(TH_FOLD_MARGIN * w)
        my = int(TH_FOLD_MARGIN * h)
        for (x1, y1, x2, y2) in lines[:, 0, :]:
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if (min(ang, abs(180 - ang)) <= TH_FOLD_ANGLE_TOL) or (abs(90 - ang) <= TH_FOLD_ANGLE_TOL):
                if min(x1, x2) > mx and max(x1, x2) < w - mx and min(y1, y2) > my and max(y1, y2) < h - my:
                    found = True
                    best = (x1, y1, x2, y2, ang)
                    break
    dbg = {"found": bool(found), "n_lines": 0 if lines is None else int(len(lines))}
    if best:
        x1, y1, x2, y2, ang = best
        dbg.update({"best": [int(x1), int(y1), int(x2), int(y2)], "angle": float(ang)})
    return found, dbg
# -----------------------------------------------------------------------------


# ----------------------------- Señales complementarias -----------------------
def _detect_black_pillars(bgr: np.ndarray) -> tuple[bool, dict]:
    """
    Detecta 'columnas negras' simétricas (marcos de un celular).
    Requiere luminancia más baja y cromática neutra en ambos lados.
    """
    h, w = bgr.shape[:2]
    m = max(6, int(0.08 * w))
    L = bgr[:, :m, :]
    R = bgr[:, w - m:, :]
    C = bgr[:, m:w - m, :]

    if C.size == 0:
        return False, {"reason": "narrow"}

    gL = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
    gR = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    gC = cv2.cvtColor(C, cv2.COLOR_BGR2GRAY)
    meanC = float(np.mean(gC))
    meanL, stdL = float(np.mean(gL)), float(np.std(gL))
    meanR, stdR = float(np.mean(gR)), float(np.std(gR))

    sL, _ = _saturation_mean(L)
    sR, _ = _saturation_mean(R)
    cfL = _colorfulness_hasler_susstrunk(L)
    cfR = _colorfulness_hasler_susstrunk(R)

    darker_ok = (meanC - meanL > 12.0) and (meanC - meanR > 12.0)
    uniform_ok = (stdL < 25.0) and (stdR < 25.0)
    chroma_ok = (sL <= 0.16 and sR <= 0.16 and cfL <= 14.0 and cfR <= 14.0)
    symmetry = (abs(meanL - meanR) < 10.0) and (abs(stdL - stdR) < 10.0)

    ok = darker_ok and uniform_ok and chroma_ok and symmetry
    dbg = {"m": int(m), "meanC": round(meanC, 2), "meanL": round(meanL, 2), "meanR": round(meanR, 2),
           "stdL": round(stdL, 2), "stdR": round(stdR, 2),
           "sL": round(float(sL), 3), "sR": round(float(sR), 3),
           "cfL": round(float(cfL), 2), "cfR": round(float(cfR), 2), "ok": bool(ok)}
    return ok, dbg


def _detect_glare_spot(gray: np.ndarray) -> tuple[bool, dict]:
    """Reflejo de vidrio (hotspot brillante aprox. circular)."""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = max(220, int(np.percentile(g, 97)))
    mask = (g >= thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, {"spots": 0}
    h, w = gray.shape[:2]
    area_img = float(h * w)
    c = max(cnts, key=cv2.contourArea)
    A = cv2.contourArea(c) / (area_img + 1e-6)
    if A < 0.002 or A > 0.06:
        return False, {"spots": len(cnts), "area": A}
    P = cv2.arcLength(c, True) + 1e-6
    circularity = 4.0 * np.pi * (cv2.contourArea(c) + 1e-6) / (P * P)
    ok = (circularity >= 0.55)
    return bool(ok), {"spots": len(cnts), "area": float(A), "circularity": float(circularity), "ok": bool(ok)}
# -----------------------------------------------------------------------------


def detectar_pantalla_en_escena(pil_img: Image.Image, moire_thr: float = 0.06) -> tuple[bool, dict]:
    ...
    pantalla = bool(global_ok or local_ok or edge_ok or grid_ok or pillars_ok or glare_ok) or \
               (moire_score >= moire_thr and area_cond and phone_like)
    dbg = {
        "global": global_dbg,
        "local": local_dbg,
        "edge_margins4": edge_dbg,
        "pillars": pillars_dbg,
        "glare": glare_dbg,
        "pillars_ok": bool(pillars_ok),
        "glare_ok": bool(glare_ok),
        "grid_ratio": round(float(grid_ratio), 2),
        "grid_ok": bool(grid_ok),
        "moire_score": round(float(moire_score), 3),
        "area_cond": bool(area_cond),
        "phone_like": bool(phone_like),
        "pantalla": bool(pantalla),
    }
    return pantalla, dbg


def _luminance_binarity(img_bgr: np.ndarray) -> tuple[float, dict]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32) / 255.0
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binmask = (gray >= thr).astype(np.float32)
    p1 = binmask.mean()
    p0 = 1.0 - p1
    eps = 1e-6
    H = -(p0 * np.log2(p0 + eps) + p1 * np.log2(p1 + eps))
    H_norm = H / 1.0
    binarity = 1.0 - H_norm
    contrast = float(np.std(g))
    return float(binarity * (1.0 + 0.5 * contrast)), {"thr_otsu": float(thr), "p_white": float(p1), "contrast": contrast}
# ----------------------------------------------------------------------------- 

# C:\repo_python\vdoc\utils_reglas.py
import cv2
import numpy as np
from PIL import Image

# ------- Reglas ya existentes -------
# (conserva tus funciones de papel)
def es_papel_impreso(pil_image: Image.Image):
    """
    Heurística robusta para COPIA EN PAPEL de una tarjeta ID‑1.
    Devuelve (es_papel: bool, debug: dict).

    Señales que sumamos:
      - esquinas_afiladas: approxPolyDP con eps 1% y 2% siguen dando 4 vértices.
      - sin_sombra_espesor: la franja exterior NO es claramente más oscura que el interior.
      - textura_papel: varianza de Laplaciano alta en banda interior (arrugas/impresiones).
      - borde_duro: transición muy abrupta de brillo (borde “afilado” típico de papel).
    Decisión:
      papel = (esquinas_afiladas AND sin_sombra_espesor) OR (señales >= 2)
    """
    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    scale = 1024 / max(H, W) if max(H, W) > 1024 else 1.0
    small = cv2.resize(bgr, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(gray_blur, 60, 180)
    edges = cv2.dilate(edges, None, 1)

    cnt = _biggest_contour(edges)
    if cnt is None:
        return (False, {"reason": "sin_contorno"})

    area_rel = cv2.contourArea(cnt) / float(edges.size)
    if area_rel < 0.15:
        return (False, {"reason": "doc_pequenho", "area_rel": area_rel})

    # --- Señal 1: esquinas afiladas (pocas muestras para esquinas redondeadas reales)
    peri = cv2.arcLength(cnt, True)
    approx1 = cv2.approxPolyDP(cnt, 0.01 * peri, True)  # 1%
    approx2 = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # 2%
    esquinas_afiladas = (len(approx1) == 4 and len(approx2) == 4)

    # --- Bandas interior / exterior respecto al contorno
    gray_full = gray  # ya escalado
    mi, mo = _band_means(gray_full, cnt, t_in=3, t_out=5)

    # --- Señal 2: sin sombra/espesor (más estricto: exterior no es >= interior - 3)
    # En tarjetas reales suele haber caída de luminancia (exterior más oscuro).
    sin_sombra_espesor = (mo >= mi - 3.0)

    # --- Señal 3: textura de papel (arrugas finas / impresión)
    band_in = np.zeros_like(gray_full, np.uint8)
    cv2.drawContours(band_in, [cnt], -1, 255, thickness=4)
    band_in = cv2.erode(band_in, None, iterations=1).astype(bool)
    lap_var = float(cv2.Laplacian(gray_full, cv2.CV_64F).var())
    textura_papel = (lap_var >= 220.0 and np.mean(gray_full[band_in]) > 50)

    # --- Señal 4: borde duro (gradiente muy alto en una franja estrecha alrededor del borde)
    # Medimos la diferencia media |grad| entre interior cercano y exterior cercano.
    sob = cv2.Sobel(gray_full, cv2.CV_32F, 1, 1, ksize=3)
    mask_border = np.zeros_like(gray_full, np.uint8)
    cv2.drawContours(mask_border, [cnt], -1, 255, thickness=3)  # banda ~3px alrededor del borde
    borde_vals = np.abs(sob)[mask_border.astype(bool)]
    borde_duro = (float(np.mean(borde_vals)) >= 18.0)

    # AR aproximada (sólo para debug)
    ar_minrect = _quad_ar_from_cnt(cnt)

    # --- Decisión
    señales = int(esquinas_afiladas) + int(sin_sombra_espesor) + int(textura_papel) + int(borde_duro)
    es_papel = (esquinas_afiladas and sin_sombra_espesor) or (señales >= 2)

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
        "señales": int(señales)
    }
    return (es_papel, dbg)


# ------- Detección de pantalla en escena (ajustada para evitar falsos) -------
def _to_bgr(pil_img: Image.Image):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def _fft_moire_score(gray):
    h, w = gray.shape
    s = min(h, w); cy, cx = h // 2, w // 2; r = s // 2
    crop = gray[cy - r: cy + r, cx - r: cx + r]
    crop = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
    f = np.fft.fftshift(np.fft.fft2(crop.astype(np.float32)))
    mag = np.log1p(np.abs(f))
    yy, xx = np.indices(mag.shape)
    rr = np.sqrt((yy - 256)**2 + (xx - 256)**2) / 256.0
    mask_mid = (rr > 0.08) & (rr < 0.35)
    mid_vals = mag[mask_mid]
    m, s = float(np.mean(mid_vals)), float(np.std(mid_vals))
    thr = m + 3.0 * s
    peaks = (mid_vals > thr).sum()
    score = float(peaks) / 400.0
    return max(0.0, min(score, 1.0))

def _find_bright_rect_with_dark_bezel(bgr):
    """
    Busca un rectángulo grande tipo pantalla (bisel oscuro + interior más claro).
    Más permisivo con longitud de líneas y con el contraste interior-bisel.
    """
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

    # ↓ Bajamos el requisito de línea larga (0.45) y permitimos 1H+1V
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=120,
                            minLineLength=int(min(h, w) * 0.45),
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

    # Caja de las líneas
    xs, ys = [], []
    for x1, y1, x2, y2 in horizontales + verticales:
        xs += [x1, x2]; ys += [y1, y2]
    x_min, x_max = max(0, min(xs)), min(w - 1, max(xs))
    y_min, y_max = max(0, min(ys)), min(h - 1, max(ys))
    bw, bh = x_max - x_min, y_max - y_min
    area_rel = (bw * bh) / float(h * w + 1e-6)

    # Requerimos que ocupe parte importante de la imagen
    if bw <= 0 or bh <= 0 or area_rel < 0.28:
        return False, {"box": [int(x_min), int(y_min), int(x_max), int(y_max)], "area_rel": round(area_rel, 3)}

    ar = bw / float(bh + 1e-6)
    phone_like = (1.55 <= ar <= 2.45) or (0.41 <= ar <= 0.64)  # 16:9..20:9 (o en landscape)

    # Promedios interior vs. bisel; añadimos desviación en bisel (uniformidad)
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

    # ↓ Umbral más realista (12) y chequeo de bisel relativamente uniforme
    strong_bezel = (inner_mean - bezel_mean) > 12.0 and bezel_std < 22.0

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


def _detect_device_bezel_global(bgr):
    """
    Detecta 'dispositivo' mirando la imagen completa:
      - comprueba si hay una banda oscura (bisel) pegada a 2–4 bordes de la imagen,
      - encuentra un rectángulo interior notablemente más brillante (la pantalla),
      - estima AR (~16:9..20:9 u horizontal equivalente) y área relativa.
    Devuelve (ok: bool, dbg: dict).
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Bandas cercanas al borde (bisel candidato)
    m = max(6, int(0.035 * min(h, w)))  # ~3.5% de la dimensión menor
    top = gray[:m, :]
    bot = gray[h-m:, :]
    lef = gray[:, :m]
    rig = gray[:, w-m:]

    # Brillo promedio en bisel vs interior central
    inner = gray[m:h-m, m:w-m] if (h > 2*m and w > 2*m) else gray
    if inner.size == 0:
        return False, {"reason": "small_image"}

    bezel = np.concatenate([top.flatten(), bot.flatten(), lef.flatten(), rig.flatten()])
    inner_mean = float(np.mean(inner))
    bezel_mean = float(np.mean(bezel))
    bezel_std  = float(np.std(bezel))

    # Requerimos bisel más oscuro y relativamente uniforme
    bezel_ok = (inner_mean - bezel_mean) > 10.0 and bezel_std < 25.0

    # Buscamos rectángulo interior brillante (pantalla)
    # Usamos umbral adaptativo y cerramos orificios
    thr = cv2.adaptiveThreshold(inner, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, -6)
    thr = cv2.medianBlur(thr, 5)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=1)

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
        # Mapea a coords globales
        x1, y1, x2, y2 = x + m, y + m, x + m + ww, y + m + hh
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        area_rel = (bw * bh) / float(h * w)
        ar = bw / float(bh)
        phone_like = (1.55 <= ar <= 2.45) or (0.41 <= ar <= 0.64)

        dbg.update({
            "screen_box": [int(x1), int(y1), int(x2), int(y2)],
            "area_rel": round(float(area_rel), 3),
            "ar": round(float(ar), 3),
            "phone_like": bool(phone_like),
        })
        ok = bezel_ok and phone_like and (area_rel >= 0.28)

    return ok, dbg


def detectar_pantalla_en_escena(pil_img: Image.Image, moire_thr: float = 0.10):
    """
    Devuelve (pantalla_en_escena: bool, debug: dict)

    True si se cumple cualquiera:
      A) Bisel global convincente + pantalla interior grande con AR de teléfono
      B) Bisel local por Hough (versión previa)              (más robusto a recortes)
      C) Moiré >= umbral + caja grande con AR de teléfono   (pantalla fotografiada)
    """
    bgr = _to_bgr(pil_img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # A) Bisel global (nuevo)
    global_ok, global_dbg = _detect_device_bezel_global(bgr)

    # B) Bisel local con Hough (ya lo teníamos)
    local_ok, local_dbg = _find_bright_rect_with_dark_bezel(bgr)

    # C) Moiré + caja grande tipo teléfono
    moire_score = _fft_moire_score(gray)
    area_cond, phone_like = False, False
    if not (global_ok or local_ok):
        # Estimar caja grande con Canny simple
        edges = cv2.Canny(gray, 60, 180)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            h, w = gray.shape
            c = max(cnts, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(c)
            area_rel = (bw * bh) / float(h * w)
            ar = bw / float(bh + 1e-6)
            phone_like = (1.55 <= ar <= 2.45) or (0.41 <= ar <= 0.64)
            area_cond = area_rel >= 0.28

    pantalla = bool(global_ok or local_ok) or (moire_score >= moire_thr and area_cond and phone_like)

    dbg = {
        "global": global_dbg,
        "local": local_dbg,
        "moire_score": round(float(moire_score), 3),
        "area_cond": bool(area_cond),
        "phone_like": bool(phone_like),
        "pantalla": bool(pantalla),
    }
    return pantalla, dbg

def _biggest_contour(bw):
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    return max(cnts, key=cv2.contourArea)

def _quad_ar_from_cnt(cnt):
    rect = cv2.minAreaRect(cnt)
    (w, h) = rect[1]
    if w == 0 or h == 0: return None
    ar = max(w, h) / (min(w, h) + 1e-6)
    return float(ar)

def _band_means(gray, cnt, t_in=3, t_out=4):
    # máscara del contorno
    mask = np.zeros_like(gray, np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=1)
    # distancia signed: dentro (+) / fuera (-)
    dist = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 3) - cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    band_in = (dist >= 0) & (dist <= t_in)
    band_out = (dist < 0) & (dist >= -t_out)
    mi = float(np.mean(gray[band_in])) if np.any(band_in) else 0.0
    mo = float(np.mean(gray[band_out])) if np.any(band_out) else 0.0
    return mi, mo
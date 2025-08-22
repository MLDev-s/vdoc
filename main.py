# C:\repo_python\vdoc\main.py
import logging, traceback
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vdoc")

import os
import json
from io import BytesIO

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from utils import preprocess_image, predict, ID1_AR  # <- devuelve también la ROI
from utils_reglas import (
    es_papel_impreso, detectar_pantalla_en_escena,
    es_fotocopia_bn, es_fotocopia_color, es_papel_arrugado,
    _colorfulness_hasler_susstrunk, _saturation_mean,  # para COLOR_GUARD
    es_doblez_fuerte
)

# ================== VARIABLES ESTRATÉGICAS (tuneables) ==================
BORDE_MARGIN = 0.05
RECHAZO_PANTALLA_SIN_LOCALIZACION = 0.95
MIN_LOC_CONF = 0.60
GATE_REGLAS_A_ROI = True
GATE_ARRUGAS_SOLO_SI_PAPEL = True
COLOR_GUARD_ARRUGAS = dict(min_sat_mean=0.12, min_colorfulness=18.0)
# ------------------------------------------------------------------------
# Sensibilidad de “pantalla en escena” y chequeos locales de bisel
PHONE_AREA_MIN = 0.18
PHONE_BEZEL_DIFF = 8.0
PHONE_BEZEL_STD_MAX = 28.0
PHONE_AR_OK = (1.45, 2.70)
# ========================================================================

DOC_OVERRIDE_THR = 0.985
MOIRE_STRONG_THR = 0.08  # más sensible

# --- guards cromáticos para distinguir bisel real de fondo coloreado ---
_BEZEL_SAT_MAX = 0.16
_BEZEL_COLORFULNESS_MAX = 14.0

# Umbrales para glare (reflejo de vidrio) usados en _roi_dentro_de_pantalla
GLARE_STRONG_THR = 0.006        # >= 0.6% del área expandida son hotspots
GLARE_EDGE_RATIO_THR = 0.10     # >= 10% de bordes caen en el hotspot

app = FastAPI()
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# ----------------------------- Umbral dinámico -----------------------------
def cargar_umbral(path: str = "umbral.json", default: float = 0.52) -> float:
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            thr = float(data.get("umbral_confianza", default))
            if 0.0 < thr < 1.0:
                return thr
    except Exception as e:
        print(f"[WARN] No se pudo leer umbral desde {path}: {e}")
    return default

UMBRAL_CONFIANZA = cargar_umbral()
print(f"[INFO] UMBRAL_CONFIANZA activo: {UMBRAL_CONFIANZA:.2f}")

# ----------------------------- Utilidades ----------------------------------
def to_jsonable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj

def _np_rgb(pil_img: Image.Image):
    return np.array(pil_img.convert("RGB"))

def _ar_ok(ar: float) -> bool:
    lo, hi = PHONE_AR_OK
    return (lo <= ar <= hi) or (lo <= (1.0 / ar) <= hi)

def _sat_and_colorfulness_from_bgr(bgr_region: np.ndarray) -> tuple[float, float]:
    if bgr_region.ndim != 3 or bgr_region.shape[2] != 3 or bgr_region.size == 0:
        return 0.0, 0.0
    h, w = bgr_region.shape[:2]
    if min(h, w) < 3:
        scale = int(np.ceil(3.0 / max(1, min(h, w))))
        bgr_region = cv2.resize(bgr_region, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    sat_mean, _ = _saturation_mean(bgr_region)
    colorfulness = _colorfulness_hasler_susstrunk(bgr_region)
    return float(sat_mean), float(colorfulness)

def _roi_dentro_de_pantalla(pil_img: Image.Image, dbg_pre):
    """
    Evidencia SUAVE de ROI dentro de pantalla; no provoca rechazo por sí sola.
    Añade 'glare' (reflejos) como pista típica de pantallas.
    """
    def _glare_metrics(gray_box: np.ndarray) -> tuple[float, float]:
        if gray_box.size == 0:
            return 0.0, 0.0
        h, w = gray_box.shape[:2]
        if max(h, w) < 256:
            scale = int(np.ceil(256 / max(1, max(h, w))))
            gray_box = cv2.resize(gray_box, (w*scale, h*scale), interpolation=cv2.INTER_AREA)
        thr = max(225, int(np.percentile(gray_box, 96)))
        bright = (gray_box >= thr).astype(np.uint8) * 255
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        area_ratio = float(np.count_nonzero(bright)) / (bright.size + 1e-6)
        edges = cv2.Canny(gray_box, 80, 160)
        e_in = np.sum((edges > 0) & (bright > 0))
        e_out = np.sum(edges > 0) + 1e-6
        edge_ratio = float(e_in) / float(e_out)
        return area_ratio, edge_ratio

    try:
        src_quad = dbg_pre.get("src_quad")
        if not src_quad:
            return False, {"reason": "sin_src_quad"}

        img = _np_rgb(pil_img)
        bgr_full = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray_full = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2GRAY)
        h, w = gray_full.shape[:2]
        q = np.array(src_quad, dtype=np.float32)

        x1, y1 = np.min(q, axis=0); x2, y2 = np.max(q, axis=0)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        expansions = [0.10, 0.18, 0.28, 0.40, 0.55, 0.70, 0.85]

        best_dbg = None
        for ex in expansions:
            mx = int(ex * (x2 - x1)); my = int(ex * (y2 - y1))
            X1 = max(0, x1 - mx); Y1 = max(0, y1 - my)
            X2 = min(w - 1, x2 + mx); Y2 = min(h - 1, y2 + my)
            BW, BH = X2 - X1, Y2 - Y1
            if BW <= 0 or BH <= 0:
                continue

            area_rel = (BW * BH) / float(h * w)
            ar = BW / float(BH + 1e-6)
            if area_rel < PHONE_AREA_MIN or not _ar_ok(ar):
                best_dbg = {"reason": "area/ar_no_ok", "area_rel": round(area_rel,3), "ar": round(ar,3)}
                continue

            gray = gray_full[Y1:Y2, X1:X2]
            if gray.size == 0:
                continue

            s = max(2, int(0.06 * min(BW, BH)))
            if (BW <= 2 * s) or (BH <= 2 * s):
                continue

            inner = gray[s:BH - s, s:BW - s]
            if inner.size == 0:
                continue
            inner_mean = float(np.mean(inner))

            L = gray[:, :s]; R = gray[:, BW - s:BW]
            T = gray[:s, :]; B = gray[BH - s:BH, :]

            def stats(a):
                return float(np.mean(a)), float(np.std(a))
            mL, sdL = stats(L); mR, sdR = stats(R)
            mT, sdT = stats(T); mB, sdB = stats(B)
            dL = inner_mean - mL; dR = inner_mean - mR
            dT = inner_mean - mT; dB = inner_mean - mB

            # guardas cromáticas (evita confundir “madera” con bisel)
            bezel_bgr = bgr_full[Y1:Y2, X1:X2]
            pad = max(1, s // 2)
            bezel_stack = np.vstack([
                bezel_bgr[:pad, :, :].reshape(-1, 3),
                bezel_bgr[-pad:, :, :].reshape(-1, 3),
                bezel_bgr[:, :pad, :].reshape(-1, 3),
                bezel_bgr[:, -pad:, :].reshape(-1, 3),
            ]).reshape(-1, 1, 3)
            bezel_patch = bezel_stack.reshape(max(2, pad * 2), -1, 3)
            sat_mean, colorfulness = _sat_and_colorfulness_from_bgr(bezel_patch.astype(np.uint8))
            chroma_ok = (sat_mean <= _BEZEL_SAT_MAX) and (colorfulness <= _BEZEL_COLORFULNESS_MAX)

            left_ok  = (dL > PHONE_BEZEL_DIFF) and (sdL < PHONE_BEZEL_STD_MAX)
            right_ok = (dR > PHONE_BEZEL_DIFF) and (sdR < PHONE_BEZEL_STD_MAX)
            top_ok   = (dT > PHONE_BEZEL_DIFF)  and (sdT < PHONE_BEZEL_STD_MAX)
            bot_ok   = (dB > PHONE_BEZEL_DIFF)  and (sdB < PHONE_BEZEL_STD_MAX)

            ok_sides = int(left_ok) + int(right_ok) + int(top_ok) + int(bot_ok)
            lr_pair = left_ok and right_ok
            tb_pair = top_ok and bot_ok
            adj_pair = (left_ok and top_ok) or (top_ok and right_ok) or (right_ok and bot_ok) or (bot_ok and left_ok)

            # === NUEVO: glare en la caja expandida (propio de pantallas) ===
            glare_area, glare_edge_ratio = _glare_metrics(gray)
            glare_ok = (glare_area >= GLARE_STRONG_THR) and (glare_edge_ratio >= GLARE_EDGE_RATIO_THR)

            dbg_here = {
                "outer_box": [int(X1), int(Y1), int(X2), int(Y2)],
                "expansion": float(ex),
                "area_rel": round(area_rel, 3),
                "ar_outer": round(ar, 3),
                "inner_mean": round(inner_mean, 2),
                "side_means": {"L": round(mL, 2), "R": round(mR, 2), "T": round(mT, 2), "B": round(mB, 2)},
                "side_stds":  {"L": round(sdL, 2), "R": round(sdR, 2), "T": round(sdT, 2), "B": round(sdB, 2)},
                "deltas":     {"L": round(dL, 2), "R": round(dR, 2), "T": round(dT, 2), "B": round(dB, 2)},
                "ok_sides": ok_sides,
                "bezel_sat_mean": round(sat_mean, 3),
                "bezel_colorfulness": round(colorfulness, 2),
                "chroma_ok": bool(chroma_ok),
                "glare_area": round(glare_area, 4),
                "glare_edge_ratio": round(glare_edge_ratio, 2),
                "glare_ok": bool(glare_ok),
            }

            # cuenta si hay pares/3+ lados Y el bisel es neutro,
            # …o si el glare es fuerte (muy típico de pantallas)
            if (chroma_ok and (lr_pair or tb_pair or adj_pair or ok_sides >= 3)) or glare_ok:
                if chroma_ok and (lr_pair or tb_pair or adj_pair or ok_sides >= 3):
                    dbg_here["pairs"] = "LR" if lr_pair else ("TB" if tb_pair else ("ADJ" if adj_pair else "3+ lados"))
                return True, dbg_here

            best_dbg = dbg_here

        return False, (best_dbg or {"reason": "sin_evidencia_bisel"})
    except Exception as e:
        return False, {"error": f"roi_dentro_de_pantalla_failed: {e}"}

# --------------------------------- API --------------------------------------
@app.post("/verificar-documento")
async def verificar_documento(file: UploadFile = File(...)):
    try:
        # 1) Validación de extensión
        ext = (file.filename or "").split(".")[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return {"resultado": "error", "status": "bad_request",
                    "razon": "Formato no permitido. Usa JPG o PNG."}

        # 2) Cargar imagen
        image_bytes = await file.read()
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return {"resultado": "error", "status": "bad_request",
                    "razon": "Imagen inválida", "detalle": str(e)}

        # 3) Normalización de orientación
        if image.width > image.height:
            image = image.rotate(270, expand=True)

        # --- Heurística ANTI-COPIA EN PAPEL (full image) ---
        try:
            h_is_papel, h_dbg = es_papel_impreso(image)
        except Exception as e:
            h_is_papel, h_dbg = False, {"error": f"heuristica_papel_failed: {e}"}

        # 4) Preprocesado → ROI + dbg
        ret = preprocess_image(image)
        if isinstance(ret, tuple) and len(ret) == 3:
            preprocessed, dbg, roi_pil = ret
        elif isinstance(ret, tuple) and len(ret) == 2:
            preprocessed, dbg = ret; roi_pil = image
        else:
            preprocessed = ret
            dbg = {"localized": None, "ar_after_warp": None, "loc_conf": 0.0}
            roi_pil = image

        # Chequeo local suave: ROI embebida en un teléfono
        roi_en_pantalla, dbg_roi_screen = _roi_dentro_de_pantalla(image, dbg)

        # --- Gates básicos ---
        if dbg.get("localized") is False:
            log.info("salida=manual:no_localized")
            return {"resultado": "indeterminado", "confianza": 0.0,
                    "status": "verificación manual necesaria",
                    "razon": "No se pudo localizar el documento.",
                    "manual": True, "debug": {"preprocess": dbg, "heuristica_papel_fullimg": h_dbg}}

        if dbg.get("ar_after_warp") is not None and abs(float(dbg["ar_after_warp"]) - ID1_AR) > 0.18:
            log.info("salida=manual:ar_anomala")
            return {"resultado": "indeterminado", "confianza": 0.0,
                    "status": "verificación manual necesaria",
                    "razon": "Relación de aspecto anómala tras la rectificación.",
                    "manual": True, "debug": {"preprocess": dbg, "heuristica_papel_fullimg": h_dbg}}

        # === Reglas SOLO en ROI ===
        reglas_dbg = {}
        usar_roi = bool(GATE_REGLAS_A_ROI and dbg.get("localized") and (dbg.get("loc_conf", 0.0) >= MIN_LOC_CONF))
        img_reglas = roi_pil if usar_roi else image

        # Papel impreso (ROI)
        try:
            roi_is_papel, roi_papel_dbg = es_papel_impreso(img_reglas)
        except Exception as e:
            roi_is_papel, roi_papel_dbg = False, {"error": f"heuristica_papel_roi_failed: {e}"}

        if h_is_papel and roi_is_papel:
            log.info("salida=rechazo:falso_impreso(papel)")
            return {"resultado": "falso_impreso", "confianza": 1.0,
                    "status": "rechazado: documento impreso/copiado en papel",
                    "razon": "Esquinas afiladas y textura de papel en ROI.",
                    "manual": False,
                    "debug": {"heuristica_papel_fullimg": h_dbg, "heuristica_papel_roi": roi_papel_dbg, "preprocess": dbg}}

        # Fotocopia B/N
        try:
            is_bn, dbg_bn = es_fotocopia_bn(img_reglas)
        except Exception as e:
            is_bn, dbg_bn = False, {"error": f"fotocopia_bn_failed: {e}"}
        reglas_dbg["fotocopia_bn"] = dbg_bn
        if is_bn and roi_is_papel:
            log.info("salida=rechazo:fotocopia_bn")
            return {"resultado": "falso_impreso", "confianza": 1.0,
                    "status": "rechazado: fotocopia en blanco y negro",
                    "razon": "Saturación y colorfulness muy bajos en la ROI.",
                    "manual": False,
                    "debug": {"preprocess": dbg, "heuristica_papel_roi": roi_papel_dbg, **reglas_dbg}}

        # Fotocopia color
        try:
            is_col, dbg_col = es_fotocopia_color(img_reglas)
        except Exception as e:
            is_col, dbg_col = False, {"error": f"fotocopia_color_failed: {e}"}
        reglas_dbg["fotocopia_color"] = dbg_col
        if is_col and roi_is_papel:
            log.info("salida=rechazo:fotocopia_color")
            return {"resultado": "falso_impreso", "confianza": 1.0,
                    "status": "rechazado: fotocopia a color",
                    "razon": "Patrón de semitonos en la ROI.",
                    "manual": False,
                    "debug": {"preprocess": dbg, "heuristica_papel_roi": roi_papel_dbg, **reglas_dbg}}

        # Arrugas (con color-guard)
        try:
            arr_bgr = np.array(img_reglas.convert("RGB"))[:, :, ::-1]
            sat_mean, _ = _saturation_mean(arr_bgr)
            colorfulness = _colorfulness_hasler_susstrunk(arr_bgr)
        except Exception:
            sat_mean, colorfulness = 0.0, 0.0

        aplicar_arrugas = True
        if GATE_ARRUGAS_SOLO_SI_PAPEL and not roi_is_papel:
            aplicar_arrugas = False
        if colorfulness >= COLOR_GUARD_ARRUGAS["min_colorfulness"] or sat_mean >= COLOR_GUARD_ARRUGAS["min_sat_mean"]:
            aplicar_arrugas = False

        if usar_roi and aplicar_arrugas:
            try:
                is_wr, dbg_wr = es_papel_arrugado(img_reglas)
            except Exception as e:
                is_wr, dbg_wr = False, {"error": f"papel_arrugado_failed: {e}"}
            reglas_dbg["papel_arrugado"] = dbg_wr
            if is_wr:
                log.info("salida=rechazo:arrugas")
                return {"resultado": "falso_impreso", "confianza": 1.0,
                        "status": "rechazado: copia en papel arrugado",
                        "razon": "Pliegues/arrugas consistentes en la ROI.",
                        "manual": False,
                        "debug": {"preprocess": dbg, "heuristica_papel_roi": roi_papel_dbg, **reglas_dbg}}

        try:
            es_fold, dbg_fold = es_doblez_fuerte(img_reglas)
        except Exception as e:
            es_fold, dbg_fold = False, {"error": f"doblez_failed: {e}"}
        reglas_dbg["doblez"] = dbg_fold
        if es_fold:
            return {"resultado": "falso_impreso", "confianza": 1.0,
                    "status": "rechazado: copia en papel con pliegue",
                    "razon": "Se detectó un pliegue/‘doblez’ largo en la ROI.",
                    "manual": False,
                    "debug": {"preprocess": dbg, **reglas_dbg}}

        # 5) Pantalla/teléfono en escena (imagen completa)
        try:
            pantalla_en_escena, dbg_screen = detectar_pantalla_en_escena(image)
        except Exception:
            pantalla_en_escena, dbg_screen = False, {}

        ROI_FUERTE = bool(dbg.get("localized")) and (dbg.get("loc_conf", 0.0) >= 0.68) \
            and (dbg.get("ar_after_warp") is not None) \
            and (abs(float(dbg["ar_after_warp"]) - ID1_AR) <= 0.12)

        # 6) Modelo
        clase_pred, conf_pred = predict(preprocessed)
        if clase_pred == "pantalla":
            prob_pantalla = float(conf_pred); prob_doc = 1.0 - prob_pantalla
        else:
            prob_doc = float(conf_pred); prob_pantalla = 1.0 - prob_doc

        # 6.a) Sin ROI fiable + modelo dice pantalla: manual
        if (dbg.get("localized") is None or dbg.get("loc_conf", 0.0) < MIN_LOC_CONF) and \
           (prob_pantalla >= UMBRAL_CONFIANZA) and (prob_pantalla < RECHAZO_PANTALLA_SIN_LOCALIZACION):
            log.info("salida=manual:modelo_pantalla_roi_debil")
            return {"resultado": "indeterminado", "confianza": round(prob_pantalla, 4),
                    "status": "verificación manual necesaria",
                    "razon": "El modelo sugiere 'pantalla' pero la ROI no es fiable.",
                    "manual": True,
                    "debug": {"preprocess": dbg, "heuristica_papel_fullimg": h_dbg,
                              "pantalla_escena": dbg_screen,
                              "prob_doc": round(prob_doc, 4), "prob_pantalla": round(prob_pantalla, 4),
                              "reglas": reglas_dbg}}

        # ---- Votación por señales de pantalla (aunque el flag global no sea True)
        g = dbg_screen.get("global", {}) or {}
        l = dbg_screen.get("local", {}) or {}
        e4 = dbg_screen.get("edge_margins4", {}) or dbg_screen.get("edge_margins", {}) or {}
        grid_ok = bool(dbg_screen.get("grid_ok", False))
        moire_ok = (float(dbg_screen.get("moire_score", 0.0)) >= MOIRE_STRONG_THR) and \
                   (bool(dbg_screen.get("area_cond")) or bool(dbg_screen.get("phone_like")))
        pillars_ok = bool(dbg_screen.get("pillars_ok", False))
        glare_ok   = bool(dbg_screen.get("glare_ok", False))

        any_signal = bool(
            pantalla_en_escena or roi_en_pantalla or pillars_ok or glare_ok or grid_ok or
            e4.get("ok", False) or (g.get("bezel_ok") and g.get("phone_like"))
        )

        # 6.b) Pantalla en escena (regla + evidencia de ROI) – versión robusta
        if any_signal:
            votes = int(bool(g.get("bezel_ok") and g.get("phone_like"))) + \
                    int((l.get("h_lines", 0) >= 1 and l.get("v_lines", 0) >= 1)) + \
                    int(bool(e4.get("ok", False))) + int(grid_ok) + int(moire_ok) + \
                    int(bool(roi_en_pantalla))
            votes += int(pillars_ok) + int(glare_ok)  # dos votos extra

            need_votes = 2 if ROI_FUERTE else 1
            weak_roi = not ROI_FUERTE

            # Motivos para debug
            reasons = []
            if g.get("bezel_ok") and g.get("phone_like"): reasons.append("bisel_oscuro_global")
            if (l.get("h_lines", 0) >= 1 and l.get("v_lines", 0) >= 1): reasons.append("recta_H+V (Hough)")
            if e4.get("ok", False): reasons.append("bandas_borde_simétricas")
            if grid_ok: reasons.append("patrón_subpíxel/FFT")
            if moire_ok: reasons.append("moiré/FFT")
            if pillars_ok: reasons.append("columnas_negras")
            if glare_ok: reasons.append("glare_de_vidrio")
            if roi_en_pantalla: reasons.append("ROI_dentro_de_pantalla")

            # Regla dura adicional
            if (pillars_ok and roi_en_pantalla) or (glare_ok and (bool(g.get("bezel_ok")) or bool(e4.get("ok", False)))):
                return JSONResponse(content=to_jsonable({
                    "resultado": "pantalla",
                    "confianza": round(max(prob_pantalla, 0.88), 4),
                    "umbral": float(UMBRAL_CONFIANZA),
                    "status": "rechazado: se detecta teléfono/pantalla en la escena",
                    "razon": "Marcos negros/glare de vidrio compatibles con un teléfono mostrando la imagen.",
                    "manual": False,
                    "debug": {"preprocess": dbg, "pantalla_escena": {**dbg_screen, "reasons": reasons},
                              "heuristica_papel_fullimg": h_dbg,
                              "prob_doc": round(prob_doc, 4), "prob_pantalla": round(prob_pantalla, 4),
                              "reglas": reglas_dbg},
                }))

            if (votes >= need_votes) and weak_roi:
                log.info("salida=rechazo:pantalla_roi_debil")
                return {"resultado": "pantalla", "confianza": round(max(prob_pantalla, 0.90), 4),
                        "status": "rechazado: se detecta teléfono/pantalla en la escena",
                        "razon": "Evidencias de pantalla con localización no fiable.",
                        "manual": False,
                        "debug": {"preprocess": dbg, "pantalla_escena": {**dbg_screen, "reasons": reasons},
                                  "heuristica_papel_fullimg": h_dbg,
                                  "prob_doc": round(prob_doc, 4), "prob_pantalla": round(prob_pantalla, 4),
                                  "reglas": reglas_dbg}}

            elif (votes >= (need_votes + 1)) and not weak_roi:
                log.info("salida=rechazo:pantalla_roi_fuerte_votos_extra")
                return {"resultado": "pantalla", "confianza": round(max(prob_pantalla, 0.80), 4),
                        "status": "rechazado: se detecta teléfono/pantalla en la escena",
                        "razon": "Evidencias de pantalla superan la confianza del modelo.",
                        "manual": False,
                        "debug": {"preprocess": dbg, "pantalla_escena": {**dbg_screen, "reasons": reasons},
                                  "heuristica_papel_fullimg": h_dbg,
                                  "prob_doc": round(prob_doc, 4), "prob_pantalla": round(prob_pantalla, 4),
                                  "reglas": reglas_dbg}}

        # --------------------- Decisión final por modelo ---------------------
        if prob_doc >= UMBRAL_CONFIANZA:
            log.info("salida=aceptado:modelo_doc")
            return {"resultado": "documento_fisico", "confianza": round(prob_doc, 4),
                    "status": "aceptado: es un documento físico válido",
                    "razon": "Modelo y verificaciones no contradicen documento físico.",
                    "manual": False,
                    "debug": {"preprocess": dbg, "heuristica_papel_fullimg": h_dbg,
                              "prob_doc": round(prob_doc, 4), "prob_pantalla": round(prob_pantalla, 4),
                              "reglas": reglas_dbg}}

        if prob_pantalla >= UMBRAL_CONFIANZA:
            log.info("salida=rechazo:modelo_pantalla")
            return {"resultado": "pantalla", "confianza": round(prob_pantalla, 4),
                    "status": "rechazado: es una copia digital",
                    "razon": "El modelo detecta pantalla con alta confianza.",
                    "manual": False,
                    "debug": {"preprocess": dbg, "heuristica_papel_fullimg": h_dbg,
                              "pantalla_escena": dbg_screen,
                              "prob_doc": round(prob_doc, 4), "prob_pantalla": round(prob_pantalla, 4),
                              "reglas": reglas_dbg}}

        # Ambiguo → manual (FALLBACK FINAL)
        log.info("salida=manual:ambiguo")
        return {"resultado": "indeterminado",
                "confianza": round(max(prob_doc, prob_pantalla), 4),
                "status": "verificación manual necesaria",
                "razon": "Confianza insuficiente o señales contradictorias.",
                "manual": True,
                "debug": {"preprocess": dbg, "heuristica_papel_fullimg": h_dbg,
                          "pantalla_escena": dbg_screen,
                          "prob_doc": round(prob_doc, 4), "prob_pantalla": round(prob_pantalla, 4),
                          "reglas": reglas_dbg}}

    except Exception as e:
        log.error("EXCEPCION en verificar_documento: %s", e)
        tb = traceback.format_exc(limit=3)
        return {"resultado": "error", "status": "exception",
                "razon": str(e), "trace": tb}

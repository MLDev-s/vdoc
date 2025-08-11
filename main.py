# C:\repo_python\vdoc\main.py
import os
import json
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from utils import preprocess_image, predict
from utils_reglas import es_papel_impreso, detectar_pantalla_en_escena

app = FastAPI()

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
BORDE_MARGIN = 0.05
ID1_AR = 85.60 / 53.98  # ≈ 1.586
RECHAZO_PANTALLA_SIN_LOCALIZACION = 0.90  # si no hay ROI fiable, no rechazamos salvo confianza muy alta

# -----------------------------
# Umbral dinámico desde umbral.json
# -----------------------------
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

@app.post("/verificar-documento")
async def verificar_documento(file: UploadFile = File(...)):
    # 1) Validación de extensión
    ext = file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Formato de imagen no permitido. Usa JPG o PNG.")

    # 2) Cargar imagen
    image_bytes = await file.read()
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo procesar la imagen. Asegúrate de que sea válida.")

    # 3) Normalización de orientación antes del preprocesado
    if image.width > image.height:
        image = image.rotate(270, expand=True)  # 90° CCW

    # --- Heurística ANTI-COPIA EN PAPEL (siempre define h_dbg) ---
    h_is_papel: bool = False
    h_dbg: dict = {}
    try:
        h_is_papel, h_dbg = es_papel_impreso(image)
    except Exception:
        h_is_papel, h_dbg = False, {"error": "heuristica_papel_failed"}

    if h_is_papel:
        payload = {
            "resultado": "falso_impreso",
            "confianza": 1.0,
            "umbral": float(UMBRAL_CONFIANZA),
            "status": "rechazado: documento impreso/copiado en papel",
            "razon": "Esquinas muy afiladas, ausencia de sombra/espesor de tarjeta y/o textura de papel detectada.",
            "manual": False,
            "debug": {"heuristica_papel": h_dbg},
        }
        return JSONResponse(content=to_jsonable(payload))
    #ADICIONALMENTE:
    # Red adicional: si NO disparó papel pero hay 4 vértices muy marcados y sin sombra ⇒ manual/rechazo suave
    if not h_is_papel and h_dbg.get("approx1_vertices") == 4 and h_dbg.get("approx2_vertices") == 4 and h_dbg.get("sin_sombra_espesor"):
        return JSONResponse(content=to_jsonable({
        "resultado": "indeterminado",
        "confianza": 0.0,
        "umbral": float(UMBRAL_CONFIANZA),
        "status": "verificación manual necesaria",
        "razon": "Borde muy afilado y sin sombra/espesor: probable copia impresa.",
        "manual": True,
        "debug": {"heuristica_papel": h_dbg},
    }))
    # 4) Preprocesado (tensor, dbg) o sólo tensor
    ret = preprocess_image(image)
    if isinstance(ret, tuple) and len(ret) == 2:
        preprocessed, dbg = ret
    else:
        preprocessed = ret
        dbg = {"localized": None, "ar_after_warp": None}

    # --- Gate heurístico EXACTO sobre dbg ---
    if dbg.get("localized") is False:
        return JSONResponse(content=to_jsonable({
            "resultado": "indeterminado",
            "confianza": 0.0,
            "umbral": float(UMBRAL_CONFIANZA),
            "status": "verificación manual necesaria",
            "razon": "No se pudo localizar el documento con suficiente confianza.",
            "manual": True,
            "debug": {"preprocess": dbg, "heuristica_papel": h_dbg},
        }))

    if dbg.get("ar_after_warp") is not None and abs(float(dbg["ar_after_warp"]) - ID1_AR) > 0.15:
        return JSONResponse(content=to_jsonable({
            "resultado": "indeterminado",
            "confianza": 0.0,
            "umbral": float(UMBRAL_CONFIANZA),
            "status": "verificación manual necesaria",
            "razon": "Relación de aspecto anómala tras la rectificación.",
            "manual": True,
            "debug": {"preprocess": dbg, "heuristica_papel": h_dbg},
        }))

    # 5) Detección de pantalla/teléfono en escena
    try:
        pantalla_en_escena, dbg_screen = detectar_pantalla_en_escena(image)
    except Exception:
        pantalla_en_escena, dbg_screen = False, {}

    # 6) Inferencia del modelo
    clase_pred, conf_pred = predict(preprocessed)
    if clase_pred == "pantalla":
        prob_pantalla = float(conf_pred); prob_doc = 1.0 - prob_pantalla
    else:
        prob_doc = float(conf_pred); prob_pantalla = 1.0 - prob_doc

    # 6.a) Si no hay localización fiable y el modelo marca pantalla pero < 0.90 ⇒ manual
    if dbg.get("localized") is None and prob_pantalla >= UMBRAL_CONFIANZA and prob_pantalla < RECHAZO_PANTALLA_SIN_LOCALIZACION:
        return JSONResponse(content=to_jsonable({
            "resultado": "indeterminado",
            "confianza": round(prob_pantalla, 4),
            "umbral": float(UMBRAL_CONFIANZA),
            "status": "verificación manual necesaria",
            "razon": "El modelo sugiere 'pantalla' pero no hay localización fiable del documento.",
            "manual": True,
            "debug": {"preprocess": dbg, "heuristica_papel": h_dbg,
                      "pantalla_escena": dbg_screen,
                      "prob_doc": round(prob_doc, 4), "prob_pantalla": round(prob_pantalla, 4)},
        }))

    # 6.b) Pantalla en escena (regla)
    if pantalla_en_escena:
        if prob_doc < 0.93:
            return JSONResponse(content=to_jsonable({
                "resultado": "pantalla",
                "confianza": round(max(prob_pantalla, 0.75), 4),
                "umbral": float(UMBRAL_CONFIANZA),
                "status": "rechazado: se detecta teléfono/pantalla en la escena",
                "razon": "Se detecta bisel/borde de dispositivo y/o patrón de moiré característico.",
                "manual": False,
                "debug": {"preprocess": dbg, "pantalla_escena": dbg_screen, "heuristica_papel": h_dbg,
                          "prob_doc": round(prob_doc, 4), "prob_pantalla": round(prob_pantalla, 4)},
            }))
        else:
            return JSONResponse(content=to_jsonable({
                "resultado": "indeterminado",
                "confianza": round(prob_doc, 4),
                "umbral": float(UMBRAL_CONFIANZA),
                "status": "verificación manual necesaria",
                "razon": "Regla detecta pantalla pero el modelo es muy confiable en 'documento_fisico'.",
                "manual": True,
                "debug": {"preprocess": dbg, "pantalla_escena": dbg_screen, "heuristica_papel": h_dbg,
                          "prob_doc": round(prob_doc, 4), "prob_pantalla": round(prob_pantalla, 4)},
            }))

    # 7) Decisión estándar con umbral + zona gris
    if prob_pantalla >= UMBRAL_CONFIANZA:
        resultado = "pantalla"
        status = "rechazado: es una copia digital"
        razon = "La imagen tiene características típicas de una pantalla (iluminación homogénea, bordes digitales, o sin textura física)."
        manual = False
        confianza_out = prob_pantalla
        razonb = None
    else:
        resultado = "documento_fisico"
        status = "aceptado: es un documento físico válido"
        razon = "La imagen muestra un documento con características físicas claras (papel, sombras, textura, fondo físico)."
        confianza_out = prob_doc
        if UMBRAL_CONFIANZA - BORDE_MARGIN <= prob_pantalla < UMBRAL_CONFIANZA:
            manual = True
            razonb = (f"La predicción fue 'documento_fisico' (prob_doc={prob_doc:.4f}), "
                      f"pero prob_pantalla={prob_pantalla:.4f} está cerca del umbral {UMBRAL_CONFIANZA:.2f}.")
        else:
            manual = False
            razonb = None

    resp = {
        "resultado": resultado,
        "confianza": round(float(confianza_out), 4),
        "umbral": float(UMBRAL_CONFIANZA),
        "status": status,
        "razon": razon,
        "manual": manual,
        "debug": {"preprocess": dbg, "heuristica_papel": h_dbg,  "pantalla_escena": dbg_screen, 
                  "prob_doc": round(prob_doc, 4), "prob_pantalla": round(prob_pantalla, 4)},
    }
    if razonb:
        resp["razonb"] = razonb
    return JSONResponse(content=to_jsonable(resp))

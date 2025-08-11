# calibrar_umbral_onnx.py
import os, glob, math
import numpy as np
from PIL import Image
import onnxruntime as ort
from torchvision import transforms

# --- Config ---
MODEL_PATH = "model.onnx"
VAL_DIR = os.path.join("dataset", "val")
# ImageFolder ordena alfabéticamente; esperamos:
#   class_to_idx = {'documento_fisico': 0, 'pantalla': 1}
CLASSES = sorted([d for d in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, d))])
try:
    IDX_DOC  = CLASSES.index("documento_fisico")
    IDX_PANT = CLASSES.index("pantalla")
except ValueError:
    raise SystemExit(f"Clases en {VAL_DIR} son {CLASSES} y faltan 'documento_fisico' o 'pantalla'.")

# Mismo preprocesamiento que en entrenamiento (sin normalización extra)
tx = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_image(path):
    im = Image.open(path).convert("RGB")
    t = tx(im).numpy()  # CxHxW
    return np.expand_dims(t, 0).astype(np.float32)  # 1xCxHxW

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)

def collect_val():
    samples = []
    for cls_name in CLASSES:
        cls_dir = os.path.join(VAL_DIR, cls_name)
        label = IDX_DOC if cls_name == "documento_fisico" else IDX_PANT
        for ext in ("*.jpg","*.jpeg","*.png"):
            for p in glob.glob(os.path.join(cls_dir, "**", ext), recursive=True):
                samples.append((p, label))
    return samples

def f1_from_cm(tp, fp, fn):
    prec = tp / (tp + fp) if (tp+fp)>0 else 0.0
    rec  = tp / (tp + fn) if (tp+fn)>0 else 0.0
    if (prec+rec)==0: return 0.0, prec, rec
    return 2*prec*rec/(prec+rec), prec, rec

def main():
    samples = collect_val()
    if not samples:
        raise SystemExit("No hay muestras en dataset/val.")

    print(f"Validación con {len(samples)} imágenes | clases: {CLASSES} (doc={IDX_DOC}, pantalla={IDX_PANT})")

    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

    # Probabilidades de 'pantalla' para cada imagen
    y_true = []
    y_score = []
    for path, label in samples:
        inp = load_image(path)
        out = sess.run(None, {"input": inp})[0]  # Nx2 logits
        probs = softmax(out)
        y_true.append(label)
        y_score.append(probs[0, IDX_PANT])  # prob de clase 'pantalla'

    y_true  = np.array(y_true)
    y_score = np.array(y_score)

    # Barrido de umbral
    best = {"thr": None, "f1": -1, "prec": 0, "rec": 0, "cm": (0,0,0,0)}
    for thr in np.arange(0.50, 0.991, 0.01):
        y_pred_pant = (y_score >= thr).astype(np.int32)  # 1 si 'pantalla'
        # métricas considerando 'pantalla' como positiva
        tp = int(np.sum((y_pred_pant==1) & (y_true==IDX_PANT)))
        fp = int(np.sum((y_pred_pant==1) & (y_true==IDX_DOC)))
        fn = int(np.sum((y_pred_pant==0) & (y_true==IDX_PANT)))
        tn = int(np.sum((y_pred_pant==0) & (y_true==IDX_DOC)))
        f1, prec, rec = f1_from_cm(tp, fp, fn)
        if f1 > best["f1"]:
            best = {"thr": float(np.round(thr, 2)), "f1": f1, "prec": prec, "rec": rec, "cm": (tp, fp, fn, tn)}

    tp, fp, fn, tn = best["cm"]
    print("\n===== Mejor umbral para 'pantalla' =====")
    print(f"  Threshold: {best['thr']:.2f}")
    print(f"  F1: {best['f1']:.4f} | Precision: {best['prec']:.4f} | Recall: {best['rec']:.4f}")
    print(f"  CM (pantalla positiva): TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print("\nSugerencia: usa este valor en main.py como UMBRAL_CONFIANZA.")

if __name__ == "__main__":
    main()

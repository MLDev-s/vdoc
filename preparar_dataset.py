import os
import shutil
import random
from PIL import Image
import matplotlib.pyplot as plt

# ==== Rutas del dataset base ====
DOCXPAND_DOCUMENTS_DIR      = r"/mnt/d/entrenamiento/DocXPand-25k.tar.gz/DocXPand-25k/documents"
MIDV2020_PHOTO_DIR          = r"/mnt/c/Users/poyo_/MIDV2020/dataset/photo"
MIDV2020_SCAN_UPRIGHT_DIR   = r"/mnt/c/Users/poyo_/MIDV2020/dataset/scan_upright"
MIDV2020_SCAN_ROTATED_DIR   = r"/mnt/c/Users/poyo_/MIDV2020/dataset/scan_rotated"
OUTPUT_DIR                  = r"/home/tsgeorge/Development/validateDoc/test2/vdoc/dataset"

# ==== Rutas extra del usuario ====
IMG_REALES_DIR              = r"/home/tsgeorge/Development/validateDoc/test2/vdoc/imgs/img_reales"
IMG_FALSOS_DIR              = r"/home/tsgeorge/Development/validateDoc/test2/vdoc/imgs/img_falsos"
TRAIN_RATIO = 0.8
EXTENSIONES_VALIDAS = [".jpg", ".jpeg", ".png"]
CLEAR_OUTPUT = True  # borra dataset/train y dataset/val antes de reconstruir

random.seed(42)

def listar_imagenes_recursivo(directorio):
    imagenes = []
    if not directorio or not os.path.isdir(directorio):
        return imagenes
    for root, _, files in os.walk(directorio):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTENSIONES_VALIDAS:
                imagenes.append(os.path.join(root, f))
    return imagenes

def es_valida(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False

def preparar_estructura():
    if CLEAR_OUTPUT and os.path.isdir(OUTPUT_DIR):
        for split in ["train", "val"]:
            split_dir = os.path.join(OUTPUT_DIR, split)
            if os.path.isdir(split_dir):
                shutil.rmtree(split_dir)
    for split in ["train", "val"]:
        for clase in ["documento_fisico", "pantalla"]:
            path = os.path.join(OUTPUT_DIR, split, clase)
            os.makedirs(path, exist_ok=True)

def copiar_lista(imgs, destino):
    os.makedirs(destino, exist_ok=True)
    for img in imgs:
        try:
            shutil.copy2(img, destino)
        except Exception as e:
            print(f"  [WARN] No copié {img}: {e}")

def dividir_y_copiar(imagenes, clase, ratio):
    imagenes = [p for p in imagenes if es_valida(p)]
    random.shuffle(imagenes)
    n = int(len(imagenes) * ratio)
    train = imagenes[:n]
    val = imagenes[n:]
    copiar_lista(train, os.path.join(OUTPUT_DIR, "train", clase))
    copiar_lista(val,   os.path.join(OUTPUT_DIR, "val",   clase))

def clasificar_docxpand():
    fisico = []
    pantalla = []
    if not os.path.isdir(DOCXPAND_DOCUMENTS_DIR):
        return fisico, pantalla
    for carpeta in os.listdir(DOCXPAND_DOCUMENTS_DIR):
        path = os.path.join(DOCXPAND_DOCUMENTS_DIR, carpeta)
        if not os.path.isdir(path):
            continue
        if carpeta.endswith("_A"):
            fisico.extend(listar_imagenes_recursivo(path))
        elif carpeta.endswith("_B"):
            pantalla.extend(listar_imagenes_recursivo(path))
    return fisico, pantalla

def contar_imagenes_por_clase():
    print("\n📊 Resumen final del dataset:")
    resumen = {}
    for split in ["train", "val"]:
        for clase in ["documento_fisico", "pantalla"]:
            path = os.path.join(OUTPUT_DIR, split, clase)
            total = len(listar_imagenes_recursivo(path))
            print(f"  → {split}/{clase}: {total} imágenes")
            resumen[f"{split}/{clase}"] = total
    return resumen

def graficar_resumen(resumen_dict):
    etiquetas = list(resumen_dict.keys())
    valores = list(resumen_dict.values())
    plt.figure(figsize=(10, 6))
    plt.bar(etiquetas, valores, color=["#4CAF50", "#81C784", "#F44336", "#E57373"])
    plt.title("Distribución de imágenes por clase y conjunto")
    plt.xlabel("Carpeta")
    plt.ylabel("Cantidad de imágenes")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("resumen_dataset.png")
    print("📈 Gráfico guardado como resumen_dataset.png")

def main():
    preparar_estructura()

    # ---- DocXPand ----
    print("🔍 DocXPand → clasificación automática por _A y _B")
    docxpand_fisico, docxpand_pantalla = clasificar_docxpand()
    print(f"  → documento_fisico (DocXPand): {len(docxpand_fisico)}")
    print(f"  → pantalla (DocXPand): {len(docxpand_pantalla)}")

    # ---- MIDV-2020 ----
    print("🔍 MIDV2020 photo → documento_fisico")
    midv_photo_imgs = listar_imagenes_recursivo(MIDV2020_PHOTO_DIR)
    print(f"  → {len(midv_photo_imgs)} imágenes")

    print("🔍 MIDV2020 scan_upright → pantalla")
    scan_upright_imgs = listar_imagenes_recursivo(MIDV2020_SCAN_UPRIGHT_DIR)
    print(f"  → {len(scan_upright_imgs)} imágenes")

    print("🔍 MIDV2020 scan_rotated → pantalla")
    scan_rotated_imgs = listar_imagenes_recursivo(MIDV2020_SCAN_ROTATED_DIR)
    print(f"  → {len(scan_rotated_imgs)} imágenes")

    # ---- Carpetas extra ----
    print("🔍 img_reales → documento_fisico")
    reales_imgs = listar_imagenes_recursivo(IMG_REALES_DIR)
    print(f"  → {len(reales_imgs)} imágenes")

    print("🔍 img_falsos → pantalla")
    falsos_imgs = listar_imagenes_recursivo(IMG_FALSOS_DIR)
    print(f"  → {len(falsos_imgs)} imágenes")

    # ---- Mezcla por clase ----
    todas_doc_fisico = docxpand_fisico + midv_photo_imgs + reales_imgs
    todas_pantalla   = docxpand_pantalla + scan_upright_imgs + scan_rotated_imgs + falsos_imgs

    # ---- Copia y split ----
    dividir_y_copiar(todas_doc_fisico, "documento_fisico", TRAIN_RATIO)
    dividir_y_copiar(todas_pantalla,   "pantalla",        TRAIN_RATIO)

    print("✅ Dataset listo en formato ImageFolder.")

    resumen = contar_imagenes_por_clase()
    graficar_resumen(resumen)

if __name__ == "__main__":
    main()

#python limpiar_imagenes_corruptas.py
from PIL import Image
import os

def eliminar_imagenes_corruptas(directorio):
    total = 0
    corruptas = 0
    for root, _, files in os.walk(directorio):
        for nombre in files:
            if nombre.lower().endswith((".jpg", ".jpeg", ".png")):
                ruta = os.path.join(root, nombre)
                total += 1
                try:
                    with Image.open(ruta) as img:
                        img.verify()
                except Exception:
                    print(f"❌ Imagen corrupta eliminada: {ruta}")
                    try:
                        os.remove(ruta)
                        corruptas += 1
                    except Exception as e:
                        print(f"⚠️ No se pudo eliminar: {e}")
    print(f"\n✅ Limpieza completa en {directorio}:")
    print(f"   Imágenes revisadas: {total}")
    print(f"   Imágenes eliminadas: {corruptas}")

if __name__ == "__main__":
    eliminar_imagenes_corruptas("dataset/train")
    eliminar_imagenes_corruptas("dataset/val")

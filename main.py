from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from utils import preprocess_image, predict
from io import BytesIO
from PIL import Image

app = FastAPI()

@app.post("/verificar-documento")
async def verificar_documento(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    preprocessed = preprocess_image(image)
    resultado, confianza = predict(preprocessed)

    # Lógica de decisión basada en la confianza
    if resultado == "pantalla" and confianza > 0.90:
        status = "rechazado: es una copia digital"
    elif resultado == "documento_fisico" and confianza > 0.90:
        status = "aceptado: es un documento físico válido"
    else:
        status = "verificación manual necesaria"

    return JSONResponse(content={
        "resultado": resultado,
        "confianza": round(float(confianza), 4),
        "status": status
    })

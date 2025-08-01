import onnxruntime as ort
import numpy as np
from PIL import Image

session = ort.InferenceSession("model.onnx")
CLASSES = ["pantalla", "documento_fisico"]

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))  # CHW
    image_np = np.expand_dims(image_np, axis=0)
    return image_np

def predict(image_array):
    inputs = {session.get_inputs()[0].name: image_array}
    outputs = session.run(None, inputs)
    logits = outputs[0][0]
    class_idx = np.argmax(logits)
    confidence = float(np.max(logits))
    return CLASSES[class_idx], confidence

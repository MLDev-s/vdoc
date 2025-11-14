# ======================================================
# ENTRENAMIENTO DE RESNET18 BINARIA (OPTIMIZADA GPU)
# ======================================================

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
import os

# ======================================================
# CONFIGURACIÓN INICIAL
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # optimiza kernels para tamaño fijo (224x224)

# ======================================================
# TRANSFORMACIONES
# ======================================================
weights = ResNet18_Weights.DEFAULT
imagenet_tfms = weights.transforms()  # incluye Resize(224), ToTensor y Normalize(mean/std)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    imagenet_tfms,  # normalización oficial de ImageNet
])

val_transform = imagenet_tfms  # sin aumentos, solo resize + normalize

# ======================================================
# CARGA DE DATASETS
# ======================================================
train_dir = "dataset/train"
val_dir = "dataset/val"

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transform)

print(f"🟢 Imágenes de entrenamiento: {len(train_dataset)}")
print(f"🔵 Imágenes de validación:   {len(val_dataset)}")

# ======================================================
# DATALOADERS (OPTIMIZADOS)
# ======================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=32, shuffle=True,
    num_workers=6, pin_memory=True, persistent_workers=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=64, shuffle=False,
    num_workers=6, pin_memory=True, persistent_workers=True
)

# ======================================================
# MODELO (RESNET18 PREENTRENADO)
# ======================================================
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)  # clasificación binaria
model = model.to(device)

# ======================================================
# OPTIMIZADOR Y FUNCIÓN DE PÉRDIDA
# ======================================================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# ======================================================
# ENTRENAMIENTO
# ======================================================
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

train_losses = []
val_accuracies = []
best_acc = 0.0

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"📘 Época {epoch+1}/{EPOCHS} | Pérdida: {avg_loss:.4f}")

    # ========================
    # VALIDACIÓN
    # ========================
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = correct / max(1, total)
    val_accuracies.append(acc)
    print(f"   🔹 Precisión validación: {acc:.4f}")

    # Guarda el mejor modelo
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        print("💾 Guardado nuevo mejor modelo: best_model.pth")

# ======================================================
# GRÁFICAS DE RESULTADOS
# ======================================================
epochs = list(range(1, len(train_losses) + 1))
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, marker='o')
plt.title('Pérdida por Época')
plt.xlabel('Época')
plt.ylabel('Pérdida')

plt.subplot(1, 2, 2)
plt.plot(epochs, val_accuracies, marker='o', color='green')
plt.title('Precisión en Validación')
plt.xlabel('Época')
plt.ylabel('Precisión')

plt.tight_layout()
plt.savefig("entrenamiento_metricas.png")
print("📊 Gráfica guardada como entrenamiento_metricas.png")

# ======================================================
# EXPORTACIÓN A ONNX (VERSIÓN POR DEFECTO)
# ======================================================
model.eval()
dummy_input = torch.randn(1, 3, 224, 224, device=device)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"]
)
print("✅ Modelo exportado como model.onnx")

# Verificar opset
import onnx
m = onnx.load("model.onnx")
print(f"🧩 Opset version usada: {m.opset_import[0].version}")

# Importación de librerías necesarias
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
import os

# Transformaciones con aumento de datos para el conjunto de entrenamiento
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),               # Redimensiona las imágenes a 224x224
    transforms.RandomHorizontalFlip(),           # Aplica inversión horizontal aleatoria
    transforms.RandomRotation(5),                # Rota la imagen aleatoriamente hasta 5 grados
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Cambia brillo y contraste aleatoriamente
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Simula distorsión física
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Simula desplazamiento
    transforms.ToTensor(),                       # Convierte la imagen a tensor
])

# Transformaciones para el conjunto de validación (sin aumentos)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Definición de las rutas a los datasets
train_dir = "dataset/train"
val_dir = "dataset/val"

# Carga de datasets desde carpetas organizadas por clases
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

# Mostrar el número de imágenes cargadas
print(f"Número de imágenes de entrenamiento: {len(train_dataset)}")
print(f"Número de imágenes de validación: {len(val_dataset)}")

# Creación de dataloaders para cargar los datos en lotes
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # shuffle=True para aleatorizar
val_loader = DataLoader(val_dataset, batch_size=16)

# Carga del modelo ResNet18 con pesos preentrenados
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Reemplazo de la capa final para clasificación binaria (2 clases)
model.fc = nn.Linear(model.fc.in_features, 2)

# Envío del modelo al dispositivo (GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Definición del optimizador y función de pérdida
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Listas para guardar métricas de entrenamiento
train_losses = []
val_accuracies = []

# Entrenamiento del modelo por 10 épocas
for epoch in range(10):
    model.train()  # Modo entrenamiento (activa dropout/batchnorm)
    total_loss = 0

    # Bucle de entrenamiento por lote
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()            # Reinicia los gradientes
        outputs = model(images)          # Forward pass
        loss = criterion(outputs, labels)  # Cálculo de pérdida
        loss.backward()                  # Backward pass
        optimizer.step()                 # Actualización de pesos
        total_loss += loss.item()        # Acumulación de pérdida

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Época {epoch+1} - Pérdida entrenamiento: {avg_loss:.4f}")

    # Evaluación del modelo en el conjunto de validación
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():  # No se calculan gradientes en validación
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Selección de clase con mayor probabilidad
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    val_accuracies.append(acc)
    print(f"  → Precisión en validación: {acc:.4f}")

# Graficar pérdida de entrenamiento y precisión de validación
epochs = list(range(1, len(train_losses) + 1))

plt.figure(figsize=(10, 5))

# Gráfico de pérdida
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, marker='o')
plt.title('Pérdida por Época')
plt.xlabel('Época')
plt.ylabel('Pérdida')

# Gráfico de precisión
plt.subplot(1, 2, 2)
plt.plot(epochs, val_accuracies, marker='o', color='green')
plt.title('Precisión en Validación')
plt.xlabel('Época')
plt.ylabel('Precisión')

plt.tight_layout()
plt.savefig("entrenamiento_metricas.png")  # Guarda la figura como imagen
print("📊 Gráfica guardada como entrenamiento_metricas.png")

# Exportación del modelo a formato ONNX
dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Entrada simulada
torch.onnx.export(model, dummy_input, "model.onnx", input_names=["input"], output_names=["output"], opset_version=11)
print("✅ Modelo exportado como model.onnx")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
import os

# Ensure output directories exist
os.makedirs('output', exist_ok=True)

# Veri setini yükle
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Veriyi eğitim ve test kümelerine ayır
x_train, x_val = x_train[:-10000], x_train[-10000:]
y_train, y_val = y_train[:-10000], y_train[-10000:]

# ResNet50 için görüntüleri hazırla (3 kanallı ve 224x224 boyutuna getir)
def prepare_images_for_resnet(images):
    # Normalleştir
    images = images.astype("float32") / 255.0
    
    # 3 kanallı yap (grayscale -> RGB)
    images_rgb = np.stack((images,) * 3, axis=-1)
    
    # ResNet50 için yeniden boyutlandır (224x224)
    resized_images = np.zeros((images.shape[0], 224, 224, 3))
    for i, img in enumerate(images_rgb):
        resized_images[i] = tf.image.resize(img, (224, 224))
    
    return resized_images

print("Görüntüleri ResNet50 için hazırlama...")
x_train_resnet = prepare_images_for_resnet(x_train)
x_val_resnet = prepare_images_for_resnet(x_val)
x_test_resnet = prepare_images_for_resnet(x_test)

print("Veri seti boyutları:")
print(f"Eğitim veri seti: {x_train_resnet.shape}, {y_train.shape}")
print(f"Doğrulama veri seti: {x_val_resnet.shape}, {y_val.shape}")
print(f"Test veri seti: {x_test_resnet.shape}, {y_test.shape}")

# Part 2: ResNet50 transfer learning modeli
def create_resnet50_model():
    # ImageNet üzerinde önceden eğitilmiş ResNet50 modelini yükle
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Üst katmanları dondur (freeze)
    for layer in base_model.layers:
        layer.trainable = False
    
    # Sınıflandırma katmanlarını ekle
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])
    
    return model

# ResNet50 modelini oluştur
resnet_model = create_resnet50_model()

# Modeli özetle
resnet_model.summary()

# Modeli derle
resnet_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Modeli eğit
print("ResNet50 transfer learning modelini eğitme başlıyor...")
start_time = time.time()
history = resnet_model.fit(
    x_train_resnet, y_train,
    batch_size=32,
    epochs=5,
    validation_data=(x_val_resnet, y_val)
)
training_time = time.time() - start_time
print(f"Eğitim süresi: {training_time:.2f} saniye")

# Eğitim ve doğrulama kayıplarını görselleştir
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('output/resnet50_training_curves.png')
print(f"Eğitim grafikleri 'output/resnet50_training_curves.png' olarak kaydedildi.")
plt.close()  # Figure'ı kapat, ekranda gösterme

# CPU çıkarım (inference) süresini ölç
start_time = time.time()
# Test veri setinin ilk 1000 örneğini kullan
predictions = resnet_model.predict(x_test_resnet[:1000], verbose=0)  # Verbose=0 ekledim çıktı azaltmak için
inference_time = (time.time() - start_time) / 1000  # ms cinsinden
print(f"CPU çıkarım süresi (örnek başına): {inference_time*1000:.2f} ms")

# Test veri seti için accuracy değerini hesapla
_, accuracy = resnet_model.evaluate(x_test_resnet, y_test, verbose=0)  # Verbose=0 ekledim çıktı azaltmak için
print(f"Test Accuracy: {accuracy:.4f}")

# Model parametrelerinin sayısını hesapla
total_params = resnet_model.count_params()
print(f"Model parametrelerinin toplam sayısı: {total_params:,}")

# Tahminleri al
y_pred = resnet_model.predict(x_test_resnet, verbose=0)  # Verbose=0 ekledim çıktı azaltmak için
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig('output/resnet50_confusion_matrix.png')
print(f"Confusion matrix 'output/resnet50_confusion_matrix.png' olarak kaydedildi.")
plt.close()  # Figure'ı kapat, ekranda gösterme

# Örnek sınıflandırma sonuçları (doğru ve yanlış tahminler)
correct_indices = np.where(y_pred_classes == y_test)[0]
incorrect_indices = np.where(y_pred_classes != y_test)[0]

# 5 doğru ve 5 yanlış örnek göster
plt.figure(figsize=(12, 8))

# Doğru tahminler
for i in range(5):
    if i < len(correct_indices):
        idx = correct_indices[i]
        plt.subplot(2, 5, i+1)
        plt.imshow(x_test[idx], cmap='gray')  # Orijinal MNIST görüntüsünü göster
        plt.title(f"Gerçek: {y_test[idx]}\nTahmin: {y_pred_classes[idx]}")
        plt.axis('off')

# Yanlış tahminler
for i in range(5):
    if i < len(incorrect_indices):
        idx = incorrect_indices[i]
        plt.subplot(2, 5, i+6)
        plt.imshow(x_test[idx], cmap='gray')  # Orijinal MNIST görüntüsünü göster
        plt.title(f"Gerçek: {y_test[idx]}\nTahmin: {y_pred_classes[idx]}")
        plt.axis('off')

plt.tight_layout()
plt.savefig('output/resnet50_examples.png')
print(f"Örnek sınıflandırma sonuçları 'output/resnet50_examples.png' olarak kaydedildi.")
plt.close()  # Figure'ı kapat, ekranda gösterme

# Modeli kaydet
resnet_model.save('output/resnet50_model.h5')
print("ResNet50 transfer learning modeli 'output/resnet50_model.h5' olarak kaydedildi.") 
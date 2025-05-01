import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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

# Veriyi normalize et
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Veriyi yeniden şekillendir
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Veriyi eğitim ve test kümelerine ayır
x_train, x_val = x_train[:-10000], x_train[-10000:]
y_train, y_val = y_train[:-10000], y_train[-10000:]

print("Veri seti boyutları:")
print(f"Eğitim veri seti: {x_train.shape}, {y_train.shape}")
print(f"Doğrulama veri seti: {x_val.shape}, {y_val.shape}")
print(f"Test veri seti: {x_test.shape}, {y_test.shape}")

# Part 1: Custom CNN modeli oluştur
def create_custom_cnn():
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model

# Custom CNN modelini oluştur
custom_model = create_custom_cnn()

# Modeli özetle
custom_model.summary()

# Modeli derle
custom_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Modeli eğit
print("Custom CNN modelini eğitme başlıyor...")
start_time = time.time()
history = custom_model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val))
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
plt.savefig('output/custom_cnn_training_curves.png')
print(f"Eğitim grafikleri 'output/custom_cnn_training_curves.png' olarak kaydedildi.")
plt.close()  # Figure'ı kapat, ekranda gösterme

# CPU çıkarım (inference) süresini ölç
start_time = time.time()
# Test veri setinin ilk 1000 örneğini kullan
predictions = custom_model.predict(x_test[:1000], verbose=0)  # Verbose=0 ekledim çıktı azaltmak için
inference_time = (time.time() - start_time) / 1000  # ms cinsinden
print(f"CPU çıkarım süresi (örnek başına): {inference_time*1000:.2f} ms")

# Test veri seti için accuracy değerini hesapla
_, accuracy = custom_model.evaluate(x_test, y_test, verbose=0)  # Verbose=0 ekledim çıktı azaltmak için
print(f"Test Accuracy: {accuracy:.4f}")

# Model parametrelerinin sayısını hesapla
total_params = custom_model.count_params()
print(f"Model parametrelerinin toplam sayısı: {total_params:,}")

# Tahminleri al
y_pred = custom_model.predict(x_test, verbose=0)  # Verbose=0 ekledim çıktı azaltmak için
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig('output/custom_cnn_confusion_matrix.png')
print(f"Confusion matrix 'output/custom_cnn_confusion_matrix.png' olarak kaydedildi.")
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
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"Gerçek: {y_test[idx]}\nTahmin: {y_pred_classes[idx]}")
        plt.axis('off')

# Yanlış tahminler
for i in range(5):
    if i < len(incorrect_indices):
        idx = incorrect_indices[i]
        plt.subplot(2, 5, i+6)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"Gerçek: {y_test[idx]}\nTahmin: {y_pred_classes[idx]}")
        plt.axis('off')

plt.tight_layout()
plt.savefig('output/custom_cnn_examples.png')
print(f"Örnek sınıflandırma sonuçları 'output/custom_cnn_examples.png' olarak kaydedildi.")
plt.close()  # Figure'ı kapat, ekranda gösterme

# Modeli kaydet
custom_model.save('output/custom_cnn_model.h5')
print("Custom CNN modeli 'output/custom_cnn_model.h5' olarak kaydedildi.") 
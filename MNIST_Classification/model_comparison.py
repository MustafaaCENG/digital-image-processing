import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

# Ensure output directories exist
os.makedirs('output', exist_ok=True)

print("Model karşılaştırma ve performans analizi...")

# Eğer modeller kaydedilmişse onları yükle, aksi takdirde ilgili scriptleri çalıştır
if os.path.exists('output/custom_cnn_model.h5') and os.path.exists('output/resnet50_model.h5'):
    custom_model = keras.models.load_model('output/custom_cnn_model.h5')
    resnet_model = keras.models.load_model('output/resnet50_model.h5')
    print("Eğitilmiş modeller başarıyla yüklendi.")
else:
    print("Lütfen önce mnist_classification.py ve resnet50_transfer_learning.py scriptlerini çalıştırınız!")
    exit()

# Veri setini yükle
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

# Custom CNN için test verilerini hazırla
x_test_cnn = x_test.astype("float32") / 255.0
x_test_cnn = np.expand_dims(x_test_cnn, -1)

# ResNet50 için test verilerini hazırla
def prepare_images_for_resnet(images):
    images = images.astype("float32") / 255.0
    images_rgb = np.stack((images,) * 3, axis=-1)
    resized_images = np.zeros((images.shape[0], 224, 224, 3))
    for i, img in enumerate(images_rgb):
        resized_images[i] = tf.image.resize(img, (224, 224))
    return resized_images

x_test_resnet = prepare_images_for_resnet(x_test)

# Part 3: Performance Comparison

# Doğruluk (Accuracy) değerlendirmesi
_, custom_accuracy = custom_model.evaluate(x_test_cnn, y_test, verbose=0)
_, resnet_accuracy = resnet_model.evaluate(x_test_resnet, y_test, verbose=0)

# CPU inference time ölçümü (1000 örnek üzerinde)
test_samples = 1000

# Custom CNN için CPU inference time
start_time = time.time()
custom_model.predict(x_test_cnn[:test_samples], verbose=0)
custom_inference_time = (time.time() - start_time) / test_samples * 1000  # ms cinsinden

# ResNet50 için CPU inference time
start_time = time.time()
resnet_model.predict(x_test_resnet[:test_samples], verbose=0)
resnet_inference_time = (time.time() - start_time) / test_samples * 1000  # ms cinsinden

# Model parametre sayıları
custom_params = custom_model.count_params()
resnet_params = resnet_model.count_params()

# Sonuçları tablo olarak göster
comparison_data = {
    'Model': ['Custom CNN', 'ResNet50 (Transfer Learning)'],
    'Test Accuracy': [f"{custom_accuracy:.4f}", f"{resnet_accuracy:.4f}"],
    'CPU Inference Time (ms)': [f"{custom_inference_time:.2f}", f"{resnet_inference_time:.2f}"],
    'Parameter Count': [f"{custom_params:,}", f"{resnet_params:,}"]
}

comparison_df = pd.DataFrame(comparison_data)
print("\nModel Karşılaştırması:")
print(comparison_df.to_string(index=False))

# Sonuçları metin dosyasına da kaydet
with open('output/model_comparison_results.txt', 'w') as f:
    f.write("Model Karşılaştırması:\n")
    f.write(comparison_df.to_string(index=False))
    f.write("\n\n")

# Görsel karşılaştırma
labels = ['Custom CNN', 'ResNet50']

# Accuracy karşılaştırması
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.bar(labels, [custom_accuracy, resnet_accuracy], color=['skyblue', 'lightgreen'])
plt.title('Test Accuracy')
plt.ylim(0.9, 1.0)  # Genellikle MNIST için doğruluk 0.9+ olduğundan
for i, v in enumerate([custom_accuracy, resnet_accuracy]):
    plt.text(i, v + 0.005, f"{v:.4f}", ha='center')

# Inference time karşılaştırması
plt.subplot(1, 3, 2)
plt.bar(labels, [custom_inference_time, resnet_inference_time], color=['skyblue', 'lightgreen'])
plt.title('CPU Inference Time (ms)')
for i, v in enumerate([custom_inference_time, resnet_inference_time]):
    plt.text(i, v + 1, f"{v:.2f} ms", ha='center')

# Parametre sayısı karşılaştırması - logaritmik ölçek
plt.subplot(1, 3, 3)
plt.bar(labels, [custom_params, resnet_params], color=['skyblue', 'lightgreen'], log=True)
plt.title('Parameter Count (log scale)')
for i, v in enumerate([custom_params, resnet_params]):
    plt.text(i, v * 1.1, f"{v:,}", ha='center')

plt.tight_layout()
plt.savefig('output/model_comparison.png')
print(f"Model karşılaştırma grafiği 'output/model_comparison.png' olarak kaydedildi.")
plt.close()  # Figure'ı kapat, ekranda gösterme

# Sonuç ve değerlendirme
print("\nPerformans Değerlendirmesi:")
print("-------------------------")

performance_results = []
if custom_accuracy > resnet_accuracy:
    acc_diff = custom_accuracy - resnet_accuracy
    result = f"- Custom CNN modeli, ResNet50'den %{acc_diff*100:.2f} daha yüksek doğruluk sağlıyor."
    performance_results.append(result)
    print(result)
elif resnet_accuracy > custom_accuracy:
    acc_diff = resnet_accuracy - custom_accuracy
    result = f"- ResNet50 modeli, Custom CNN'den %{acc_diff*100:.2f} daha yüksek doğruluk sağlıyor."
    performance_results.append(result)
    print(result)
else:
    result = "- Her iki model de aynı doğruluğa sahip."
    performance_results.append(result)
    print(result)

if custom_inference_time < resnet_inference_time:
    time_factor = resnet_inference_time / custom_inference_time
    result = f"- Custom CNN modeli, ResNet50'den {time_factor:.1f}x daha hızlı çıkarım yapıyor."
    performance_results.append(result)
    print(result)
else:
    time_factor = custom_inference_time / resnet_inference_time
    result = f"- ResNet50 modeli, Custom CNN'den {time_factor:.1f}x daha hızlı çıkarım yapıyor."
    performance_results.append(result)
    print(result)

param_factor = resnet_params / custom_params
result = f"- ResNet50 modeli, Custom CNN'den {param_factor:.1f}x daha fazla parametreye sahip."
performance_results.append(result)
print(result)

print("\nSonuç:")
print("------")

conclusion = []
if custom_accuracy >= resnet_accuracy and custom_inference_time <= resnet_inference_time:
    result = "Custom CNN modeli, hem doğruluk hem de hız açısından ResNet50'den daha iyi performans gösteriyor."
    conclusion.append(result)
    print(result)
    result = "MNIST gibi basit veri setleri için özel tasarlanmış küçük ağlar, büyük transfer learning modellerinden daha etkili olabilir."
    conclusion.append(result)
    print(result)
elif resnet_accuracy > custom_accuracy and resnet_inference_time < custom_inference_time:
    result = "ResNet50 modeli, hem doğruluk hem de hız açısından Custom CNN'den daha iyi performans gösteriyor."
    conclusion.append(result)
    print(result)
    result = "Transfer learning modelinin gücü, bu basit görevde bile kendini gösteriyor."
    conclusion.append(result)
    print(result)
else:
    result = "Her iki modelin de güçlü ve zayıf yönleri var:"
    conclusion.append(result)
    print(result)
    if custom_accuracy > resnet_accuracy:
        result = "- Custom CNN daha yüksek doğruluk sağlıyor"
        conclusion.append(result)
        print(result)
    else:
        result = "- ResNet50 daha yüksek doğruluk sağlıyor"
        conclusion.append(result)
        print(result)
    
    if custom_inference_time < resnet_inference_time:
        result = "- Custom CNN daha hızlı çıkarım yapıyor"
        conclusion.append(result)
        print(result)
    else:
        result = "- ResNet50 daha hızlı çıkarım yapıyor"
        conclusion.append(result)
        print(result)
    
    result = "\nGörev için en uygun modeli seçerken önceliklerinize göre karar vermelisiniz (doğruluk, hız, model boyutu vb.)."
    conclusion.append(result)
    print(result)

# Sonuçları metin dosyasına da ekle
with open('output/model_comparison_results.txt', 'a') as f:
    f.write("\nPerformans Değerlendirmesi:\n")
    f.write("-------------------------\n")
    for result in performance_results:
        f.write(f"{result}\n")
    
    f.write("\nSonuç:\n")
    f.write("------\n")
    for result in conclusion:
        f.write(f"{result}\n")

print("\nTüm sonuçlar 'output/model_comparison_results.txt' dosyasına da kaydedildi.") 
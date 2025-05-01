# MNIST Sınıflandırma Projesi

Bu proje, MNIST veri seti üzerinde özel bir CNN ve ResNet50 transfer learning modellerini karşılaştırmaktadır.

## Gereksinimler

Aşağıdaki Python kütüphaneleri gereklidir:

```
tensorflow>=2.4.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
seaborn>=0.11.0
pandas>=1.2.0
```

Gereksinimleri yüklemek için:

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn pandas
```

## Proje Dosyaları

- `mnist_classification.py`: Özel CNN modelini uygular
- `resnet50_transfer_learning.py`: ResNet50 transfer learning modelini uygular
- `model_comparison.py`: İki modelin performansını karşılaştırır
- `technical_report.md`: Teknik rapor
- `mnist.npz`: MNIST veri seti (opsiyonel, TensorFlow keras otomatik olarak indirir)

## Nasıl Çalıştırılır

1. İlk olarak özel CNN modelini eğitin:

```bash
python mnist_classification.py
```

2. Sonra ResNet50 transfer learning modelini eğitin:

```bash
python resnet50_transfer_learning.py
```

3. Son olarak iki modeli karşılaştırın:

```bash
python model_comparison.py
```

## Çıktılar

Tüm çıktı dosyaları otomatik olarak oluşturulan `output` klasörüne kaydedilir.

Her iki model de eğitim sırasında kayıp/doğruluk grafikleri oluşturur ve aşağıdaki metrikleri rapor eder:

- Test doğruluğu
- CPU çıkarım süresi (ms/örnek)
- Model parametre sayısı
- Confusion matrix
- Doğru ve yanlış sınıflandırma örnekleri

Modeller şu dosyalara kaydedilir:
- `output/custom_cnn_model.h5`: Özel CNN modeli
- `output/resnet50_model.h5`: ResNet50 transfer learning modeli

Görsel çıktılar:
- `output/custom_cnn_training_curves.png`: Özel CNN eğitim grafikleri
- `output/custom_cnn_confusion_matrix.png`: Özel CNN karışıklık matrisi
- `output/custom_cnn_examples.png`: Özel CNN örnek tahminleri
- `output/resnet50_training_curves.png`: ResNet50 eğitim grafikleri
- `output/resnet50_confusion_matrix.png`: ResNet50 karışıklık matrisi
- `output/resnet50_examples.png`: ResNet50 örnek tahminleri
- `output/model_comparison.png`: Görsel karşılaştırma

## Notlar

- ResNet50 modeli, özellikle ilk çalıştırmada ImageNet ağırlıklarını indireceği için internet bağlantısı gerektirir
- Eğitim, donanımınıza bağlı olarak uzun sürebilir, özellikle ResNet50 modeli için
- GPU kullanılabilir durumda ise, TensorFlow otomatik olarak kullanacaktır

## Teknik Rapor

Ayrıntılı analiz ve sonuçlar için `technical_report.md` dosyasına bakınız. 
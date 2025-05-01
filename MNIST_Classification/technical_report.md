Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 26, 26, 32)          │             320 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 13, 13, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 11, 11, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 5, 5, 64)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 1600)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 1600)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 10)                  │          16,010 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 34,826 (136.04 KB)
 Trainable params: 34,826 (136.04 KB)
 Non-trainable params: 0 (0.00 B)
Custom CNN modelini eğitme başlıyor...
Epoch 1/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - accuracy: 0.7580 - loss: 0.7814 - val_accuracy: 0.9731 - val_loss: 0.1036
Epoch 2/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - accuracy: 0.9592 - loss: 0.1327 - val_accuracy: 0.9814 - val_loss: 0.0692
Epoch 3/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.9689 - loss: 0.1003 - val_accuracy: 0.9844 - val_loss: 0.0563
Epoch 4/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.9752 - loss: 0.0768 - val_accuracy: 0.9859 - val_loss: 0.0504
Epoch 5/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - accuracy: 0.9798 - loss: 0.0660 - val_accuracy: 0.9871 - val_loss: 0.0484
Epoch 6/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - accuracy: 0.9814 - loss: 0.0609 - val_accuracy: 0.9873 - val_loss: 0.0452
Epoch 7/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - accuracy: 0.9816 - loss: 0.0579 - val_accuracy: 0.9879 - val_loss: 0.0429
Epoch 8/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - accuracy: 0.9835 - loss: 0.0524 - val_accuracy: 0.9890 - val_loss: 0.0400
Epoch 9/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - accuracy: 0.9841 - loss: 0.0496 - val_accuracy: 0.9900 - val_loss: 0.0377
Epoch 10/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - accuracy: 0.9856 - loss: 0.0448 - val_accuracy: 0.9902 - val_loss: 0.0377
Eğitim süresi: 33.53 saniye
Eğitim grafikleri 'output/custom_cnn_training_curves.png' olarak kaydedildi.
CPU çıkarım süresi (örnek başına): 0.16 ms
Test Accuracy: 0.9902
Model parametrelerinin toplam sayısı: 34,826
Confusion matrix 'output/custom_cnn_confusion_matrix.png' olarak kaydedildi.
Örnek sınıflandırma sonuçları 'output/custom_cnn_examples.png' olarak kaydedildi.
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
Custom CNN modeli 'output/custom_cnn_model.h5' olarak kaydedildi.
(.venv) PS C:\Users\musta\PycharmProjects\PythonProject\MNIST_Classification> python resnet50_transfer_learning.py
2025-05-01 22:01:13.119535: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-05-01 22:01:14.098411: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
Görüntüleri ResNet50 için hazırlama...
2025-05-01 22:01:17.140786: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Veri seti boyutları:
Eğitim veri seti: (50000, 224, 224, 3), (50000,)
Doğrulama veri seti: (10000, 224, 224, 3), (10000,)
Test veri seti: (10000, 224, 224, 3), (10000,)
Model: "sequential"

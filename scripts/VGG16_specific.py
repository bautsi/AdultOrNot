import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 載入 VGG16 模型，不包含頂層
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 增加新的頂層
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # 二分类输出层
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 創建一個 DataFrame，用於 ImageDataGenerator
def create_dataframe(folder):
    images = []
    labels = []  # 'A' for adult, 'M' for minor
    for filename in os.listdir(folder):
        if filename.startswith('A') and filename.lower().endswith(('png', 'jpg', 'jpeg')):
            images.append(os.path.join(folder, filename))
            labels.append('A')
        elif filename.startswith('M') and filename.lower().endswith(('png', 'jpg', 'jpeg')):
            images.append(os.path.join(folder, filename))
            labels.append('M')
    return pd.DataFrame({'filename': images, 'class': labels})

# 準備訓練和測試資料
train_df = create_dataframe('data/specific_images')
test_df = train_df.sample(frac=0.2)  # 抽取20%作為測試集
train_df = train_df.drop(test_df.index)

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=8
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=8,
    shuffle=False
)

# 訓練模型
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# 視覺化訓練結果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Test Accuracy', color='orange')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Test Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
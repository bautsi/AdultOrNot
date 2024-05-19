import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 建立一個簡單的卷積神經網絡模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # 二分类输出层
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 创建一个 DataFrame，用于 ImageDataGenerator
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

# 准备训练和测试数据
train_df = create_dataframe('data/specific_images')
test_df = train_df.sample(frac=0.2)  # 抽取20%作为测试集
train_df = train_df.drop(test_df.index)

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=4
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=4,
    shuffle=False
)

# 训练模型
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# 视觉化训练结果
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

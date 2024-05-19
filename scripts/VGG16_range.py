import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

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
    labels = []  # 'N' for non-range, 'R' for range
    for filename in os.listdir(folder):
        if filename.startswith('N') and filename.lower().endswith(('png', 'jpg', 'jpeg')):
            images.append(os.path.join(folder, filename))
            labels.append('N')
        elif filename.startswith('R') and filename.lower().endswith(('png', 'jpg', 'jpeg')):
            images.append(os.path.join(folder, filename))
            labels.append('R')
    return pd.DataFrame({'filename': images, 'class': labels})

# 準備訓練和測試資料
train_df = create_dataframe('data/range_images')
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
    batch_size=6
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=6,
    shuffle=False
)

# 設置模型保存的回調
checkpoint_callback = ModelCheckpoint(
    'range_VGG16_model_1.keras',  # 模型保存的文件名，使用.keras擴展
    monitor='val_accuracy',  # 監控驗證準確率
    save_best_only=True,  # 只保存最好的模型
    mode='max',  # 監控的指標是最大化
    save_format='tf',  # 指定保存格式為TensorFlow SavedModel格式
    verbose=1  # 日誌顯示模型被保存的資訊
)

# 在模型訓練時添加回調函數
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    callbacks=[checkpoint_callback]  # 將回調列表加入訓練函數中
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

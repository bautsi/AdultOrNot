import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD  # 導入SGD優化器
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# 啟用混合精度訓練
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# 載入 VGG16 預訓練模型，不包含頂層
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 凍結除了最後四層以外的所有層
for layer in base_model.layers:
    layer.trainable = False

# 增加新的頂層
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='linear', dtype='float32')(x)  # 確保輸出層使用float32
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=SGD(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# 準備圖片數據
def create_dataframe(folder):
    filenames = os.listdir(folder)
    ages = [int(name.split('A')[1].split('.')[0]) for name in filenames if 'A' in name]
    filepaths = [os.path.join(folder, name) for name in filenames]
    return pd.DataFrame({'filename': filepaths, 'age': ages})

df = create_dataframe('data\\original_images')

# 資料增強與生成器設置
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 使用20%的資料作為驗證集
)

train_generator = datagen.flow_from_dataframe(
    df,
    x_col='filename',
    y_col='age',
    target_size=(224, 224),
    batch_size=1,  # 減小批次大小
    class_mode='raw',
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    df,
    x_col='filename',
    y_col='age',
    target_size=(224, 224),
    batch_size=1,  # 減小批次大小
    class_mode='raw',
    subset='validation'
)

# 設置模型保存回調與學習率調節
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_mae', mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001, verbose=1)

# 訓練模型
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[checkpoint, reduce_lr]
)

# 結果可視化
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE Over Epochs')
plt.legend()

plt.show()

import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# 建立模型
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),  # 修改輸入大小為128x128
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.6),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 創建資料框架
def create_dataframe(base_folder):
    images = []
    labels = []
    for class_folder in ['Adult', 'Minor', 'Augmented_Minor']:
        folder_path = os.path.join(base_folder, class_folder)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                images.append(os.path.join(folder_path, filename))
                label = '1' if class_folder == 'Adult' else '0'
                labels.append(label)
    return pd.DataFrame({'filename': images, 'class': labels})

# 準備訓練和測試資料
train_df = create_dataframe('data/specific_2_images/train')
test_df = create_dataframe('data/specific_2_images/test')

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 使用disk caching技術
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),  # 修改圖像尺寸
    class_mode='binary',
    batch_size=1,  # 批量大小維持為1
    shuffle=True,
    save_to_dir='cache/train',  # 指定缓存目錄
    save_prefix='aug_',
    save_format='jpeg'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),  # 修改圖像尺寸
    class_mode='binary',
    batch_size=1,  # 批量大小維持為1
    shuffle=False,
    save_to_dir='cache/test',  # 指定缓存目錄
    save_prefix='aug_',
    save_format='jpeg'
)

# 設置模型保存回調
checkpoint_path = "best_specific_model.keras"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

# 訓練模型
history = model.fit(
    train_generator,
    epochs=10,  # 將迭代次數維持為10
    validation_data=test_generator,
    callbacks=[checkpoint, reduce_lr]
)

# 視覺化訓練結果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(range(0, 11))  # 調整 x 軸刻度以顯示整數 epoch 值
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(0, 11))  # 調整 x 軸刻度以顯示整數 epoch 值
plt.legend()

plt.show()

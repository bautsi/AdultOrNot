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
    Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),  # 增加濾波器數量從32到64
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),  # 增加濾波器數量從64到128
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),  # 新增第三層卷積層，濾波器數量為256
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(1024, activation='relu'),  # 增加全連接層神經元數量從512到1024
    Dropout(0.6),  # 提高 Dropout 比例從0.5到0.6
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

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=1  # 將批次大小從32調整為1以適應計算資源限制
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=1,  # 將批次大小從32調整為1
    shuffle=False
)

# 設置模型保存回調
checkpoint_path = "best_specific_model.keras"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

# 訓練模型
history = model.fit(
    train_generator,
    epochs=10,  # 將迭代次數從20調整為10
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

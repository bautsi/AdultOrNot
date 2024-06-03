from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
from math import ceil

def augment_images(source_dir, dest_dir, augment_count):
    """
    對圖片進行增強。

    Args:
    source_dir (str): 原始圖片存放的目錄。
    dest_dir (str): 增強後圖片要保存的目錄。
    augment_count (int): 需要生成的圖片數量。
    """
    # 確保目標目錄存在
    os.makedirs(dest_dir, exist_ok=True)

    # 初始化ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=10,  # 隨機旋轉的角度範圍
        width_shift_range=0.1,  # 寬度偏移範圍
        height_shift_range=0.1,  # 高度偏移範圍
        shear_range=0.1,  # 剪切強度
        zoom_range=0.1,  # 隨機縮放範圍
        horizontal_flip=True,  # 隨機水平翻轉
        fill_mode='nearest'
    )

    # 計算每個現有圖片需要增強的次數
    existing_images = os.listdir(source_dir)
    num_existing = len(existing_images)
    augment_per_image = ceil(augment_count / num_existing)

    # 進行圖片增強
    for image_name in existing_images:
        image_path = os.path.join(source_dir, image_name)
        image = load_img(image_path)
        image_array = img_to_array(image)
        image_array = image_array.reshape((1,) + image_array.shape)

        i = 0
        for batch in datagen.flow(image_array, batch_size=1, save_to_dir=dest_dir, save_prefix='aug_', save_format='jpeg'):
            i += 1
            if i >= augment_per_image:
                break

# 設定來源和目標目錄
source_train_minor = 'data/specific_2_images/train/Minor'
dest_train_minor = 'data/specific_2_images/train/Augmented_Minor'
source_test_minor = 'data/specific_2_images/test/Minor'
dest_test_minor = 'data/specific_2_images/test/Augmented_minor'

# 計算需要增強的圖片數量
needed_train = 18841 - 4816  # train的目標數量 - 現有minor的數量
needed_test = 8118 - 1944   # test的目標數量 - 現有minor的數量

# 執行增強
augment_images(source_train_minor, dest_train_minor, needed_train)
augment_images(source_test_minor, dest_test_minor, needed_test)

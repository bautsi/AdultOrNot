from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
from math import ceil

def augment_images(source_dir, dest_dir, total_required):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    existing_images = os.listdir(source_dir)
    num_existing = len(existing_images)
    generated_count = 0

    while generated_count < total_required:
        for image_name in existing_images:
            if generated_count >= total_required:
                break

            image_path = os.path.join(source_dir, image_name)
            image = load_img(image_path)
            image_array = img_to_array(image)
            image_array = image_array.reshape((1,) + image_array.shape)

            for batch in datagen.flow(image_array, batch_size=1, save_to_dir=dest_dir, save_prefix='aug_', save_format='jpeg'):
                generated_count += 1
                if generated_count >= total_required:
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

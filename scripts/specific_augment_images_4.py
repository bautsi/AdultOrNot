from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import shutil
from math import ceil

def augment_images(source_dir, dest_dir, augment_count):
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
    augment_per_image = ceil(augment_count / num_existing)

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

def remove_excess_files(directory, target_count):
    files = os.listdir(directory)
    if len(files) > target_count:
        files_to_delete = files[target_count:]
        for file in files_to_delete:
            file_path = os.path.join(directory, file)
            os.remove(file_path)

# 設定來源和目標目錄
source_train_minor = 'data/specific_2_images/train/Minor'
dest_train_minor = 'data/specific_2_images/train/Augmented_Minor'
source_test_minor = 'data/specific_2_images/test/Minor'
dest_test_minor = 'data/specific_2_images/test/Augmented_Minor'

# 計算需要增強的圖片數量
needed_train = 18841 - 4816 - 9393
needed_test = 8118 - 1944 - 7075

# 執行增強
augment_images(source_train_minor, dest_train_minor, needed_train)

# 調整測試集中增強Minor的數量
remove_excess_files(dest_test_minor, 8118 - 1944)

print("Data augmentation and pruning are completed.")

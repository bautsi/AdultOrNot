import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

def augment_images(source_dir, target_dir, augment_count):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # 建立增強資料夾，如果已存在則先清空
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        shutil.rmtree(target_dir)  # 刪除再重新建立可以避免重複增強
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if augment_count <= 0:
            break
        
        image_path = os.path.join(source_dir, filename)
        img = load_img(image_path)  # PIL image
        x = img_to_array(img)  # 形狀 (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # 形狀 (1, 3, 150, 150)

        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=target_dir, 
                                  save_prefix='aug', 
                                  save_format='jpeg'):
            i += 1
            if i >= 10:  # 這邊我們對每一個原始圖像增強10次
                augment_count -= 10
                break

# 需要增強的圖像數量計算
minor_files_train = len(os.listdir('data/specific_2_images/train/Minor'))
adult_files_train = len(os.listdir('data/specific_2_images/train/Adult'))
minor_files_test = len(os.listdir('data/specific_2_images/test/Minor'))
adult_files_test = len(os.listdir('data/specific_2_images/test/Adult'))

# 計算需要增強到多少，這裡假設我們希望 train 和 test 中的 minor 數量都接近 adult 的數量
augment_count_train = adult_files_train - minor_files_train
augment_count_test = adult_files_test - minor_files_test

# 執行增強
augment_images('data/specific_2_images/train/minor', 'data/specific_2_images/train/Augmented_Minor', augment_count_train)
augment_images('data/specific_2_images/test/minor', 'data/specific_2_images/test/Augmented_Minor', augment_count_test)

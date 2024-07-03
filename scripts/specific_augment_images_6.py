from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import time  # 導入 time 模塊來生成時間戳

def generate_one_augment_per_image(source_dir, dest_dir):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # 確保目標目錄存在
    os.makedirs(dest_dir, exist_ok=True)

    # 遍歷來源資料夾中的圖片文件
    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [img for img in os.listdir(source_dir) if os.path.splitext(img)[1].lower() in valid_extensions]
    
    for image_name in image_files:
        try:
            image_path = os.path.join(source_dir, image_name)
            image = load_img(image_path)
            image_array = img_to_array(image)
            image_array = image_array.reshape((1,) + image_array.shape)

            # 為每個圖片生成一個增強版本，並加入時間戳
            timestamp = int(time.time())
            for batch in datagen.flow(image_array, batch_size=1, save_to_dir=dest_dir, save_prefix=f'aug_{timestamp}_', save_format='jpeg'):
                break  # 只生成一個增強版本並退出循環
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    print(f"Augmentation completed. Total augmented images in {dest_dir}: {len(os.listdir(dest_dir))}")

# 設定來源和目標目錄
source_train_minor = 'data/specific_2_images/train/Minor'
dest_train_minor = 'data/specific_2_images/train/Augmented_Minor'

# 執行增強
generate_one_augment_per_image(source_train_minor, dest_train_minor)

print("One augmentation per image completed.")

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

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

    print(f"Starting augmentation in {source_dir}. Need to generate {total_required} more images.")

    while generated_count < total_required:
        for image_name in existing_images:
            if generated_count >= total_required:
                print(f"Reached the target of {total_required} images, stopping.")
                break

            image_path = os.path.join(source_dir, image_name)
            image = load_img(image_path)
            image_array = img_to_array(image)
            image_array = image_array.reshape((1,) + image_array.shape)

            for batch in datagen.flow(image_array, batch_size=1, save_to_dir=dest_dir, save_prefix='aug_', save_format='jpeg'):
                generated_count += 1
                if generated_count >= total_required:
                    print(f"Reached the target of {total_required} images, stopping.")
                    break

    print(f"Completed augmentation. Total images in {dest_dir}: {len(os.listdir(dest_dir))}")

# 設定來源和目標目錄
source_train_minor = 'data/specific_2_images/train/Minor'
dest_train_minor = 'data/specific_2_images/train/Augmented_Minor'

# 計算需要增強的圖片數量
needed_train = 18841 - (4816 + 9914)  # train的目標數量 - 現有minor的總數

# 執行增強
augment_images(source_train_minor, dest_train_minor, needed_train)

print("Training data augmentation completed.")

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# 創建數據增強的配置
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

source_dir = 'data/specific_images'  # 已處理過的含M開頭圖片的資料夾

# 處理每一個文件
for filename in os.listdir(source_dir):
    if filename.startswith('M') and filename.lower().endswith(('png', 'jpg', 'jpeg')):
        image_path = os.path.join(source_dir, filename)
        image = load_img(image_path)
        x = img_to_array(image)  # 將圖片轉換為numpy數組
        x = x.reshape((1,) + x.shape)  # 改變形狀為(1, height, width, channels)

        # 生成1個增強的圖片
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=source_dir, save_prefix='M', save_format='jpeg'):
            i += 1
            if i >= 1:
                break  # 停止循環以避免生成過多的圖片

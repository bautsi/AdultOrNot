import os
from PIL import Image

def check_image_sizes(folder_path, min_size=60):
    small_images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                with Image.open(image_path) as img:
                    width, height = img.size
                    if width < min_size or height < min_size:
                        small_images.append((image_path, img.size))
    return small_images

# 設定你的資料夾路徑
folder_path = 'data/specific_2_images/train/Adult'
small_images = check_image_sizes(folder_path)

# 打印過小的圖片路徑和尺寸
for img_info in small_images:
    print(f"Image path: {img_info[0]}, Size: {img_info[1]}")

# 如果需要，可以顯示有多少圖片過小
print(f"Total small images: {len(small_images)}")

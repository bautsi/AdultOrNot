import os
from PIL import Image

def remove_small_images(directory, min_width=160, min_height=160):
    """
    移除指定目錄中尺寸小於min_width和min_height的圖片。
    
    Args:
    directory (str): 要檢查的圖片目錄路徑。
    min_width (int): 圖片的最小寬度。
    min_height (int): 圖片的最小高度。
    """
    # 計數器，記錄已刪除的圖片數量
    removed_count = 0

    # 遍歷目錄中的所有文件
    for filename in os.listdir(directory):
        # 獲取文件完整路徑
        file_path = os.path.join(directory, filename)

        try:
            # 開啟圖片
            with Image.open(file_path) as img:
                width, height = img.size

            # 檢查圖片尺寸
            if width < min_width or height < min_height:
                # 尺寸太小，移除圖片
                os.remove(file_path)
                removed_count += 1
                print(f'Removed: {file_path}')
        except Exception as e:
            print(f'Error processing {file_path}: {e}')

    print(f'Total removed images: {removed_count}')

# 指定成年人圖片目錄
adult_directory = 'data/specific_2_images/test/adult'  # 替換成成年人圖片所在的文件夾路徑
remove_small_images(adult_directory)

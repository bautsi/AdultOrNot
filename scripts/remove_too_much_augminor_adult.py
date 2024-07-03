import os
import random

def remove_extra_files(minor_dir, augmented_minor_dir, target_total):
    # 讀取 minor 和 augmented_minor 中的文件數
    minor_files = os.listdir(minor_dir)
    augmented_minor_files = os.listdir(augmented_minor_dir)
    total_current = len(minor_files) + len(augmented_minor_files)
    
    # 計算需要刪除的文件數量
    excess_count = total_current - target_total
    
    # 如果超出目標，隨機刪除 augmented_minor 中的文件
    if excess_count > 0:
        files_to_delete = random.sample(augmented_minor_files, excess_count)
        for file in files_to_delete:
            os.remove(os.path.join(augmented_minor_dir, file))
        print(f"Removed {excess_count} files from {augmented_minor_dir}. Total now matches target total of {target_total}.")
    else:
        print(f"No files need to be removed. Current total ({total_current}) is less than or equal to target total ({target_total}).")

# 設定目標總數
target_total_count = 18841  # train\Adult 的文件數量

# 設定 Minor 和 Augmented_Minor 文件夾的路徑
minor_dir = 'data/specific_2_images/train/Minor'
augmented_minor_dir = 'data/specific_2_images/train/Augmented_Minor'

# 執行刪除多餘文件的函數
remove_extra_files(minor_dir, augmented_minor_dir, target_total_count)

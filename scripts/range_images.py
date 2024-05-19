import os
import shutil

def process_images(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    files = os.listdir(source_dir)
    rename_count = {'R': 0, 'N': 0}  # 記錄重新命名的數量
    
    for file in files:
        if file.endswith('.jpg'):  # 假設檔案是JPG格式，可以根據實際情況修改
            # 解析年齡
            age = int(file.split('A')[-1][:2])  # 從檔名中提取年齡部分
            if 13 <= age <= 23:
                new_name = f"R{rename_count['R']}.jpg"
                rename_count['R'] += 1
            else:
                new_name = f"N{rename_count['N']}.jpg"
                rename_count['N'] += 1
            
            # 複製並重命名檔案
            shutil.copy(os.path.join(source_dir, file), os.path.join(target_dir, new_name))

    print(f"Total files processed: R: {rename_count['R']}, N: {rename_count['N']}")

# 設定原始和目標資料夾路徑
source_directory = 'data\original_images'
target_directory = 'data\range_images'

# 呼叫函數處理圖片
process_images(source_directory, target_directory)

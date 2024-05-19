import os
import shutil

# 指定原始資料夾和目標資料夾的路徑
source_dir = 'data/original_images'
target_dir = 'data/specific_images'

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 初始編號
a_count = 1
m_count = 1

# 處理每個文件
for filename in os.listdir(source_dir):
    if filename.endswith(".jpg"):  # 或者其他圖像格式
        # 解析年齡
        age = int(filename.split('A')[1][:2])
        
        # 檢查年齡是否在指定範圍內
        if 13 <= age <= 23:
            new_name = ''
            if age >= 18:
                new_name = f'A{a_count}.jpg'
                a_count += 1
            else:
                new_name = f'M{m_count}.jpg'
                m_count += 1
            
            # 複製和重命名檔案
            shutil.copy(os.path.join(source_dir, filename), os.path.join(target_dir, new_name))

print(f'Files are processed and saved in {target_dir}. Adult files: {a_count - 1}, Minor files: {m_count - 1}')

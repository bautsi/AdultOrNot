import os

def count_files(directory):
    """計算指定目錄中每個子目錄的檔案數量。"""
    for root, dirs, files in os.walk(directory):
        if root == directory:  # 只處理最頂層目錄
            continue
        total_files = len([name for name in files if not name.startswith('.')])  # 忽略隱藏檔案
        print(f"{root.split('/')[-1]} contains {total_files} files")

# 路徑設置
train_dir = 'data/specific_2_images/train'
test_dir = 'data/specific_2_images/test'

print("Training data:")
count_files(train_dir)

print("\nTesting data:")
count_files(test_dir)

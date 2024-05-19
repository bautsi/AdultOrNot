import os

def count_files(directory, prefix):
    """ Count files in a directory with a given prefix. """
    count = sum(1 for item in os.listdir(directory) if item.startswith(prefix))
    return count

# 設置您的目錄路徑
directory_path = 'data/specific_images'

# 計算以'R'和'N'開頭的檔案數量
count_A = count_files(directory_path, 'A')
count_M = count_files(directory_path, 'M')

print(f"Number of files starting with 'A': {count_A}")
print(f"Number of files starting with 'M': {count_M}")

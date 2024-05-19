import os

def count_files(directory, prefix):
    """ Count files in a directory with a given prefix. """
    count = sum(1 for item in os.listdir(directory) if item.startswith(prefix))
    return count

# 設置您的目錄路徑
directory_path = 'data/range_images'

# 計算以'R'和'N'開頭的檔案數量
count_R = count_files(directory_path, 'R')
count_N = count_files(directory_path, 'N')

print(f"Number of files starting with 'R': {count_R}")
print(f"Number of files starting with 'N': {count_N}")

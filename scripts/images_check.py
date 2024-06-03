import os

def count_files(directory):
    """
    計算指定目錄下的文件數量。
    
    Args:
    directory (str): 要檢查的目錄路徑。
    
    Returns:
    int: 目錄中的文件數量。
    """
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

# 指定目錄路徑
directories = [
    'data/specific_2_images/t.est/adult',
    'data/specific_2_images/train/adult',
    'data/specific_2_images/test/minor',
    'data/specific_2_images/train/minor'
]

# 遍歷每個目錄並打印文件數量
for directory in directories:
    file_count = count_files(directory)
    print(f'{directory} contains {file_count} files')

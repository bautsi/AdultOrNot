import os
import shutil
import numpy as np

def create_dirs(base_path):
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    os.makedirs(os.path.join(base_path, 'train', 'Adult'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'train', 'Minor'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'test', 'Adult'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'test', 'Minor'), exist_ok=True)

def process_original_images(source, dest, train_ratio=0.7):
    files = [f for f in os.listdir(source) if 'A' in f]
    np.random.shuffle(files)
    for file in files:
        age = int(file.split('A')[1][:2])
        if 13 <= age <= 23:
            category = 'Adult' if age >= 18 else 'Minor'
            prefix = 'A' if category == 'Adult' else 'M'
            new_name = f"{prefix}_{file}"
            subdir = 'train' if np.random.rand() < train_ratio else 'test'
            shutil.copy(os.path.join(source, file), os.path.join(dest, subdir, category, new_name))

def process_afad_full(source, dest, train_ratio=0.7):
    for age in range(13, 24):
        age_path = os.path.join(source, str(age))
        if os.path.exists(age_path):
            subfolders = ['111', '112']
            for subfolder in subfolders:
                sub_path = os.path.join(age_path, subfolder)
                if os.path.exists(sub_path):
                    files = os.listdir(sub_path)
                    np.random.shuffle(files)
                    for file in files:
                        category = 'Adult' if age >= 18 else 'Minor'
                        prefix = 'A' if category == 'Adult' else 'M'
                        new_name = f"{prefix}_{file}"
                        subdir = 'train' if np.random.rand() < train_ratio else 'test'
                        shutil.copy(os.path.join(sub_path, file), os.path.join(dest, subdir, category, new_name))

# Set the base directory for the new dataset
base_dir = 'data\specific_2_images'

create_dirs(base_dir)
process_original_images('data\original_images', base_dir)
process_afad_full('data\AFAD-Full', base_dir)

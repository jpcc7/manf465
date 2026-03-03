import cv2
import os
import shutil
import random

# Clears all contents in train and val images and labels folders
def clear_dataset(base_data_path):
    subsets = ['train', 'val']
    folders = ['images', 'labels']
    
    print(f"Cleaning directory: {base_data_path}...")
    
    for subset in subsets:
        for folder in folders:
            target_path = os.path.join(base_data_path, subset, folder)
            
            if os.path.exists(target_path):
                # Remove the entire directory and its contents
                shutil.rmtree(target_path)
                print(f"  - Cleared {subset}/{folder}")
            
            # Recreate the empty directory
            os.makedirs(target_path, exist_ok=True)

    print("Previous data cleared.")

# Splits and shuffles raw data into train and val sets with corresponding images and labels folders
def process_dataset(raw_dir, output_dir, split_ratio=0.8):
    classes = ['zero_fuse', 'one_fuse', 'two_fuse']
    
    for cls in classes:
        folder_path = os.path.join(raw_dir, cls)
        images = os.listdir(folder_path)
        random.shuffle(images)
        
        for i, img_name in enumerate(images):
            # Determine if this goes to Train or Val
            subset = 'train' if i < len(images) * split_ratio else 'val'
            
            # Paths
            src_path = os.path.join(folder_path, img_name)
            img = cv2.imread(src_path)
            h, w, _ = img.shape
            
            # Define label destination
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(output_dir, subset, 'labels', label_name)
            img_dest_path = os.path.join(output_dir, subset, 'images', img_name)

            # Copy image to YOLO structure
            shutil.copy(src_path, img_dest_path)

# Usage
# clear_dataset('data')
process_dataset('data/raw', 'data')
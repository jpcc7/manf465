import os
import random
import shutil

def split_dataset(export_path, output_path, train_ratio=0.8):
    # 1. Define subdirectories
    for split in ['train', 'val']:
        for folder in ['images', 'labels']:
            os.makedirs(os.path.join(output_path, split, folder), exist_ok=True)

    # 2. Get all image filenames (ignoring extensions for matching)
    image_dir = os.path.join(export_path, 'images')
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    random.shuffle(images)

    # 3. Calculate split point
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    def move_files(file_list, subset):
        for img_name in file_list:
            label_name = img_name.rsplit('.', 1)[0] + '.txt'
            
            # Move Image
            shutil.copy(os.path.join(export_path, 'images', img_name),
                        os.path.join(output_path, subset, 'images', img_name))
            # Move Label
            shutil.copy(os.path.join(export_path, 'labels', label_name),
                        os.path.join(output_path, subset, 'labels', label_name))

    move_files(train_images, 'train')
    move_files(val_images, 'val')
    print(f"Done! Train: {len(train_images)} | Val: {len(val_images)}")

if __name__ == "__main__":
    split_dataset('data/label_export', 'data')
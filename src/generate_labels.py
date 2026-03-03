import os

def create_matching_labels(image_dir, label_dir):
    # 1. Ensure the label directory exists
    os.makedirs(label_dir, exist_ok=True)

    # 2. Get list of all images
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]

    print(f"Found {len(images)} images in {image_dir}")

    count = 0
    for img_name in images:
        # 3. Create the matching .txt filename
        base_name = os.path.splitext(img_name)[0]
        label_name = f"{base_name}.txt"
        label_path = os.path.join(label_dir, label_name)

        # 4. Create the empty file if it doesn't exist
        # We use 'a' (append) or 'w' (write) to ensure the file is created/cleared
        if not os.path.exists(label_path):
            with open(label_path, 'w') as f:
                pass  # Just create an empty file
            count += 1

    print(f"Successfully created {count} new empty label files in {label_dir}.")

if __name__ == "__main__":
    # Update these paths based on your current WSL2 project structure
    img_path = "data/cvat/images/train"
    lbl_path = "data/cvat/labels/train"
    
    create_matching_labels(img_path, lbl_path)
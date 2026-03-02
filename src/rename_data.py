import os

# Standardizes names of images in raw dataset
def rename_raw_data(base_path):
    # The subfolders we defined earlier
    categories = ['zero_fuse', 'one_fuse', 'two_fuse']
    
    for category in categories:
        folder_path = os.path.join(base_path, category)
        
        if not os.path.exists(folder_path):
            print(f"Skipping {category}: Folder not found.")
            continue
            
        # Get all files and sort them to keep naming consistent
        files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        
        print(f"Renaming {len(files)} files in {category}...")
        
        for i, filename in enumerate(files):
            # Get the file extension (e.g., .jpg, .png)
            ext = os.path.splitext(filename)[1].lower()
            
            # Create the new standardized name (e.g., one_fuse_042.jpg)
            new_name = f"{category}_{i:03d}{ext}"
            
            src = os.path.join(folder_path, filename)
            dst = os.path.join(folder_path, new_name)
            
            # Rename the file
            try:
                os.rename(src, dst)
            except OSError as e:
                print(f"Error renaming {filename}: {e}")

    print("Standardization complete.")

# Usage
rename_raw_data('data/raw')
    
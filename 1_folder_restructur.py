import os
import shutil

def flatten_directory(target_dir):
    """
    Move all files from all subdirectories to the target directory,
    then remove the empty subdirectories.
    
    Args:
        target_dir (str): Path to the target directory
    """
    # Validate if target directory exists
    if not os.path.exists(target_dir):
        print(f"Error: Target directory '{target_dir}' does not exist.")
        return
    
    if not os.path.isdir(target_dir):
        print(f"Error: '{target_dir}' is not a directory.")
        return
    
    print(f"Processing directory: {target_dir}")
    
    moved_count = 0
    removed_dirs = 0
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(target_dir, topdown=False):
        # Skip the target directory itself
        if root == target_dir:
            continue
            
        # Move each file to target directory
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(target_dir, file)
            
            # Handle filename conflicts by adding counter
            counter = 1
            original_dst = dst_path
            while os.path.exists(dst_path):
                name, ext = os.path.splitext(file)
                dst_path = os.path.join(target_dir, f"{name}_{counter}{ext}")
                counter += 1
            
            try:
                shutil.move(src_path, dst_path)
                moved_count += 1
                if dst_path != original_dst:
                    print(f"  Moved (renamed): {src_path} -> {dst_path}")
                else:
                    print(f"  Moved: {src_path} -> {dst_path}")
            except Exception as e:
                print(f"  Error moving {src_path}: {e}")
        
        # Remove empty directory after moving all files
        try:
            if not os.listdir(root):  # Check if directory is empty
                os.rmdir(root)
                removed_dirs += 1
                print(f"  Removed empty directory: {root}")
            else:
                print(f"  Warning: Directory not empty, skipping removal: {root}")
        except Exception as e:
            print(f"  Error removing directory {root}: {e}")
    
    print(f"\nOperation completed:")
    print(f"  Files moved: {moved_count}")
    print(f"  Directories removed: {removed_dirs}")

if __name__ == "__main__":
    # Specify the target directory here
    target_directory = "/xxx/SSL-SI-tool/TV_output_2025-09-08_train"  # Change this to your directory path
    
    # Confirm operation
    confirm = input(f"WARNING: This will move all files from subdirectories in '{target_directory}' "
                   f"to the main directory and remove empty subdirectories.\n"
                   f"Continue? (y/N): ").strip().lower()
    
    if confirm == 'y':
        flatten_directory(target_directory)
    else:
        print("Operation cancelled.")
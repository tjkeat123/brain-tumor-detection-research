import numpy as np

import os
import glob
import shutil

def split_data(image_path, label_path, output_path, train_ratio=0.7, val_ratio=0.1):
    """
    Split the data into train, val, and test sets while keeping images and labels together.
    
    Args:
        image_path: Path to folder containing images
        label_path: Path to folder containing labels
        output_path: Path where split folders will be created
        train_ratio: Proportion of data for training (default 0.7)
        val_ratio: Proportion of data for validation (default 0.2)
        test_ratio: Remainder goes to test (default 0.1)
    """
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, 'labels'), exist_ok=True)
    
    # Get all image files and sort them for consistency
    image_files = sorted(glob.glob(os.path.join(image_path, '*.jpg')))
    
    # Get base filenames (without extension and path)
    base_names = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
    
    # Shuffle the data with a fixed seed for reproducibility
    np.random.seed(42)
    indices = np.random.permutation(len(base_names))
    
    # Calculate split points
    n_total = len(base_names)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Function to copy files to split folders
    def copy_split(split_indices, split_name):
        for idx in split_indices:
            base_name = base_names[idx]
            
            # Copy image
            src_image = os.path.join(image_path, f'{base_name}.jpg')
            dst_image = os.path.join(output_path, split_name, 'images', f'{base_name}.jpg')
            shutil.copy2(src_image, dst_image)
            
            # Copy label
            src_label = os.path.join(label_path, f'{base_name}.txt')
            dst_label = os.path.join(output_path, split_name, 'labels', f'{base_name}.txt')
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                print(f'Warning: Label file not found for {base_name}')
    
    # Copy files to respective splits
    copy_split(train_indices, 'train')
    copy_split(val_indices, 'val')
    copy_split(test_indices, 'test')
    
    print(f'Split complete:')
    print(f'  Train: {len(train_indices)} samples')
    print(f'  Val: {len(val_indices)} samples')
    print(f'  Test: {len(test_indices)} samples')
    print(f'  Total: {n_total} samples')
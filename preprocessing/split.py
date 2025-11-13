import numpy as np

import os
import glob
import shutil

def split_data(image_path, label_path, output_path, train_ratio=0.7, val_ratio=0.1):
    """
    Split the data into train, val, and test sets by patient ID.
    All images from the same patient stay together in the same split.
    
    Args:
        image_path: Path to folder containing patient ID subfolders with images
        label_path: Path to folder containing patient ID subfolders with labels
        output_path: Path where split folders will be created
        train_ratio: Proportion of patients for training (default 0.7)
        val_ratio: Proportion of patients for validation (default 0.1)
        test_ratio: Remainder goes to test (default 0.2)
    """
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, 'labels'), exist_ok=True)
    
    # Get all patient ID folders
    patient_folders = [d for d in os.listdir(image_path) 
                      if os.path.isdir(os.path.join(image_path, d))]
    patient_ids = sorted(patient_folders)
    
    # Shuffle patient IDs with a fixed seed for reproducibility
    np.random.seed(42)
    shuffled_patients = np.random.permutation(patient_ids)
    
    # Calculate split points based on number of patients
    n_patients = len(patient_ids)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    # Split patient IDs
    train_patients = shuffled_patients[:n_train]
    val_patients = shuffled_patients[n_train:n_train + n_val]
    test_patients = shuffled_patients[n_train + n_val:]
    
    # Function to copy all files from patients to split folders
    def copy_split(patient_list, split_name):
        total_images = 0
        for patient_id in patient_list:
            # Get all images for this patient
            patient_image_folder = os.path.join(image_path, patient_id)
            image_files = glob.glob(os.path.join(patient_image_folder, '*.jpg'))
            
            for image_file in image_files:
                base_name = os.path.splitext(os.path.basename(image_file))[0]
                
                # Copy image
                dst_image = os.path.join(output_path, split_name, 'images', f'{base_name}.jpg')
                shutil.copy2(image_file, dst_image)
                
                # Copy corresponding label
                src_label = os.path.join(label_path, patient_id, f'{base_name}.txt')
                dst_label = os.path.join(output_path, split_name, 'labels', f'{base_name}.txt')
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)
                else:
                    print(f'Warning: Label file not found for patient {patient_id}, image {base_name}')
                
                total_images += 1
        
        return total_images
    
    # Copy files to respective splits
    train_count = copy_split(train_patients, 'train')
    val_count = copy_split(val_patients, 'val')
    test_count = copy_split(test_patients, 'test')
    
    print(f'Split complete (by patient ID):')
    print(f'  Train: {len(train_patients)} patients ({train_count} images)')
    print(f'  Val: {len(val_patients)} patients ({val_count} images)')
    print(f'  Test: {len(test_patients)} patients ({test_count} images)')
    print(f'  Total: {n_patients} patients ({train_count + val_count + test_count} images)')
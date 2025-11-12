import numpy as np
import h5py

import os

def mat_to_yolo_label(file_path, output_folder):
    """
    Extract bounding box from .mat file using tumorBorder coordinates.
    Writes YOLO format annotation to .txt file.
    
    Args:
        file_path: Path to .mat file
        output_folder: Path to folder where .txt file will be saved
    """
    # Load .mat file (MATLAB v7.3 uses HDF5 format)
    with h5py.File(file_path, 'r') as mat_data:
        cjdata = mat_data['cjdata']
        
        # Extract label (class index)
        # YOLO uses 0-indexed classes, so subtract 1 from label (1,2,3 -> 0,1,2)
        label = int(cjdata['label'][0, 0])
        class_index = label - 1
        
        # Extract tumorBorder - stored as flat array [x1, y1, x2, y2, ...]
        tumor_border = np.array(cjdata['tumorBorder']).flatten()
        
        # Reshape into coordinate pairs [(x1, y1), (x2, y2), ...]
        # Every even index is x, every odd index is y
        x_coords = tumor_border[0::2]  # [x1, x2, x3, ...]
        y_coords = tumor_border[1::2]  # [y1, y2, y3, ...]
        
        # Find bounding box in original 512x512 image space
        x_min = x_coords.min()
        x_max = x_coords.max()
        y_min = y_coords.min()
        y_max = y_coords.max()
        
        # Convert to YOLO format (normalized to 0-1) using original 512x512 dimensions
        x_center = (x_min + x_max) / 2.0 / 512.0
        y_center = (y_min + y_max) / 2.0 / 512.0
        width = (x_max - x_min) / 512.0
        height = (y_max - y_min) / 512.0
        
        # Get base filename without extension
        file_name = os.path.basename(file_path)
        file_name_base = os.path.splitext(file_name)[0]
        
        # Create output txt file path
        txt_file_path = os.path.join(output_folder, f'{file_name_base}.txt')
        
        # Write YOLO format: class_index x_center y_center width height
        with open(txt_file_path, 'w') as f:
            f.write(f'{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')
        
        print(f'Processed: {file_name} -> {txt_file_path}')


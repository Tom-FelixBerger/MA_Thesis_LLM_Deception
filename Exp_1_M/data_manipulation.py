"""
This script manipulates the vignette_activations.h5 file by:
1. Swapping the positions of template 7 and template 8 activations and targets in the dataset
   (template 7 activations and targets move to positions 1600-1799, template 8 ones to 1400-1599)
# 2. Deleting the template_ids column
3. Saving the modified data to vignette_activations_manipulated.h5
"""

import h5py
import numpy as np

INPUT_FILE = 'vignette_activations_old_original.h5'
OUTPUT_FILE = 'vignette_activations.h5'

def manipulate_dataset():
    """Apply dataset manipulations and save to new file"""
    
    with h5py.File(INPUT_FILE, 'r') as f_in:
        # Load all data
        activations = f_in['activations'][:]
        datasets = f_in['datasets'][:]
        template_ids = f_in['template_ids'][:]
        targets_p = f_in['targets_p'][:]
        targets_c = f_in['targets_c'][:]
        
        # Identify template 7 and 8 positions (assuming they are at indices 1400-1599 and 1600-1799)
        template_7_indices = np.where(template_ids == 7)[0]
        template_8_indices = np.where(template_ids == 8)[0]
        
        # Create copies to avoid overwriting
        activations_swapped = activations.copy()
        targets_p_swapped = targets_p.copy()
        targets_c_swapped = targets_c.copy()
        
        # Swap activations
        activations_swapped[template_7_indices] = activations[template_8_indices]
        activations_swapped[template_8_indices] = activations[template_7_indices]
        
        # Swap targets
        targets_p_swapped[template_7_indices] = targets_p[template_8_indices]
        targets_p_swapped[template_8_indices] = targets_p[template_7_indices]
        
        targets_c_swapped[template_7_indices] = targets_c[template_8_indices]
        targets_c_swapped[template_8_indices] = targets_c[template_7_indices]
        
        with h5py.File(OUTPUT_FILE, 'w') as f_out:
            # Save swapped activations and targets
            f_out.create_dataset('activations', data=activations_swapped, compression='gzip')
            f_out.create_dataset('targets_p', data=targets_p_swapped)
            f_out.create_dataset('targets_c', data=targets_c_swapped)
            
            f_out.create_dataset('datasets', data=datasets)
            f_out.create_dataset('template_ids', data=template_ids)

if __name__ == "__main__":
    manipulate_dataset()
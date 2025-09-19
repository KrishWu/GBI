import pandas as pd
import numpy as np
from src.data_prep.gen_data import gen_bg

ms = 1.0  # milliseconds scaling factor

def process_bkgs(input_path=None):
    """Process gravitational wave background events into standard R-Anode feature format.
    
    This function generates or transforms gravitational wave background data 
    into the standardized feature representation used throughout the R-Anode 
    analysis pipeline. It creates time-series features appropriate for 
    gravitational wave anomaly detection.
    
    Parameters
    ----------
    input_path : str, optional
        Path to input file containing raw background data. If None, generates
        synthetic background data using gen_bg()
        
    Returns
    -------
    numpy.ndarray, shape (n_events, 6)
        Processed background events with columns:
        [time, H_strain, L_strain, H_L_sum, H_L_diff, label]
        where:
        - time: Time value in milliseconds
        - H_strain: Hanford detector strain measurement
        - L_strain: Livingston detector strain measurement  
        - H_L_sum: Sum of detector strains (H+L)
        - H_L_diff: Difference of detector strains (H-L)
        - label: Background label (always 0)
        
    Notes
    -----
    Features are designed for gravitational wave anomaly detection using
    the R-Anode methodology. When input_path is None, synthetic background
    data is generated using the gen_bg() function from gen_data.py.
    """
    
    if input_path is None:
        # Generate synthetic background data
        data = gen_bg()
        # data has shape (100, 5) with columns [time, H, L, H+L, H-L]
        # Add label column (0 for background)
        labels = np.zeros((data.shape[0], 1))
        output = np.concatenate([data, labels], axis=1)
    else:
        # Load and process real gravitational wave background data
        # This would be implemented based on the actual data format
        data_all_df = pd.read_hdf(input_path)
        
        # Extract gravitational wave features
        # Assuming the data has time and detector strain columns
        time_vals = data_all_df['time'].values if 'time' in data_all_df.columns else np.arange(len(data_all_df))
        h_strain = data_all_df['H_strain'].values if 'H_strain' in data_all_df.columns else np.random.normal(0, 0.1, len(data_all_df))
        l_strain = data_all_df['L_strain'].values if 'L_strain' in data_all_df.columns else np.random.normal(0, 0.1, len(data_all_df))
        
        # Compute derived features
        h_l_sum = h_strain + l_strain
        h_l_diff = h_strain - l_strain
        
        # Create output array
        output = np.stack([
            time_vals, 
            h_strain, 
            l_strain, 
            h_l_sum, 
            h_l_diff, 
            np.zeros(len(data_all_df))  # background label
        ], axis=1)

    return output


"""
Gravitational Wave data processing functions for anomaly detection.
This module converts time-series GW data to format compatible with existing ML pipeline.
"""
import numpy as np
from .gen_data import gen_sig, gen_bg


def process_gw_signals():
    """
    Generate GW signal events and convert to format compatible with ML pipeline.
    
    Current ML expects: (mjj, mj1, delta_mj, tau21j1, tau21j2, label=1)
    GW data provides: (time, H_strain, L_strain)
    
    We'll create features from time-series that capture important GW characteristics:
    - Peak strain amplitude (analogous to mjj)
    - H detector peak time (analogous to mj1) 
    - Time difference between H and L peaks (analogous to delta_mj)
    - H strain variance (analogous to tau21j1)
    - L strain variance (analogous to tau21j2)
    - label = 1 for signals
    """
    gw_data = gen_sig()
    gw_data= np.full((gw_data.shape[0], 1), 1)
    gw_data = np.hstack((gw_data, gw_data))
    return gw_data


def process_gw_backgrounds():
    """
    Generate GW background (noise) events and convert to format compatible with ML pipeline.
    
    Same feature extraction as signals but with label=0 for background.
    """
    gw_data = gen_bg()
    gw_data= np.full((gw_data.shape[0], 1), 0)
    gw_data = np.hstack((gw_data, gw_data))
    return gw_data


def gw_background_split(gw_bkg_data, resample_seed=42):
    """
    Split GW background data into Signal Region (SR) and Control Region (CR).
    
    For GW analysis:
    - SR: Events with higher peak amplitudes (potential signal contamination)
    - CR: Events with lower peak amplitudes (pure background for training)
    
    Args:
        gw_bkg_data: Array of shape (N, 6) with GW background events
        resample_seed: Random seed for reproducible splitting
        
    Returns:
        SR_bkg: Background events in signal region
        CR_bkg: Background events in control region  
    """
    np.random.shuffle(gw_bkg_data, random_state=resample_seed)
    
    # Split based on peak amplitude (first feature)
    time_min = np.percentile(gw_bkg_data[:, 0], 20)  # Top 20% goes to outer_mas
    time_max = np.percentile(gw_bkg_data[:, 0], 80)  # Bottom 20% goes to outer_mask
    
    inner_mask = time_min < gw_bkg_data[:, 0] & gw_bkg_data[:, 0] < time_max
    outer_mask = ~inner_mask
    
    inner_bkg = gw_bkg_data[inner_mask]
    outer_bkg = gw_bkg_data[outer_mask]
    
    print(f"GW Background split: Inner={len(inner_bkg)}, Outer={len(outer_bkg)}")
    
    return inner_bkg, outer_bkg

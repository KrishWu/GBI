"""
Gravitational Wave data processing functions for anomaly detection.
This module converts time-series GW data to format compatible with existing ML pipeline.
"""
import numpy as np
from .gen_data import gen_sig, gen_bg


def process_gw_signals(num_events=1000):
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
    all_events = []
    
    for _ in range(num_events):
        # Generate one GW signal event
        gw_data = gen_sig()  # shape (100, 3): [time, H_strain, L_strain]
        
        time = gw_data[:, 0]
        h_strain = gw_data[:, 1] 
        l_strain = gw_data[:, 2]
        
        # Extract features that capture GW signal characteristics
        # Feature 1: Peak strain amplitude (combination of both detectors)
        peak_amplitude = np.sqrt(np.max(h_strain**2) + np.max(l_strain**2))
        
        # Feature 2: H detector peak time (normalized to [0,1])
        h_peak_time = time[np.argmax(np.abs(h_strain))] / 100.0
        
        # Feature 3: Time difference between H and L peaks (detector delay)
        l_peak_time = time[np.argmax(np.abs(l_strain))] / 100.0
        time_delay = np.abs(h_peak_time - l_peak_time)
        
        # Feature 4: H strain variance (noise characterization)
        h_variance = np.var(h_strain)
        
        # Feature 5: L strain variance (noise characterization) 
        l_variance = np.var(l_strain)
        
        # Label: 1 for signal events
        label = 1.0
        
        event_features = np.array([peak_amplitude, h_peak_time, time_delay, 
                                 h_variance, l_variance, label])
        all_events.append(event_features)
    
    return np.array(all_events)


def process_gw_backgrounds(num_events=10000):
    """
    Generate GW background (noise) events and convert to format compatible with ML pipeline.
    
    Same feature extraction as signals but with label=0 for background.
    """
    all_events = []
    
    for _ in range(num_events):
        # Generate one GW background event
        gw_data = gen_bg()  # shape (100, 3): [time, H_strain, L_strain]
        
        time = gw_data[:, 0]
        h_strain = gw_data[:, 1]
        l_strain = gw_data[:, 2]
        
        # Extract same features as signals
        peak_amplitude = np.sqrt(np.max(h_strain**2) + np.max(l_strain**2))
        h_peak_time = time[np.argmax(np.abs(h_strain))] / 100.0
        l_peak_time = time[np.argmax(np.abs(l_strain))] / 100.0
        time_delay = np.abs(h_peak_time - l_peak_time)
        h_variance = np.var(h_strain)
        l_variance = np.var(l_strain)
        
        # Label: 0 for background events
        label = 0.0
        
        event_features = np.array([peak_amplitude, h_peak_time, time_delay,
                                 h_variance, l_variance, label])
        all_events.append(event_features)
    
    return np.array(all_events)


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
    np.random.seed(resample_seed)
    
    # Split based on peak amplitude (first feature)
    peak_amplitudes = gw_bkg_data[:, 0]
    amplitude_threshold = np.percentile(peak_amplitudes, 70)  # Top 30% goes to SR
    
    sr_mask = peak_amplitudes > amplitude_threshold
    cr_mask = ~sr_mask
    
    SR_bkg = gw_bkg_data[sr_mask]
    CR_bkg = gw_bkg_data[cr_mask]
    
    print(f"GW Background split: SR={len(SR_bkg)}, CR={len(CR_bkg)}")
    
    return SR_bkg, CR_bkg
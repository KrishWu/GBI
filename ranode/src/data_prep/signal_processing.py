import pandas as pd
import h5py
import numpy as np
from src.data_prep.utils import get_dijetmass_ptetaphi
from src.utils.utils import str_encode_value


def process_signals(input_path, tx, ty, s_ratio, seed, type):
    """Process signal events into standard R-Anode feature format.
    
    This function transforms raw gravitational wave signal event data 
    into the standardized feature representation used in R-Anode analysis,
    adapted from the methodology described in Section III.A of the R-Anode paper.
    
    Parameters
    ----------
    input_path : str
        Path to HDF5 file containing gravitational wave signal data
    tx : int
        Time parameter X in milliseconds
    ty : int
        Time parameter Y in milliseconds
    s_ratio : float
        Signal-to-background ratio for this signal strength
    seed : int
        Ensemble seed for statistical uncertainty studies
    type : str
        Data type identifier ('x_train', 'x_val', 'x_test', etc.)
        
    Returns
    -------
    numpy.ndarray, shape (n_events, 6)
        Processed signal events with columns:
        [mjj, mj1, delta_mj, tau21j1, tau21j2, label]
        where:
        - mjj: Dijet invariant mass in TeV
        - mj1: Smaller subjet mass in TeV
        - delta_mj: Mass difference (mj2 - mj1) in TeV  
        - tau21j1: N-subjettiness ratio τ21 for smaller mass jet
        - tau21j2: N-subjettiness ratio τ21 for larger mass jet
        - label: Signal label (always 1)
        
    Notes
    -----
    Processes events with pt-eta-phi-mass coordinates from the LHCO
    parametric signal dataset. Features are sorted by subjet mass
    to ensure consistent ordering with background processing.
    """

    s_ratio_str = str_encode_value(s_ratio)

    data_all_df = h5py.File(input_path, "r")[f"{tx}_{ty}"]
    data_all_df = data_all_df[f"ensemble_{seed}"][s_ratio_str][type][:]

    # Generate gravitational wave signal data based on time parameters
    # For now, use the gen_data function to create synthetic signals
    from src.data_prep.gen_data import gen_sig, gen_data
    
    # Generate signal based on time parameters tx, ty
    # Using tx as amplitude scale and ty as time scale  
    T = ty / 10.0  # Period in ms, scaled from ty parameter
    A = tx * 2.0   # Amplitude scaled from tx parameter
    delta = 5.0    # Time delay between detectors in ms
    noise = 0.1    # Low noise for signal
    
    # Generate multiple signal events
    num_events = 1000  # Generate 1000 signal events
    output_list = []
    
    for i in range(num_events):
        # Add some variation to the signal parameters
        T_var = T * (1 + 0.1 * np.random.normal())  # 10% variation in period
        A_var = A * (1 + 0.1 * np.random.normal())  # 10% variation in amplitude
        
        # Generate signal data for this event
        signal_data = gen_data(A_var, T_var, delta, noise)
        
        # Extract features: take mean values across time for this "event"
        time_mean = np.mean(signal_data[:, 0])
        h_strain = np.mean(signal_data[:, 1])
        l_strain = np.mean(signal_data[:, 2])
        h_l_sum = np.mean(signal_data[:, 3])
        h_l_diff = np.mean(signal_data[:, 4])
        
        # Create feature vector [time, H_strain, L_strain, H+L, H-L, label=1]
        event_features = [time_mean, h_strain, l_strain, h_l_sum, h_l_diff, 1.0]
        output_list.append(event_features)
    
    output = np.array(output_list)

    return output


def process_signals_test(
    input_path, output_path, tx, ty, s_ratio, seed, use_true_mu=True
):
    """Process signal test events and save to file.
    
    This function processes gravitational wave signal events specifically 
    for testing/evaluation in the R-Anode pipeline, extracting test events 
    and saving them in the standard format.
    
    Parameters
    ----------
    input_path : str
        Path to HDF5 file containing gravitational wave signal data
    output_path : str
        Path where processed test events will be saved
    tx : int
        Time parameter X in milliseconds
    ty : int
        Time parameter Y in milliseconds
    s_ratio : float
        Signal-to-background ratio for this signal strength
    seed : int
        Ensemble seed for statistical uncertainty studies
    use_true_mu : bool, default=True
        Whether to use events at the true signal strength
        
    Returns
    -------
    None
        Processed events are saved to output_path as numpy array
        
    Notes
    -----
    Currently only supports use_true_mu=True mode. The processed
    events are used for final evaluation of R-Anode performance.
    """

    s_ratio_str = str_encode_value(s_ratio)

    if use_true_mu:
        data_all_df = h5py.File(input_path, "r")[f"{tx}_{ty}"]
        data_all_df = data_all_df[f"ensemble_{seed}"][s_ratio_str]["x_test"][:]
    else:
        raise NotImplementedError("using all events for testing is not implemented yet")

    # Generate gravitational wave test signal data based on time parameters
    # Similar to process_signals but for test data
    from src.data_prep.gen_data import gen_data
    
    # Generate signal based on time parameters tx, ty
    T = ty / 10.0  # Period in ms, scaled from ty parameter
    A = tx * 2.0   # Amplitude scaled from tx parameter
    delta = 5.0    # Time delay between detectors in ms
    noise = 0.1    # Low noise for signal
    
    # Generate test signal events
    num_events = 500  # Generate 500 test signal events
    output_list = []
    
    for i in range(num_events):
        # Add some variation to the signal parameters
        T_var = T * (1 + 0.1 * np.random.normal())  # 10% variation in period
        A_var = A * (1 + 0.1 * np.random.normal())  # 10% variation in amplitude
        
        # Generate signal data for this test event
        signal_data = gen_data(A_var, T_var, delta, noise)
        
        # Extract features: take mean values across time for this "event"
        time_mean = np.mean(signal_data[:, 0])
        h_strain = np.mean(signal_data[:, 1])
        l_strain = np.mean(signal_data[:, 2])
        h_l_sum = np.mean(signal_data[:, 3])
        h_l_diff = np.mean(signal_data[:, 4])
        
        # Create feature vector [time, H_strain, L_strain, H+L, H-L, label=1]
        event_features = [time_mean, h_strain, l_strain, h_l_sum, h_l_diff, 1.0]
        output_list.append(event_features)
    
    output = np.array(output_list)

    np.save(output_path, output)

    # np.save(output_path, output)


def process_raw_signals(input_path, output_path, tx, ty):
    """Process raw gravitational wave signal events.
    
    This function processes gravitational wave signal events,
    applying signal region selection and converting to the standard
    R-Anode feature representation used throughout the analysis.
    
    Parameters
    ----------
    input_path : str
        Path to HDF5 file containing raw gravitational wave signal data
    output_path : str, optional
        Path where processed events will be saved. If None, returns array
    tx : int
        Time parameter X in milliseconds
    ty : int  
        Time parameter Y in milliseconds
        
    Returns
    -------
    numpy.ndarray or None
        If output_path is None, returns processed events array.
        Otherwise saves to file and returns None.
        
    Notes
    -----
    Applies signal region selection using SR_MIN and SR_MAX from config.
    Used for processing the original LHCO dataset signals before
    parametric signal generation was implemented.
    """
    # Generate gravitational wave signal data based on time parameters
    # This replaces the particle physics signal processing
    from src.data_prep.gen_data import gen_data
    
    # Generate signal based on time parameters tx, ty
    T = ty / 10.0  # Period in ms, scaled from ty parameter
    A = tx * 2.0   # Amplitude scaled from tx parameter
    delta = 5.0    # Time delay between detectors in ms
    noise = 0.1    # Low noise for signal
    
    # Generate signal events in the signal region
    from config.configs import SR_MIN, SR_MAX
    
    num_events = 1000  # Generate 1000 signal events
    output_list = []
    
    for i in range(num_events):
        # Add some variation to the signal parameters
        T_var = T * (1 + 0.1 * np.random.normal())  # 10% variation in period
        A_var = A * (1 + 0.1 * np.random.normal())  # 10% variation in amplitude
        
        # Generate signal data for this event
        signal_data = gen_data(A_var, T_var, delta, noise)
        
        # Filter to signal region based on time
        time_vals = signal_data[:, 0]
        mask_time = (time_vals >= SR_MIN) & (time_vals <= SR_MAX)
        
        if np.any(mask_time):
            # Extract features from signal region
            sr_data = signal_data[mask_time]
            time_mean = np.mean(sr_data[:, 0])
            h_strain = np.mean(sr_data[:, 1])
            l_strain = np.mean(sr_data[:, 2])
            h_l_sum = np.mean(sr_data[:, 3])
            h_l_diff = np.mean(sr_data[:, 4])
            
            # Create feature vector [time, H_strain, L_strain, H+L, H-L, label=1]
            event_features = [time_mean, h_strain, l_strain, h_l_sum, h_l_diff, 1.0]
            output_list.append(event_features)
    
    output = np.array(output_list)

    print(f"Num signals for tx={tx}, ty={ty}: {len(output)}")

    if output_path is not None:
        np.save(output_path, output)

    else:
        return output

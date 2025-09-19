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

    # shape is (N, 14) where N is the number of events, the columns orders are
    # pt_j1, eta_j1, phi_j1, mj1, Nj1, tau12j1, tau23j1, pt_j2, eta_j2, phi_j2, mj2, Nj2, tau12j2, tau23j2
    # all units are in TeV already

    pt_j1 = data_all_df[:, 0]
    eta_j1 = data_all_df[:, 1]
    phi_j1 = data_all_df[:, 2]
    mj1 = data_all_df[:, 3]
    tau21j1 = data_all_df[:, 5]
    jet1_p4 = np.stack([pt_j1, eta_j1, phi_j1, mj1], axis=1)

    pt_j2 = data_all_df[:, 7]
    eta_j2 = data_all_df[:, 8]
    phi_j2 = data_all_df[:, 9]
    mj2 = data_all_df[:, 10]
    tau21j2 = data_all_df[:, 12]
    jet2_p4 = np.stack([pt_j2, eta_j2, phi_j2, mj2], axis=1)

    jets = np.stack([jet1_p4, jet2_p4], axis=1)
    mjj = get_dijetmass_ptetaphi(jets)

    # get mj1 and mj2, sort them with mj1 being the smaller one
    mj1mj2 = np.stack([mj1, mj2], axis=1)
    mjmin = mj1mj2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    mjmax = mj1mj2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    # get tau21j1, tau21j2 and sort by mj1 mj2 in the same way
    tau21j1j2 = np.stack([tau21j1, tau21j2], axis=1)
    tau21min = tau21j1j2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    tau21max = tau21j1j2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    output = np.stack(
        [mjj, mjmin, mjmax - mjmin, tau21min, tau21max, np.ones(len(mj1mj2))], axis=1
    )

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

    # shape is (N, 14) where N is the number of events, the columns orders are
    # pt_j1, eta_j1, phi_j1, mj1, Nj1, tau12j1, tau23j1, pt_j2, eta_j2, phi_j2, mj2, Nj2, tau12j2, tau23j2
    # all units are in TeV already

    pt_j1 = data_all_df[:, 0]
    eta_j1 = data_all_df[:, 1]
    phi_j1 = data_all_df[:, 2]
    mj1 = data_all_df[:, 3]
    tau21j1 = data_all_df[:, 5]
    jet1_p4 = np.stack([pt_j1, eta_j1, phi_j1, mj1], axis=1)

    pt_j2 = data_all_df[:, 7]
    eta_j2 = data_all_df[:, 8]
    phi_j2 = data_all_df[:, 9]
    mj2 = data_all_df[:, 10]
    tau21j2 = data_all_df[:, 12]
    jet2_p4 = np.stack([pt_j2, eta_j2, phi_j2, mj2], axis=1)

    jets = np.stack([jet1_p4, jet2_p4], axis=1)
    mjj = get_dijetmass_ptetaphi(jets)

    # get mj1 and mj2, sort them with mj1 being the smaller one
    mj1mj2 = np.stack([mj1, mj2], axis=1)
    mjmin = mj1mj2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    mjmax = mj1mj2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    # get tau21j1, tau21j2 and sort by mj1 mj2 in the same way
    tau21j1j2 = np.stack([tau21j1, tau21j2], axis=1)
    tau21min = tau21j1j2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    tau21max = tau21j1j2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    output = np.stack(
        [mjj, mjmin, mjmax - mjmin, tau21min, tau21max, np.ones(len(mj1mj2))], axis=1
    )

    np.save(output_path, output)

    # target_process_df = data_all_df.query(f"mx=={mx} & my=={my}")

    # # get jet p4 info to calculate dijet mjj
    # jet1_p4 = target_process_df[["ptj1", "etaj1", "phij1", "mj1"]].values
    # jet2_p4 = target_process_df[["ptj2", "etaj2", "phij2", "mj2"]].values
    # jets = np.stack([jet1_p4, jet2_p4], axis=1)
    # mjj = get_dijetmass_ptetaphi(jets) / TeV

    # # get other features
    # # get mj1 and mj2, sort them with mj1 being the smaller one
    # mj1mj2 = np.array(target_process_df[['mj1', 'mj2']]) / TeV
    # mjmin = mj1mj2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    # mjmax = mj1mj2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    # # get tau21j1, tau21j2 and sort by mj1 mj2 in the same way
    # tau21j1 = target_process_df["tau2j1"].values / ( 1e-5 + target_process_df["tau1j1"].values )
    # tau21j2 = target_process_df["tau2j2"].values / ( 1e-5 + target_process_df["tau1j2"].values )
    # tau21j1j2 = np.stack([tau21j1, tau21j2], axis=1)
    # tau21min = tau21j1j2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    # tau21max = tau21j1j2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    # output = np.stack([mjj, mjmin, mjmax-mjmin, tau21min, tau21max, np.ones(len(mj1mj2))], axis=1)

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
    data_all_df = pd.read_hdf(input_path)
    target_process_df = data_all_df.query(f"tx=={tx} & ty=={ty}")

    # get jet p4 info to calculate dijet mjj
    jet1_p4 = target_process_df[["ptj1", "etaj1", "phij1", "mj1"]].values
    jet2_p4 = target_process_df[["ptj2", "etaj2", "phij2", "mj2"]].values
    jets = np.stack([jet1_p4, jet2_p4], axis=1)
    mjj = get_dijetmass_ptetaphi(jets) / 1000

    from config.configs import SR_MIN, SR_MAX

    mask_mjj = (mjj > SR_MIN) & (mjj < SR_MAX)
    target_process_df = target_process_df[mask_mjj]
    mjj = mjj[mask_mjj]

    # get other features
    # get mj1 and mj2, sort them with mj1 being the smaller one
    mj1mj2 = np.array(target_process_df[["mj1", "mj2"]]) / 1000
    mjmin = mj1mj2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    mjmax = mj1mj2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    # get tau21j1, tau21j2 and sort by mj1 mj2 in the same way
    tau21j1 = target_process_df["tau2j1"].values / (
        1e-5 + target_process_df["tau1j1"].values
    )
    tau21j2 = target_process_df["tau2j2"].values / (
        1e-5 + target_process_df["tau1j2"].values
    )
    tau21j1j2 = np.stack([tau21j1, tau21j2], axis=1)
    tau21min = tau21j1j2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    tau21max = tau21j1j2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    output = np.stack(
        [mjj, mjmin, mjmax - mjmin, tau21min, tau21max, np.ones(len(mj1mj2))], axis=1
    )

    print(f"Num signals for tx={tx}, ty={ty}: {len(output)}")

    if output_path is not None:
        np.save(output_path, output)

    else:
        return output

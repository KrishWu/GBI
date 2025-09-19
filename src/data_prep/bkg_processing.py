import pandas as pd
import numpy as np
from src.data_prep.utils import get_dijetmass_pxyz

TeV = 1000.0

def process_bkgs(input_path):
    """Process background events into standard R-Anode feature format.
    
    This function transforms raw background event data into the standardized
    feature representation used throughout the R-Anode analysis pipeline.
    It computes dijet masses, sorts subjet features, and formats the data
    according to the LHCO R&D dataset specifications used in the R-Anode paper.
    
    Parameters
    ----------
    input_path : str
        Path to input HDF5 file containing raw background event data
        
    Returns
    -------
    numpy.ndarray, shape (n_events, 6)
        Processed background events with columns:
        [mjj, mj1, delta_mj, tau21j1, tau21j2, label]
        where:
        - mjj: Dijet invariant mass in TeV
        - mj1: Smaller subjet mass in TeV  
        - delta_mj: Mass difference (mj2 - mj1) in TeV
        - tau21j1: N-subjettiness ratio τ21 for smaller mass jet
        - tau21j2: N-subjettiness ratio τ21 for larger mass jet
        - label: Background label (always 0)
        
    Notes
    -----
    Features correspond to the baseline features defined in the LHCO R&D
    dataset, matching the setup described in Section III.A of the R-Anode paper.
    Subjet masses and τ21 ratios are sorted by subjet mass to ensure
    consistent feature ordering.
    """

    data_all_df = pd.read_hdf(input_path)

    # bkgs only
    if "label" in data_all_df.columns:
        data_all_df = data_all_df.query("label == 0")

    # get jet p4 info to calculate dijet mjj
    jet1_p4 = data_all_df[["pxj1", "pyj1", "pzj1", "mj1"]].values
    jet2_p4 = data_all_df[["pxj2", "pyj2", "pzj2", "mj2"]].values
    jets = np.stack([jet1_p4, jet2_p4], axis=1)
    mjj = get_dijetmass_pxyz(jets) / TeV

    # get other features
    # get mj1 and mj2, sort them with mj1 being the smaller one
    mj1mj2 = np.array(data_all_df[['mj1', 'mj2']]) / TeV
    mjmin = mj1mj2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    mjmax = mj1mj2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    # get tau21j1, tau21j2 and sort by mj1 mj2 in the same way
    tau21j1 = data_all_df["tau2j1"].values / ( 1e-5 + data_all_df["tau1j1"].values )
    tau21j2 = data_all_df["tau2j2"].values / ( 1e-5 + data_all_df["tau1j2"].values )
    tau21j1j2 = np.stack([tau21j1, tau21j2], axis=1)
    tau21min = tau21j1j2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    tau21max = tau21j1j2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    output = np.stack([mjj, mjmin, mjmax-mjmin, tau21min, tau21max, np.zeros(len(mj1mj2))], axis=1)

    return output


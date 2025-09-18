import os, sys
import importlib
import luigi
import law
import numpy as np
import pandas as pd
from scipy.stats import rv_histogram
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import json
from src.utils.utils import NumpyEncoder

from src.utils.law import (
    BaseTask,
    SignalStrengthMixin,
    ProcessMixin,
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
)


class ProcessSignal(SignalStrengthMixin, ProcessMixin, BaseTask):
    """Task for processing signal events in R-Anode workflow.
    
    This task implements signal data preprocessing as part of the R-Anode
    workflow pipeline. It handles signal event format standardization,
    feature extraction, and preparation for density model training.
    
    Inherits from SignalStrengthMixin for signal ratio management,
    ProcessMixin for mass point configuration, and BaseTask for
    common workflow functionality.
    
    Notes
    -----
    Part of the Luigi/Law workflow system for R-Anode analysis pipeline.
    Ensures reproducible signal processing with proper dependency tracking.
    """

    def output(self):
        return {
            "signals": self.local_target("reprocessed_signals.npy"),
        }

    @law.decorator.safe_output
    def run(self):
        from src.data_prep.gw_processing import process_gw_signals

        sig = process_gw_signals()
        np.save(self.output()["signals"].path, sig)


class ProcessBkg(BaseTask):
    """Task for processing background events in R-Anode workflow.
    
    This task implements background data preprocessing as part of the R-Anode
    analysis pipeline. It handles background event processing, signal region/
    control region separation, and preprocessing parameter calculation for
    normalizing flows.
    
    The task processes background events into the standardized format used
    throughout R-Anode, splits them into signal and control regions, and
    computes preprocessing parameters needed for normalizing flow training.
    
    Notes
    -----
    Background events are split into SR (signal region) and CR (control region).
    The CR events are used to calculate normalizing parameters and then split
    into training and validation sets for background model training.
    This follows the R-Anode methodology where background models are learned
    from sideband data.
    """

    def output(self):
        return {
            "SR_bkg": self.local_target("reprocessed_bkgs_sr.npy"),
            "CR_train": self.local_target("reprocessed_bkgs_cr_train.npy"),
            "CR_val": self.local_target("reprocessed_bkgs_cr_val.npy"),
            "pre_parameters": self.local_target("pre_parameters.json"),
        }

    @law.decorator.safe_output
    def run(self):
        from src.data_prep.gw_processing import process_gw_backgrounds

        bg = process_gw_backgrounds()

        # split into trainval and test set
        from src.data_prep.gw_processing import gw_background_split

        SR_bkg, CR_bkg = gw_background_split(
            bg,
            resample_seed=42,
        )

        # save SR data
        self.output()["SR_bkg"].parent.touch()
        np.save(self.output()["SR_bkg"].path, SR_bkg)

        from src.data_prep.utils import (
            logit_transform,
            preprocess_params_transform,
            preprocess_params_fit,
        )

        # ----------------------- calculate normalizing parameters -----------------------
        pre_parameters = preprocess_params_fit(CR_bkg)
        # save pre_parameters
        self.output()["pre_parameters"].parent.touch()
        with open(self.output()["pre_parameters"].path, "w") as f:
            json.dump(pre_parameters, f, cls=NumpyEncoder)

        # ----------------------- process data in CR -----------------------
        CR_bkg = preprocess_params_transform(CR_bkg, pre_parameters)
        CR_bkg_train, CR_bkg_val = train_test_split(
            CR_bkg, test_size=0.25, random_state=42
        )

        # save training and validation data in CR
        np.save(self.output()["CR_train"].path, CR_bkg_train)
        np.save(self.output()["CR_val"].path, CR_bkg_val)


class PreprocessingFold(
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask,
):
    """Task for fold-based data preprocessing in R-Anode workflow.
    
    This task implements k-fold cross-validation data preparation for R-Anode
    analysis. It combines signal and background data according to specified
    signal strength ratios, applies preprocessing transformations, and performs
    fold-based splitting for robust uncertainty estimation.
    
    The task handles the critical data mixing step where signal events are
    combined with background events at the specified signal fraction,
    preprocessed using parameters learned from control regions, and split
    into training/validation/test sets for R-Anode model training.
    
    Attributes
    ----------
    Inherits from multiple mixins:
    - FoldSplitRandomMixin: Controls randomness in fold splitting
    - FoldSplitUncertaintyMixin: Manages number of folds for uncertainty estimation
    - SignalStrengthMixin: Handles signal-to-background ratio parameters
    - ProcessMixin: Manages mass point configuration
    - BaseTask: Provides basic workflow functionality
    
    Notes
    -----
    Creates separate datasets for signal model (mass-shifted) and background 
    model training. The mass shifting by -3.5 TeV follows the R-Anode workflow
    for conditional density estimation.
    """

    def requires(self):

        if self.s_ratio != 0:
            return {
                "signal": ProcessSignal.req(self),
                "bkg": ProcessBkg.req(self),
            }
        else:
            return {
                "bkg": ProcessBkg.req(self),
            }

    def output(self):
        return {
            "SR_data_trainval_model_S": self.local_target(
                "data_SR_data_trainval_model_S.npy"
            ),
            "SR_data_test_model_S": self.local_target("data_SR_data_test_model_S.npy"),
            "SR_data_trainval_model_B": self.local_target(
                "data_SR_data_trainval_model_B.npy"
            ),
            "SR_data_test_model_B": self.local_target("data_SR_data_test_model_B.npy"),
            "SR_time_hist": self.local_target("SR_time_hist.json"),
        }

    @law.decorator.safe_output
    def run(self):

        from src.data_prep.utils import (
            logit_transform,
            preprocess_params_transform,
            preprocess_params_fit,
        )

        # load data
        if self.s_ratio != 0:
            SR_signal = np.load(self.input()["signal"]["signals"].path)
        SR_bkg = np.load(self.input()["bkg"]["SR_bkg"].path)

        pre_parameters = json.load(
            open(self.input()["bkg"]["pre_parameters"].path, "r")
        )
        for key in pre_parameters.keys():
            pre_parameters[key] = np.array(pre_parameters[key])

        # ----------------------- time hist in SR -----------------------
        time = SR_bkg[SR_bkg[:, -1] == 0, 0]
        bins = np.linspace(np.min(time), np.max(time), 50)
        hist_back = np.histogram(time, bins=bins, density=True)
        # save time histogram and bins
        self.output()["SR_time_hist"].parent.touch()
        with open(self.output()["SR_time_hist"].path, "w") as f:
            json.dump({"hist": hist_back[0], "bins": hist_back[1]}, f, cls=NumpyEncoder)

        # ----------------------- make SR data -----------------------

        # concatenate signal and bkg
        if self.s_ratio != 0:
            SR_data = np.concatenate([SR_signal, SR_bkg], axis=0)
        else:
            SR_data = SR_bkg

        # process data
        _, mask = logit_transform(
            SR_data[:, 1:-1], pre_parameters["min"], pre_parameters["max"]
        )
        SR_data = SR_data[mask]
        SR_data = preprocess_params_transform(SR_data, pre_parameters)

        # split into trainval and test set
        from src.data_prep.utils import fold_splitting

        SR_data_trainval, SR_data_test = fold_splitting(
            SR_data,
            n_folds=self.fold_split_num,
            random_seed=self.ensemble,
            test_fold=self.fold_split_seed,
        )

        # For signal model, we shift the mass by -3.5 following RANODE workflow
        # copy one set for signal model
        SR_data_trainval_model_S = SR_data_trainval.copy()
        SR_data_test_model_S = SR_data_test.copy()
        # shift mass by -3.5 for signals
        SR_data_trainval_model_S[:, 0] -= 3.5
        SR_data_test_model_S[:, 0] -= 3.5

        np.save(
            self.output()["SR_data_trainval_model_S"].path, SR_data_trainval_model_S
        )
        np.save(self.output()["SR_data_test_model_S"].path, SR_data_test_model_S)

        # copy another set for background model
        SR_data_trainval_model_B = SR_data_trainval.copy()
        SR_data_test_model_B = SR_data_test.copy()

        np.save(
            self.output()["SR_data_trainval_model_B"].path, SR_data_trainval_model_B
        )
        np.save(self.output()["SR_data_test_model_B"].path, SR_data_test_model_B)

        # print out some info
        trainval_sig_num = SR_data_trainval_model_B[:, -1].sum()
        trainval_bkg_num = (SR_data_trainval_model_B[:, -1] == 0).sum()
        train_mu = trainval_sig_num / (trainval_sig_num + trainval_bkg_num)
        test_sig_num = SR_data_test_model_B[:, -1].sum()
        test_bkg_num = (SR_data_test_model_B[:, -1] == 0).sum()
        test_mu = test_sig_num / (test_sig_num + test_bkg_num)
        print("Fold splitting index is: ", self.fold_split_seed)
        print("true mu: ", self.s_ratio)
        print(
            "trainval mu: ",
            train_mu,
            "trainval sig num: ",
            trainval_sig_num,
            "trainval bkg num: ",
            trainval_bkg_num,
        )
        print(
            "test mu: ",
            test_mu,
            "test sig num: ",
            test_sig_num,
            "test bkg num: ",
            test_bkg_num,
        )


class PlotMjjDistribution(
    ProcessMixin,
    BaseTask,
):
    """Task for plotting dijet mass distributions in R-Anode analysis.
    
    This task creates diagnostic plots showing the dijet invariant mass (mjj)
    distributions for background and signal events, with signal region
    boundaries clearly marked. Used for validating data processing and
    understanding the signal/background separation in mass space.
    
    The plot helps visualize the mass distributions used in R-Anode analysis
    and shows the signal region definition relative to the background
    distribution shape.
    
    Attributes
    ----------
    Inherits from ProcessMixin for mass point configuration and BaseTask
    for basic workflow functionality.
    
    Notes
    -----
    Plots both background events (from all regions) and signal events,
    with vertical lines marking the signal region boundaries at 3.3-3.7 TeV.
    Essential for validating the R-Anode analysis setup and data quality.
    """

    def requires(self):
        return {
            "bkg": ProcessBkg.req(self),
        }

    def output(self):
        return self.local_target("mjj_distribution.pdf")

    @law.decorator.safe_output
    def run(self):

        # load bkg
        SR_bkg = np.load(self.input()["bkg"]["SR_bkg"].path)
        SR_bkg_mjj = SR_bkg[:, 0]
        SB_bkg_train = np.load(self.input()["bkg"]["CR_train"].path)
        SB_bkg_mjj_train = SB_bkg_train[:, 0]
        SB_bkg_val = np.load(self.input()["bkg"]["CR_val"].path)
        SB_bkg_mjj_val = SB_bkg_val[:, 0]
        SB_bkg_mjj = np.concatenate([SB_bkg_mjj_train, SB_bkg_mjj_val], axis=0)
        bkg_mjj = np.concatenate([SR_bkg_mjj, SB_bkg_mjj], axis=0)

        # process signal
        data_dir = os.environ.get("DATA_DIR")

        data_path = f"{data_dir}/extra_raw_lhco_samples/events_anomalydetection_Z_XY_qq_parametric.h5"

        from src.data_prep.signal_processing import process_raw_signals

        signal = process_raw_signals(
            data_path, output_path=None, mx=self.mx, my=self.my
        )
        signal_mjj = signal[:, 0]

        # make plot
        from quickstats.plots import VariableDistributionPlot
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        dfs = {
            "background": pd.DataFrame({"mjj": bkg_mjj}),
            "signal": pd.DataFrame({"mjj": signal_mjj}),
        }

        plot_options = {
            "background": {
                "styles": {
                    "color": "black",
                    "histtype": "step",
                    "lw": 2,
                }
            },
            "signal": {
                "styles": {
                    "color": "red",
                    "histtype": "stepfilled",
                    "lw": 2,
                }
            },
        }

        plotter = VariableDistributionPlot(dfs, plot_options=plot_options)

        self.output().parent.touch()
        output_path = self.output().path
        with PdfPages(output_path) as pdf:

            axis = plotter.draw(
                "mjj",
                logy=True,
                normalize=False,
                bins=np.linspace(
                    2,
                    8,
                    151,
                ),
                unit="TeV",
                show_error=False,
                comparison_options=None,
                xlabel="mjj",
            )

            plt.axvline(
                x=3.3,
                color="black",
                linestyle="--",
                label="Signal Region Boundary (3.3 - 3.7 TeV)",
            )

            plt.axvline(x=3.7, color="black", linestyle="--")

            plt.xlim(2, 8)
            pdf.savefig(bbox_inches="tight")
            plt.close()

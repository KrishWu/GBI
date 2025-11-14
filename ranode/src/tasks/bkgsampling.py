import os, sys
import importlib
import luigi
import law
import json
import copy
import numpy as np
import pandas as pd
from scipy.stats import rv_histogram
from sklearn.model_selection import train_test_split
import pickle
import torch

from src.utils.law import (
    BaseTask,
    TemplateRandomMixin,
    BkgTemplateUncertaintyMixin,
    BkgModelMixin,
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    ProcessMixin,
)
from src.tasks.preprocessing import PreprocessingFold, ProcessBkg, ProcessSignal
from src.tasks.bkgtemplate import BkgTemplateTraining
from src.utils.utils import NumpyEncoder, str_encode_value


class SampleModelBinSR(
    ProcessMixin,
    BaseTask,
):
    device = luigi.Parameter(default="cuda:0")

    def requires(self):
        return {
            "bkg_model": BkgTemplateTraining.req(self, train_random_seed=0),
            "bkg_data": ProcessBkg.req(self),
        }

    def output(self):
        return self.local_target("sampled_bkgs.npy")

    @law.decorator.safe_output
    def run(self):
        # load SR real bkgs
        SR_bkg_data = np.load(self.input()["bkg_data"]["SR_bkg"].path)
        # load time in the first column
        time_bkg_data = SR_bkg_data[:, 0]
        time_bkg_data = (
            torch.from_numpy(time_bkg_data)
            .reshape((-1, 1))
            .type(torch.FloatTensor)
            .to(self.device)
        )

        # load bkg model
        # define model
        from src.models.model_B import DensityEstimator

        config_file = os.path.join("src", "models", "DE_MAF_model.yml")

        model_B = DensityEstimator(config_file, eval_mode=True, device=self.device)
        model_B.model.load_state_dict(
            torch.load(self.input()["bkg_model"]["bkg_model"].path)
        )
        model_B.model.to(self.device)
        model_B.model.eval()

        # fix randomness based on ensemble number
        torch.manual_seed(self.ensemble + 100000)

        with torch.no_grad():
            sampled_SR_events = model_B.model.sample(
                num_samples=SR_bkg_data.shape[0],
                cond_inputs=time_bkg_data,
            )

        sampled_SR_events = sampled_SR_events.cpu().numpy()

        # add time column
        sampled_SR_events = np.concatenate(
            (time_bkg_data.cpu().numpy(), sampled_SR_events), axis=1
        )
        # add last column of 0 for bkg
        sampled_SR_events = np.concatenate(
            (sampled_SR_events, np.zeros((sampled_SR_events.shape[0], 1))), axis=1
        )

        # save sampled bkgs
        self.output().parent.touch()
        np.save(self.output().path, sampled_SR_events)


class PreprocessingFoldwModelBGen(
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    ProcessMixin,
    BaseTask,
):

    def requires(self):

        if self.s_ratio != 0:
            return {
                "signal": ProcessSignal.req(self),
                "real_bkg": ProcessBkg.req(self),
                "gen_bkg": SampleModelBinSR.req(self),
            }
        else:
            return {
                "real_bkg": ProcessBkg.req(self),
                "gen_bkg": SampleModelBinSR.req(self),
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

    def _combine_gw_data_by_time(self, signal_data, background_data, amplitude_scale):
        """Combine gravitational wave signal and background data by matching timestamps.
        
        Instead of concatenating events, this method combines strain measurements
        at corresponding time points, scaling the signal by the amplitude factor.
        
        Parameters
        ----------
        signal_data : numpy.ndarray, shape (n_events, n_features)
            Signal data with columns [time, H_strain, L_strain, H+L, H-L, label]
        background_data : numpy.ndarray, shape (n_events, n_features)  
            Background data with same structure as signal_data
        amplitude_scale : float
            Amplitude scaling factor to apply to signal strains
            
        Returns
        -------
        numpy.ndarray, shape (n_events, n_features)
            Combined data with signal strains added to background at matching times
        """
        # Extract time columns
        signal_times = signal_data[:, 0]
        background_times = background_data[:, 0]
        
        # Find overlapping time points (should be identical for GW data)
        combined_data = background_data.copy()
        
        # For each background time point, find matching signal time point
        for i, bg_time in enumerate(background_times):
            # Find closest signal time (should be exact match for synthetic data)
            time_diffs = np.abs(signal_times - bg_time)
            closest_signal_idx = np.argmin(time_diffs)
            
            if time_diffs[closest_signal_idx] < 1e-6:  # Close enough to be same time
                # Add scaled signal strains to background strains (columns 1-4: H, L, H+L, H-L)
                combined_data[i, 1:5] += amplitude_scale * signal_data[closest_signal_idx, 1:5]
                # Keep background label (should be 0 for background)
        
        return combined_data

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
        SR_bkg = np.load(self.input()["gen_bkg"].path)

        # preprocessing parameters are from CR
        pre_parameters = json.load(
            open(self.input()["real_bkg"]["pre_parameters"].path, "r")
        )
        for key in pre_parameters.keys():
            pre_parameters[key] = np.array(pre_parameters[key])

        # ----------------------- time hist in SR -----------------------
        from config.configs import SR_MIN, SR_MAX

        time = SR_bkg[SR_bkg[:, -1] == 0, 0]
        bins = np.linspace(SR_MIN, SR_MAX, 50)
        hist_back = np.histogram(time, bins=bins, density=True)
        # save time histogram and bins
        self.output()["SR_time_hist"].parent.touch()
        with open(self.output()["SR_time_hist"].path, "w") as f:
            json.dump({"hist": hist_back[0], "bins": hist_back[1]}, f, cls=NumpyEncoder)

        # ----------------------- make SR data -----------------------

        # combine signal and background at matching timestamps
        if self.s_ratio != 0:
            # process signals only since bkgs have already been processed
            _, mask = logit_transform(
                SR_signal[:, 1:-1], pre_parameters["min"], pre_parameters["max"]
            )
            SR_signal = SR_signal[mask]
            SR_signal = preprocess_params_transform(SR_signal, pre_parameters)

            # Time-based combination: merge strain values at matching timestamps
            SR_data = self._combine_gw_data_by_time(SR_signal, SR_bkg, self.s_ratio)
        else:
            SR_data = SR_bkg

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


class PredictBkgProbGen(
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    ProcessMixin,
    BaseTask,
):

    device = luigi.Parameter(default="cuda")

    def requires(self):
        return {
            "bkg_model": BkgTemplateTraining.req(self, train_random_seed=0),
            "preprocessed_data": PreprocessingFoldwModelBGen.req(self),
        }

    def output(self):
        return {
            "log_B_trainval": self.local_target("log_B_trainval.npy"),
            "log_B_test": self.local_target("log_B_test.npy"),
            "truth_label_trainval": self.local_target("truth_label_trainval.npy"),
            "truth_label_test": self.local_target("truth_label_test.npy"),
        }

    @law.decorator.safe_output
    def run(self):
        # load the models
        from src.models.model_B import DensityEstimator, anode

        config_file = os.path.join("src", "models", "DE_MAF_model.yml")

        model_B = DensityEstimator(config_file, eval_mode=True, device=self.device)
        best_model_dir = self.input()["bkg_model"]["bkg_model"].path
        model_B.model.load_state_dict(torch.load(best_model_dir))
        model_B.model.to(self.device)
        model_B.model.eval()

        # load the sample to compare with
        data_trainval_SR_B = np.load(
            self.input()["preprocessed_data"]["SR_data_trainval_model_B"].path
        )
        trainvaltensor_SR_B = torch.from_numpy(data_trainval_SR_B.astype("float32")).to(
            self.device
        )

        truth_label_trainval = data_trainval_SR_B[:, -1]

        data_test_SR_B = np.load(
            self.input()["preprocessed_data"]["SR_data_test_model_B"].path
        )
        testtensor_SR_B = torch.from_numpy(data_test_SR_B.astype("float32")).to(
            self.device
        )

        truth_label_test = data_test_SR_B[:, -1]

        # get avg probility of 10 models
        log_B_trainval_list = []
        log_B_test_list = []
        with torch.no_grad():
            log_B_trainval = model_B.model.log_probs(
                inputs=trainvaltensor_SR_B[:, 1:-1],
                cond_inputs=trainvaltensor_SR_B[:, 0].reshape(-1, 1),
            )
            # set all nans to 0
            log_B_trainval[torch.isnan(log_B_trainval)] = 0
            log_B_trainval_list.append(log_B_trainval.cpu().numpy())

            log_B_test = model_B.model.log_probs(
                inputs=testtensor_SR_B[:, 1:-1],
                cond_inputs=testtensor_SR_B[:, 0].reshape(-1, 1),
            )
            # set all nans to 0
            log_B_test[torch.isnan(log_B_test)] = 0
            log_B_test_list.append
            log_B_test_list.append(log_B_test.cpu().numpy())

        log_B_trainval = np.array(log_B_trainval_list)
        B_trainval = np.exp(log_B_trainval).mean(axis=0)
        log_B_trainval = np.log(B_trainval + 1e-32)

        log_B_test = np.array(log_B_test_list)
        B_test = np.exp(log_B_test).mean(axis=0)
        log_B_test = np.log(B_test + 1e-32)

        self.output()["log_B_trainval"].parent.touch()
        np.save(self.output()["log_B_trainval"].path, log_B_trainval)
        np.save(self.output()["log_B_test"].path, log_B_test)
        np.save(self.output()["truth_label_trainval"].path, truth_label_trainval)
        np.save(self.output()["truth_label_test"].path, truth_label_test)

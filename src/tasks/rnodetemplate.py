import os, sys
import importlib
import luigi
import law
import numpy as np
import pandas as pd
import json

from src.utils.law import (
    BaseTask,
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    TemplateRandomMixin,
    SigTemplateTrainingUncertaintyMixin,
    ProcessMixin,
    BkgModelMixin,
)
from src.tasks.preprocessing import PreprocessingFold
from src.tasks.bkgtemplate import PredictBkgProb
from src.utils.utils import NumpyEncoder, str_encode_value
from src.tasks.bkgsampling import PredictBkgProbGen, PreprocessingFoldwModelBGen


class RNodeTemplate(
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    TemplateRandomMixin,
    BkgModelMixin,
    ProcessMixin,
    BaseTask,
):
    """Task for training R-Anode signal model templates.
    
    This task implements the core R-Anode signal model training described
    in Section II of the paper. It trains normalizing flow models to learn
    the signal density p_sig(x,m) by fitting to the residual component
    after subtracting the fixed background model from the data.
    
    This is the central innovation of R-Anode: fitting directly to the
    signal component while holding the background model fixed, rather
    than fitting to the full data density as in the original ANODE approach.
    
    Attributes
    ----------
    device : luigi.Parameter, default='cuda:0'
        Device for model training
    batchsize : luigi.IntParameter, default=2048
        Batch size for training
    epoches : luigi.IntParameter, default=50
        Number of training epochs
    early_stopping_patience : luigi.IntParameter, default=10
        Patience for early stopping
        
    Notes
    -----
    Inherits from multiple mixins providing fold splitting, template
    uncertainty, background model configuration. Central to the R-Anode 
    methodology for residual signal density estimation.
    Now uses all available signals (no signal fraction control).
    """

    device = luigi.Parameter(default="cuda:0")
    batchsize = luigi.IntParameter(default=2048)
    epoches = luigi.IntParameter(default=50)
    early_stopping_patience = luigi.IntParameter(default=10)

    def store_parts(self):
        return super().store_parts() + ("all_signals",)

    def requires(self):

        if self.use_bkg_model_gen_data:
            return {
                "preprocessed_data": PreprocessingFoldwModelBGen.req(self),
                "bkgprob": PredictBkgProbGen.req(self),
            }
        else:
            return {
                "preprocessed_data": PreprocessingFold.req(self),
                "bkgprob": PredictBkgProb.req(self),
            }

    def output(self):
        return {
            "sig_model": self.local_target(f"model_S.pt"),
            "trainloss_list": self.local_target("trainloss_list.npy"),
            "valloss_list": self.local_target("valloss_list.npy"),
            "metadata": self.local_target("metadata.json"),
        }

    @law.decorator.safe_output
    def run(self):

        input_dict = {
            "preprocessing": {
                "data_trainval_SR_model_S": self.input()["preprocessed_data"][
                    "SR_data_trainval_model_S"
                ],
                "data_trainval_SR_model_B": self.input()["preprocessed_data"][
                    "SR_data_trainval_model_B"
                ],
                "SR_mass_hist": self.input()["preprocessed_data"]["SR_mass_hist"],
            },
            "bkgprob": {
                "log_B_trainval": self.input()["bkgprob"]["log_B_trainval"],
            },
        }

        print(
            f"train model S with train random seed {self.train_random_seed}, sample fold {self.fold_split_seed}"
        )
        from src.models.train_model_S import train_model_S

        train_model_S(
            input_dict,
            self.output(),
            1.0,  # Fixed: use all signals (s_ratio = 1.0)
            1.0,  # Fixed: use all signals (w_value = 1.0)
            self.batchsize,
            self.epoches,
            self.early_stopping_patience,
            self.train_random_seed,
            self.device,
        )


class ScanRANODEFixedSeed(
    TemplateRandomMixin,
    SigTemplateTrainingUncertaintyMixin,
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    BkgModelMixin,
    ProcessMixin,
    BaseTask,
):
    """Simplified R-Anode model training without signal fraction scanning.
    
    Trains a single R-Anode model using all available signals mixed with background.
    No longer scans over different signal fractions.
    """

    def requires(self):
        # Just train one model with all signals
        return {
            "model": RNodeTemplate.req(
                self,
                train_random_seed=self.train_random_seed,
            )
        }

    def output(self):
        return {
            "model_list": self.local_target("model_list.json"),
        }

    @law.decorator.safe_output
    def run(self):
        # Simplified: just save the single model path
        model_path = [self.input()["model"]["sig_model"].path]
        model_path_list = {"scan_index_0": model_path}

        self.output()["model_list"].parent.touch()
        with open(self.output()["model_list"].path, "w") as f:
            json.dump(model_path_list, f, cls=NumpyEncoder)


class ScanRANODE(
    SigTemplateTrainingUncertaintyMixin,
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    BkgModelMixin,
    ProcessMixin,
    BaseTask,
):
    """Task for performing R-Anode model evaluation.
    
    Simplified version that trains and evaluates R-Anode models without
    signal fraction scanning. All signals are mixed with background.
    
    Attributes
    ----------
    device : luigi.Parameter, default='cuda'
        Device for model evaluation
        
    Notes
    -----
    Central to the R-Anode methodology but simplified to use all signals
    without scanning over different signal fractions.
    """

    device = luigi.Parameter(default="cuda")

    def requires(self):
        model_results = {}

        for index in range(self.train_num_sig_templates):
            model_results[f"model_seed_{index}"] = ScanRANODEFixedSeed.req(
                self, train_random_seed=index
            )

        if self.use_bkg_model_gen_data:
            return {
                "model_S_scan_result": model_results,
                "data": PreprocessingFoldwModelBGen.req(self),
                "bkgprob": PredictBkgProbGen.req(self),
            }
        else:
            return {
                "model_S_scan_result": model_results,
                "data": PreprocessingFold.req(self),
                "bkgprob": PredictBkgProb.req(self),
            }

    def output(self):
        return {
            "prob_S_scan": self.local_target("prob_S_scan.npy"),
            "prob_B_scan": self.local_target("prob_B_scan.npy"),
        }

    @law.decorator.safe_output
    def run(self):
        from src.models.ranode_pred import ranode_pred

        prob_S_list = []

        # Simplified: evaluate single model (no scanning)
        print("Evaluating R-Anode model with all signals")

        # for each random seed, load the model, evaluate the model on test data, and save the prob_S
        for index in range(self.train_num_sig_templates):
            # data path
            test_data_path = self.input()["data"]

            # prob B path
            bkg_prob_test_path = self.input()["bkgprob"]["log_B_test"]

            # model list
            model_list_path = self.input()["model_S_scan_result"][
                f"model_seed_{index}"
            ]["model_list"].path

            print(model_list_path)
            with open(model_list_path, "r") as f:
                model_scan_dict = json.load(f)

            model_list = model_scan_dict["scan_index_0"]  # Use the single model

            prob_S, prob_B = ranode_pred(
                model_list, test_data_path, bkg_prob_test_path, device=self.device
            )

            # prob_S shape is (num_models, num_samples), prob_B shape is (num_samples,)
            if len(prob_S_list) == 0:
                prob_S_list = prob_S
            else:
                prob_S_list = np.concatenate([prob_S_list, prob_S], axis=0)

        # Create single-point "scan" for compatibility with downstream tasks
        prob_S_scan = np.array([prob_S_list])  # Shape: (1, num_models, num_samples)
        prob_B_scan = prob_B

        self.output()["prob_S_scan"].parent.touch()
        np.save(self.output()["prob_S_scan"].path, prob_S_scan)
        np.save(self.output()["prob_B_scan"].path, prob_B_scan)

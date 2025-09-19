import os
import subprocess
import numpy as np
import luigi
import law
import pandas as pd

from src.utils.utils import str_encode_value


class BaseTask(law.Task):
    """Base task class providing common functionality for R-Anode workflow tasks.
    
    This class serves as the foundation for all R-Anode computational tasks,
    providing standardized path management, versioning, and output directory
    structure. It extends the Luigi/Law task framework for workflow management.
    
    Attributes
    ----------
    version : law.Parameter
        Version identifier for the task execution
    use_full_stats : luigi.BoolParameter, default=False
        Whether to use full statistical analysis or simplified version
    """

    version = law.Parameter()

    use_full_stats = luigi.BoolParameter(default=False)

    def store_parts(self):
        """Generate standardized directory structure components.
        
        Returns
        -------
        tuple
            Tuple of path components for organizing task outputs
        """
        task_name = self.__class__.__name__
        return (
            os.getenv("OUTPUT_DIR"),
            f"version_{self.version}",
            task_name,
            f"use_full_stats_{self.use_full_stats}",
        )

    def local_path(self, *path):
        """Generate local file system path for task outputs.
        
        Parameters
        ----------
        *path : str
            Additional path components to append
            
        Returns
        -------
        str
            Complete local file system path
        """
        sp = self.store_parts()
        sp += path
        return os.path.join(*(str(p) for p in sp))

    def local_target(self, *path, **kwargs):
        """Create a local file target for task output.
        
        Parameters
        ----------
        *path : str
            Path components for the target file
        **kwargs : dict
            Additional arguments passed to LocalFileTarget
            
        Returns
        -------
        law.LocalFileTarget
            File target object for the specified path
        """
        return law.LocalFileTarget(self.local_path(*path), **kwargs)

    def local_directory_target(self, *path, **kwargs):
        """Create a local directory target for task output.
        
        Parameters
        ----------
        *path : str
            Path components for the target directory
        **kwargs : dict
            Additional arguments passed to LocalDirectoryTarget
            
        Returns
        -------
        law.LocalDirectoryTarget
            Directory target object for the specified path
        """
        return law.LocalDirectoryTarget(self.local_path(*path), **kwargs)


class ProcessMixin:
    """Mixin class for tasks involving signal processing.
    
    This mixin provides parameters for the R-Anode analysis using
    gravitational wave signals with fixed characteristics.
    
    Attributes
    ----------
    ensemble : luigi.IntParameter, default=1
        Ensemble index for statistical uncertainty estimation
    """

    ensemble = luigi.IntParameter(default=1)

    def store_parts(self):
        return super().store_parts() + (
            f"ensemble_{self.ensemble}",
        )


class TemplateRandomMixin:
    """Mixin class for controlling randomness in template training.
    
    This mixin manages random seed parameters for reproducible template
    generation in the R-Anode workflow. Ensures consistent results across
    multiple runs while allowing systematic uncertainty studies.
    
    Attributes
    ----------
    train_random_seed : luigi.IntParameter, default=233
        Random seed for template training reproducibility
    """

    train_random_seed = luigi.IntParameter(default=233)

    def store_parts(self):
        return super().store_parts() + (f"train_seed_{self.train_random_seed}",)


class FoldSplitRandomMixin:
    """Mixin class for controlling fold splitting randomness.
    
    This mixin manages random seeds for data splitting in cross-validation
    and uncertainty estimation procedures used in R-Anode analysis.
    
    Attributes
    ----------
    fold_split_seed : luigi.IntParameter, default=0
        Random seed for data fold splitting
    """

    fold_split_seed = luigi.IntParameter(default=0)

    def store_parts(self):
        return super().store_parts() + (f"fold_split_seed_{self.fold_split_seed}",)


class FoldSplitUncertaintyMixin:
    """Mixin class for controlling uncertainty estimation through data splitting.
    
    This mixin manages the number of data splits used for statistical
    uncertainty estimation in R-Anode analysis, implementing the bootstrap
    and cross-validation strategies described in the R-Anode paper.
    
    Attributes
    ----------
    fold_split_num : luigi.IntParameter, default=5
        Number of data splits for uncertainty estimation
    """

    # controls how many times we split the data for uncertainty estimation
    fold_split_num = luigi.IntParameter(default=5)

    def store_parts(self):
        return super().store_parts() + (f"fold_split_num_{self.fold_split_num}",)


class BkgTemplateUncertaintyMixin:
    """Mixin class for background template uncertainty estimation.
    
    This mixin controls the number of background templates used in R-Anode
    to estimate systematic uncertainties in the background model. Multiple
    templates allow assessment of background modeling uncertainties.
    
    Attributes
    ----------
    num_bkg_templates : luigi.IntParameter, default=1
        Number of background templates for uncertainty estimation
    """

    num_bkg_templates = luigi.IntParameter(default=1)

    def store_parts(self):
        return super().store_parts() + (f"num_templates_{self.num_bkg_templates}",)


class BkgModelMixin:
    """Mixin class for background model configuration in R-Anode.
    
    This mixin provides options for using different background model
    configurations: perfect simulation-based models, data-driven models
    learned from sidebands, or models used for data generation.
    
    Attributes
    ----------
    use_perfect_bkg_model : luigi.BoolParameter, default=False
        Whether to use perfect simulation-based background model
    use_bkg_model_gen_data : luigi.BoolParameter, default=False
        Whether to use background model for data generation
    """

    use_perfect_bkg_model = luigi.BoolParameter(default=False)

    use_bkg_model_gen_data = luigi.BoolParameter(default=False)

    def store_parts(self):

        # use perfect bkg model and use bkg model to generate data cannot both be true
        assert not (
            self.use_perfect_bkg_model and self.use_bkg_model_gen_data
        ), "use_perfect_bkg_model and use_bkg_model_gen_data cannot both be true"

        return super().store_parts() + (
            f"use_perfect_bkg_model_{self.use_perfect_bkg_model}",
            f"use_bkg_model_gen_data_{self.use_bkg_model_gen_data}",
        )


class SigTemplateTrainingUncertaintyMixin:
    """Mixin class for signal template training uncertainty estimation.
    
    This mixin controls the number of signal templates trained with different
    random initializations to assess training uncertainties in the R-Anode
    signal model fitting procedure.
    
    Attributes
    ----------
    train_num_sig_templates : luigi.IntParameter, default=1
        Number of signal templates for training uncertainty estimation
    """

    # controls the random seed for the training
    train_num_sig_templates = luigi.IntParameter(default=1)

    def store_parts(self):
        return super().store_parts() + (
            f"train_num_templates_{self.train_num_sig_templates}",
        )



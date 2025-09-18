import numpy as np
from sklearn.utils import shuffle


def get_dijetmass_ptetaphi(jets):
    """Calculate dijet invariant mass from pt-eta-phi-mass coordinates.
    
    This function computes the invariant mass of the two-jet system using
    the standard relativistic four-momentum addition, taking input jets
    in pt-eta-phi-mass coordinate system as used in the LHCO dataset.
    
    Parameters
    ----------
    jets : array-like, shape (n_events, 2, 4)
        Jet four-momentum data where jets[i,j,k] represents:
        - i: event index
        - j: jet index (0 or 1)
        - k: coordinate index (0=pt, 1=eta, 2=phi, 3=mass)
        
    Returns
    -------
    numpy.ndarray, shape (n_events,)
        Dijet invariant mass for each event
        
    Notes
    -----
    Converts pt-eta-phi coordinates to px-py-pz-E and computes:
    m_jj = sqrt(E^2 - px^2 - py^2 - pz^2)
    where E, px, py, pz are the total four-momentum components.
    """
    jet_e = np.sqrt(
        jets[:, 0, 3] ** 2 + jets[:, 0, 0] ** 2 * np.cosh(jets[:, 0, 1]) ** 2
    )
    jet_e += np.sqrt(
        jets[:, 1, 3] ** 2 + jets[:, 1, 0] ** 2 * np.cosh(jets[:, 1, 1]) ** 2
    )
    jet_px = jets[:, 0, 0] * np.cos(jets[:, 0, 2]) + jets[:, 1, 0] * np.cos(
        jets[:, 1, 2]
    )
    jet_py = jets[:, 0, 0] * np.sin(jets[:, 0, 2]) + jets[:, 1, 0] * np.sin(
        jets[:, 1, 2]
    )
    jet_pz = jets[:, 0, 0] * np.sinh(jets[:, 0, 1]) + jets[:, 1, 0] * np.sinh(
        jets[:, 1, 1]
    )
    mjj = np.sqrt(np.abs(jet_px**2 + jet_py**2 + jet_pz**2 - jet_e**2))
    return mjj


def get_dijetmass_pxyz(jets):
    """Calculate dijet invariant mass from px-py-pz-mass coordinates.
    
    This function computes the invariant mass of the two-jet system using
    direct px-py-pz coordinates, as used in some dataset formats where
    jets are already provided in Cartesian momentum coordinates.
    
    Parameters
    ----------
    jets : array-like, shape (n_events, 2, 4)
        Jet four-momentum data where jets[i,j,k] represents:
        - i: event index  
        - j: jet index (0 or 1)
        - k: coordinate index (0=px, 1=py, 2=pz, 3=mass)
        
    Returns
    -------
    numpy.ndarray, shape (n_events,)
        Dijet invariant mass for each event
        
    Notes
    -----
    Computes total energy from E = sqrt(px^2 + py^2 + pz^2 + m^2)
    and then invariant mass as m_jj = sqrt(E_tot^2 - p_tot^2).
    """
    jet_e = np.sqrt(
        jets[:, 0, 3] ** 2
        + jets[:, 0, 0] ** 2
        + jets[:, 0, 1] ** 2
        + jets[:, 0, 2] ** 2
    )
    jet_e += np.sqrt(
        jets[:, 1, 3] ** 2
        + jets[:, 1, 0] ** 2
        + jets[:, 1, 1] ** 2
        + jets[:, 1, 2] ** 2
    )
    jet_px = jets[:, 0, 0] + jets[:, 1, 0]
    jet_py = jets[:, 0, 1] + jets[:, 1, 1]
    jet_pz = jets[:, 0, 2] + jets[:, 1, 2]
    mjj = np.sqrt(np.abs(jet_px**2 + jet_py**2 + jet_pz**2 - jet_e**2))
    return mjj


def standardize(x, mean, std):
    """Apply z-score standardization to input data.
    
    Parameters
    ----------
    x : array-like
        Input data to standardize
    mean : array-like
        Mean values for each feature  
    std : array-like
        Standard deviation values for each feature
        
    Returns
    -------
    array-like
        Standardized data with zero mean and unit variance
    """
    return (x - mean) / std


def logit_transform(x, min_vals, max_vals):
    """Apply logit transformation to bounded data.
    
    This transformation maps bounded variables to the real line using
    the logit function, which is useful for normalizing flow training
    where unbounded support is required.
    
    Parameters
    ----------
    x : array-like
        Input data to transform (must be within [min_vals, max_vals])
    min_vals : array-like
        Minimum values for each feature
    max_vals : array-like  
        Maximum values for each feature
        
    Returns
    -------
    tuple
        (transformed_data, valid_mask) where:
        - transformed_data: logit-transformed values
        - valid_mask: boolean mask indicating valid (finite) values
        
    Notes
    -----
    Applies the transformation: logit((x - min)/(max - min))
    Invalid values (resulting in NaN or inf) are masked out.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        x_norm = (x - min_vals) / (max_vals - min_vals)
        logit = np.log(x_norm / (1 - x_norm))
    domain_mask = ~(np.isnan(logit).any(axis=1) | np.isinf(logit).any(axis=1))
    return logit, domain_mask


def inverse_logit_transform(x, min_vals, max_vals):
    """Apply inverse logit transformation to recover bounded data.
    
    This function reverses the logit transformation to map data from
    the real line back to the original bounded domain.
    
    Parameters
    ----------
    x : array-like
        Logit-transformed data to invert
    min_vals : array-like
        Minimum values for each feature in original space
    max_vals : array-like
        Maximum values for each feature in original space
        
    Returns
    -------
    array-like
        Data transformed back to original bounded domain
        
    Notes
    -----
    Applies: sigmoid(x) * (max - min) + min
    """
    x_norm = 1 / (1 + np.exp(-x))
    return x_norm * (max_vals - min_vals) + min_vals


def inverse_standardize(x, mean, std):
    """Reverse z-score standardization to recover original scale.
    
    Parameters
    ----------
    x : array-like
        Standardized data to invert
    mean : array-like
        Mean values used in original standardization
    std : array-like
        Standard deviation values used in original standardization
        
    Returns
    -------
    array-like
        Data in original scale
    """
    return x * std + mean


def preprocess_params_fit(data):
    """Fit preprocessing parameters for normalizing flow training.
    
    This function computes the preprocessing parameters needed to transform
    data for normalizing flow training in R-Anode. It applies logit 
    transformation followed by standardization to prepare features.
    
    Parameters
    ----------
    data : array-like, shape (n_events, n_features)
        Input data where columns are [mjj, feature1, ..., featureN, label]
        
    Returns
    -------
    dict
        Dictionary containing preprocessing parameters:
        - 'min': minimum values for each feature
        - 'max': maximum values for each feature  
        - 'mean': mean values after logit transformation
        - 'std': standard deviations after logit transformation
        
    Notes
    -----
    Excludes the first column (invariant mass) and last column (label)
    from preprocessing, as these are handled separately in R-Anode.
    """
    preprocessing_params = {}
    preprocessing_params["min"] = np.min(data[:, 1:-1], axis=0)
    preprocessing_params["max"] = np.max(data[:, 1:-1], axis=0)

    preprocessed_data_x, mask = logit_transform(
        data[:, 1:-1], preprocessing_params["min"], preprocessing_params["max"]
    )
    preprocessed_data = np.hstack([data[:, 0:1], preprocessed_data_x, data[:, -1:]])[
        mask
    ]

    preprocessing_params["mean"] = np.mean(preprocessed_data[:, 1:-1], axis=0)
    preprocessing_params["std"] = np.std(preprocessed_data[:, 1:-1], axis=0)

    return preprocessing_params


def preprocess_params_transform(data, params):
    """Apply preprocessing transformation using fitted parameters.
    
    This function applies the preprocessing transformation to new data
    using parameters computed by preprocess_params_fit, ensuring
    consistent preprocessing between training and evaluation data.
    
    Parameters
    ----------
    data : array-like, shape (n_events, n_features)
        Input data to transform
    params : dict
        Preprocessing parameters from preprocess_params_fit
        
    Returns
    -------
    array-like
        Preprocessed data ready for normalizing flow training/evaluation
        
    Notes
    -----
    Applies logit transformation followed by standardization using
    the fitted parameters. Events with invalid transformations are
    automatically filtered out.
    """
    preprocessed_data_x, mask = logit_transform(
        data[:, 1:-1], params["min"], params["max"]
    )
    preprocessed_data = np.hstack([data[:, 0:1], preprocessed_data_x, data[:, -1:]])[
        mask
    ]
    preprocessed_data[:, 1:-1] = standardize(
        preprocessed_data[:, 1:-1], params["mean"], params["std"]
    )
    return preprocessed_data


def inverse_transform(data, params):
    """Apply inverse preprocessing to recover original feature space.
    
    This function reverses the preprocessing applied for normalizing flow
    training, transforming data from the standardized logit space back
    to the original physical feature space.
    
    Parameters
    ----------
    data : array-like, shape (n_events, n_features)
        Preprocessed data to invert
    params : dict
        Preprocessing parameters used for the original transformation
        
    Returns
    -------
    array-like
        Data in original physical feature space
        
    Notes
    -----
    Applies inverse standardization followed by inverse logit
    transformation to recover the original bounded feature values.
    """
    inverse_data = inverse_standardize(
        data[:, 1:-1], np.array(params["mean"]), np.array(params["std"])
    )
    inverse_data = inverse_logit_transform(
        inverse_data, np.array(params["min"]), np.array(params["max"])
    )
    inverse_data = np.hstack([data[:, 0:1], inverse_data, data[:, -1:]])

    return inverse_data


def fold_splitting(
    data,
    n_folds=5,
    random_seed=42,
    test_fold=0,
):
    """Split data into training/validation and test sets using k-fold strategy.
    
    This function implements the k-fold data splitting strategy used in
    R-Anode for statistical uncertainty estimation, as described in the
    paper's methodology for robust evaluation.
    
    Parameters
    ----------
    data : array-like
        Input data to split
    n_folds : int, default=5
        Number of folds for cross-validation
    random_seed : int, default=42
        Random seed for reproducible splitting
    test_fold : int, default=0
        Which fold to use as test set (0 to n_folds-1)
        
    Returns
    -------
    tuple
        (data_trainval, data_test) where:
        - data_trainval: training/validation data (n_folds-1 folds)
        - data_test: test data (1 fold)
        
    Notes
    -----
    The test fold is held out completely, while the remaining folds
    are combined for training and validation. This enables systematic
    uncertainty estimation across multiple data splits.
    """
    np.random.seed(random_seed)

    data = shuffle(data, random_state=random_seed)

    # split into fold_split_num folds, test fold is fold_split_seed-th fold
    data_folds = {}
    for fold in range(n_folds):
        data_folds[fold] = data[
            fold * int(len(data) / n_folds) : (fold + 1) * int(len(data) / n_folds)
        ]

    # get the signal trainval and test set index
    sig_test_index = test_fold
    sig_trainval_index_list = [i for i in range(n_folds) if i != sig_test_index]

    data_test = data_folds[sig_test_index]
    data_trainval = np.concatenate([data_folds[i] for i in sig_trainval_index_list])

    return data_trainval, data_test

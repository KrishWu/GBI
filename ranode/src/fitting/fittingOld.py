import numpy as np
import json
from tqdm import tqdm
from array import array
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, RationalQuadratic
from scipy.optimize import fmin_l_bfgs_b
from src.utils.utils import NumpyEncoder
from src.utils.utils import find_zero_crossings


def fit_likelihood(
    x_values,
    y_values_mean,
    y_values_std,
    w_true,
    events_num,
    output_path,
    logbased=True,
):
    """Fit likelihood function using Gaussian Process regression.
    
    This function performs the likelihood fitting step described in the R-Anode
    paper, using Gaussian Process regression to interpolate between discrete
    likelihood evaluations and find confidence intervals for signal strength
    parameters.
    
    Parameters
    ----------
    x_values : array-like
        Signal fraction values where likelihood was evaluated (usually log10 scale)
    y_values_mean : array-like
        Mean likelihood values at each x_value
    y_values_std : array-like
        Standard deviation of likelihood values at each x_value
    w_true : float
        True signal fraction value for comparison
    events_num : int
        Total number of events, used for confidence interval calculation
    output_path : str
        Path for saving the likelihood fit plot
    logbased : bool, default=True
        Whether x_values are in log10 scale
        
    Returns
    -------
    dict
        Dictionary containing fit results including:
        - mu_pred: Predicted signal fraction at likelihood maximum
        - true_mu: True signal fraction value
        - best_model_index: Index of best-fit model
        - x_pred, y_pred: Fine-grid predictions from GP
        - sigma: Uncertainty estimates from GP
        - CI_95_likelihood: 95% confidence interval threshold
        - Various raw input arrays
        
    Notes
    -----
    Uses RBF kernel with constant scale for Gaussian Process regression.
    Confidence intervals computed using Wilks' theorem with log(2)/N threshold.
    """

    x_values = x_values.reshape(-1, 1)

    # define the kernel
    kernel = C(1.0, (1e-3, 1e3)) * RBF(
        1e-3, (1e-5, 1e2)
    )  # + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-10, 1e+1))
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=y_values_std**2, n_restarts_optimizer=100
    )
    gp.fit(x_values, y_values_mean)

    # --- Make Predictions on a Fine Grid ---
    x_pred = np.linspace(x_values.min(), x_values.max(), 1001).reshape(-1, 1)
    y_pred, sigma = gp.predict(x_pred, return_std=True)

    # --- Plot the Fit ---
    x_pred = x_pred.flatten()
    y_pred = y_pred.flatten()
    sigma = sigma.flatten()

    y_lower_bound = y_pred - 1.96 * sigma
    y_upper_bound = y_pred + 1.96 * sigma

    # find peak of the likelihood
    arg_max_likelihood = np.argmax(y_pred)
    max_likelihood = y_pred[arg_max_likelihood]

    if logbased:
        mu_pred = np.power(10, x_pred[arg_max_likelihood])
        mu_true = np.power(10, w_true)
    else:
        mu_pred = x_pred[arg_max_likelihood]
        mu_true = w_true
    # get the cloest mu value to the pred mu in x_values
    best_model_index = np.argmin(np.abs(x_values - x_pred[arg_max_likelihood]))

    # ------------------------------- 95% CI of max likelihood -------------------------------
    # given x_pred, y_pred, and 95CI likelihood drop = np.log(2)/events_num, find the first left
    # intersection of max likelihood - drop with y_pred as the lower bound of the 95% CI
    CI_95_likelihood = max_likelihood - np.log(2) / events_num

    # ---------------------------------------------------------------------------------------

    with PdfPages(output_path) as pdf:
        f = plt.figure(figsize=(10, 8))
        plt.scatter(
            x_values.flatten(), y_values_mean, label="test points", color="black"
        )
        plt.errorbar(
            x_values.flatten(), y_values_mean, yerr=y_values_std, fmt="o", color="black"
        )
        plt.plot(x_pred, y_pred, label="fit func", color="red")
        plt.fill_between(
            x_pred, y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color="red"
        )

        # plot 95% CI and the peak
        plt.scatter(
            [x_pred[arg_max_likelihood]],
            [max_likelihood],
            color="red",
            label=f"peak $\mu$ {mu_pred:.4f}",
        )  # peak w value
        plt.axhline(y=CI_95_likelihood, color="blue", linestyle="--")
        # true w value
        plt.axvline(
            x=w_true, color="black", linestyle="--", label=f"true $\mu$ {mu_true:.4f}"
        )

        plt.title(f"Likelihood fit at true $\mu$ {mu_true:.4f}")

        if logbased:
            plt.xlabel("$log_{10}(\mu)$")
        else:
            plt.xlabel("$\mu$")

        plt.ylabel("likelihood")

        plt.legend()
        pdf.savefig(f)
        plt.close()

    output_metadata = {
        "mu_pred": mu_pred,
        "true_mu": mu_true,
        "best_model_index": best_model_index,
        "x_pred": x_pred,
        "y_pred": y_pred,
        "sigma": sigma,
        "CI_95_likelihood": CI_95_likelihood,
        "CI_95_likelihood_drop": np.log(2) / events_num,
        "x_raw": x_values.flatten(),
        "y_raw": y_values_mean,
        "y_raw_std": y_values_std,
    }

    return output_metadata


def bootstrap_and_fit(
    prob_S_list,
    prob_B_list,
    mu_scan_values,
    mu_true,
    output_dir,
    bootstrap_num=100,
):
    """Perform bootstrap uncertainty estimation and likelihood fitting.
    
    This function implements the bootstrap procedure described in the R-Anode
    paper for estimating uncertainties in the likelihood function. It performs
    multiple bootstrap samples of the signal models, computes likelihood at
    each scan point, and fits a Gaussian Process to obtain confidence intervals.
    
    Parameters
    ----------
    prob_S_list : array-like, shape (num_scan_points, num_models, num_events)
        Signal probability predictions for different mu values and models
    prob_B_list : array-like, shape (num_scan_points, num_events)  
        Background probability predictions for different mu values
    mu_scan_values : array-like
        Array of signal fraction values where likelihood was evaluated
    mu_true : float
        True signal fraction for comparison and validation
    output_dir : dict
        Dictionary containing output file paths for plots and results
    bootstrap_num : int, default=100
        Number of bootstrap samples for uncertainty estimation
        
    Returns
    -------
    None
        Results are saved to files specified in output_dir
        
    Notes
    -----
    Implements the R-Anode likelihood construction from Equation (3) in the paper:
    L = -⟨log pdata(x,m)⟩ where pdata = w*psig + (1-w)*pbg
    
    Uses RBF kernel Gaussian Process with L-BFGS-B optimization for robust
    likelihood surface fitting. Confidence intervals computed using Wilks'
    theorem with Δlog(L) = log(2)/N for 95% intervals.
    """

    # prob_S_list has shape (num_scan_points, num_models, num_events)
    # prob_B_list has shape (num_scan_points, num_events)

    event_num = prob_S_list.shape[-1]

    # compute the nominal likelihood
    prob_S_nominal = prob_S_list.mean(axis=1)  # shape is (num_scan_points, num_events)
    prob_B_nominal = prob_B_list  # shape is (num_scan_points, num_events)
    likelihood_nominal = (
        mu_scan_values.reshape(-1, 1) * prob_S_nominal
        + (1 - mu_scan_values.reshape(-1, 1)) * prob_B_nominal
    )
    log_likelihood_nominal = np.log(likelihood_nominal)
    log_likelihood_nominal_mean = log_likelihood_nominal.mean(axis=1)

    # Now bootstrap classifiers in model_S to get the uncertainty in the likelihood
    bootstrap_model_S_index = np.random.choice(
        len(prob_S_list[0]), size=(bootstrap_num, len(prob_S_list[0])), replace=True
    )
    # bootstrap_model_B_index = np.random.choice(len(prob_B_list[0]), size=(bootstrap_num, len(prob_B_list[0])), replace=True)
    bootstrap_log_likelihood = []

    for index in tqdm(range(bootstrap_num)):
        bootstrap_model_S_index_i = bootstrap_model_S_index[index]
        prob_S_bootstrap_i = prob_S_list[:, bootstrap_model_S_index_i].mean(axis=1)

        # bootstrap_model_B_index_i = bootstrap_model_B_index[index]
        # prob_B_bootstrap_i = prob_B_list[:, bootstrap_model_B_index_i].mean(axis=1)

        likelihood_bootstrap_i = (
            mu_scan_values.reshape(-1, 1) * prob_S_bootstrap_i
            + (1 - mu_scan_values.reshape(-1, 1)) * prob_B_nominal
        )
        log_likelihood_bootstrap_i = np.log(likelihood_bootstrap_i)
        log_likelihood_bootstrap_i_mean = log_likelihood_bootstrap_i.mean(axis=1)

        bootstrap_log_likelihood.append(log_likelihood_bootstrap_i_mean)

    bootstrap_log_likelihood = np.array(bootstrap_log_likelihood)
    log_likelihood_nominal_std = np.std(bootstrap_log_likelihood, axis=0)

    # -------------------- make fitting and save the info --------------------
    mu_scan_values_log = np.log10(mu_scan_values)
    x_values = mu_scan_values_log
    y_values = log_likelihood_nominal_mean

    x_values = x_values.reshape(-1, 1)

    def custom_optimizer(obj_func, initial_theta, bounds):
        xopt, fopt, _dict =  fmin_l_bfgs_b(
            obj_func,
            initial_theta,
            bounds=bounds,
            maxiter=10_000,      # ↑ increase iteration budget
        )
        return xopt, fopt
    
    # define the kernel
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.2, length_scale_bounds=(1e-3, 1e3))
    # kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=0.5,
    #                                                          length_scale_bounds=(0.1, 10),
    #                                                          alpha_bounds=(1e-4, 1e3))
        #  + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-12, 1e-3))
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=100, optimizer=custom_optimizer, alpha=log_likelihood_nominal_std**2
    )
    gp.fit(x_values, y_values)

    # --- Make Predictions on a Fine Grid ---
    x_pred = np.linspace(x_values.min(), x_values.max(), 1001).reshape(-1, 1)
    y_pred, sigma = gp.predict(x_pred, return_std=True)
    y_pred_lower_bound = y_pred - 1.96 * sigma
    y_pred_upper_bound = y_pred + 1.96 * sigma

    x_pred = x_pred.flatten()
    y_pred = y_pred.flatten()

    max_likelihood = y_pred.max()
    max_likelihood_index = np.argmax(y_pred)
    max_likelihood_w = np.power(10, x_pred[max_likelihood_index])
    max_likelihood_w_log = x_pred[max_likelihood_index]

    max_likelihood_lower_bound = y_pred_lower_bound[max_likelihood_index]

    # ------------------------------- 95% CI of max likelihood -------------------------------
    CI95_likelihood = max_likelihood - np.log(2) / event_num
    # add y_values to y_pred_upper_bound to make sure the 95CI is more conservative
    # and it needs to be added in where x_values are in x_pred
    # indices = np.searchsorted(x_pred, x_values.flatten())
    # y_likelihood_upper_bound = np.insert(y_pred_upper_bound, indices, y_values, axis=0)
    # x_likelihood_upper_bound = np.insert(x_pred, indices, x_values.flatten(), axis=0)
    diff = y_pred - CI95_likelihood
    crossings = find_zero_crossings(x_pred, diff)
    # Separate them into those on the left vs. right of the maximum
    left_crossings = [c for c in crossings if c < max_likelihood_w_log]
    right_crossings = [c for c in crossings if c > max_likelihood_w_log]
    # If there is more than one intersection on each side, we take the
    # minimum on the left and the maximum on the right to be conservative
    x_left = min(left_crossings) if left_crossings else None
    x_right = max(right_crossings) if right_crossings else None

    mu_left = 10**x_left if x_left is not None else 0
    mu_right = 10**x_right if x_right is not None else 1

    # if at the leftmost point we have y_likelihood_upper_bound > CI95_likelihood
    # then we need to set mu_left to 0
    if y_pred[0] > CI95_likelihood:
        mu_left = 0
    # if hit the left most test ppint, set it to be 0
    if mu_left <= 10**x_pred[0]:
        mu_left = 0

    # make plots
    f = plt.figure()

    plt.scatter(x_values.flatten(), y_values, label="test points", color="black")
    plt.errorbar(
        x_values.flatten(),
        y_values,
        yerr=log_likelihood_nominal_std,
        fmt="o",
        color="black",
    )

    plt.plot(x_pred, y_pred, label="fit func", color="red")
    plt.fill_between(
        x_pred, y_pred_lower_bound, y_pred_upper_bound, alpha=0.2, color="red"
    )

    # plot 95% CI and the peak
    plt.scatter(
        [max_likelihood_w_log],
        [max_likelihood],
        color="red",
        label=f"peak $\mu$ {max_likelihood_w:.4f}",
    )  # peak w value
    plt.axhline(
        y=CI95_likelihood,
        color="blue",
        linestyle="--",
        label=f"95% CI, [{mu_left:.4f}, {mu_right:.4f}]",
    )

    plt.axvline(
        x=np.log10(mu_true),
        color="black",
        linestyle="--",
        label=f"true $\mu$ {mu_true:.4f}",
    )

    plt.title(f"Likelihood fit at true $\mu$ {mu_true:.4f}")

    plt.xlabel("$log_{10}(\mu)$")

    # <log likelihood>_perevent
    plt.ylabel("$<log(L)>_{event}$")

    plt.tight_layout()

    plt.legend()
    plt.savefig(output_dir["scan_plot"].path)
    plt.close()

    output_info = {
        "mu_true": mu_true,
        "mu_pred": max_likelihood_w,
        "left_CI": mu_left,
        "right_CI": mu_right,
    }

    with open(output_dir["peak_info"].path, "w") as f:
        json.dump(output_info, f, cls=NumpyEncoder)
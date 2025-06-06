import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

from scipy.signal import savgol_filter, firwin, filtfilt
from scipy.special import expit

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score,
    confusion_matrix,
    make_scorer,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression

import pywt
import spectrochempy as scp
from orpl.baseline_removal import bubblefill
import pybaselines
import ast


# ----- Utility Functions ----- #
def find_closest_index(wavenumber: np.ndarray, target_wavenumber: float) -> int:
    """
    Finds the index of the wavenumber closest to the specified target.

    Args:
        wavenumber (np.ndarray): 1D array of wavenumbers.
        target_wavenumber (float): Wavenumber to match.

    Returns:
        int: Index of the closest wavenumber in the array.
    """
    return np.abs(wavenumber - target_wavenumber).argmin()


# ----- Region Selection ----- #
def region_selection(
    wavenumber: np.ndarray,
    data: np.ndarray,
    regions_to_delete: List[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes specified spectral regions from the data based on wavenumber range.

    Args:
        wavenumber (np.ndarray): 1D array of wavenumbers.
        data (np.ndarray): 2D array of spectral data.
        regions_to_delete (List[Tuple[float, float]]):
            List of (start_wavenumber, end_wavenumber) tuples to remove.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Filtered wavenumber and corresponding data arrays.
    """
    mask = np.ones(wavenumber.shape, dtype=bool)

    for start_wn, end_wn in regions_to_delete:
        start_idx = find_closest_index(wavenumber, start_wn)
        end_idx = find_closest_index(wavenumber, end_wn)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        mask[start_idx : end_idx + 1] = False

    return wavenumber[mask], data[:, mask]


# ----- Baseline Correction ----- #
def baseline_correction(
    data: np.ndarray, wavenumber: np.ndarray, method: str, **kwargs
) -> np.ndarray:
    """
    Applies baseline correction to spectral data using the specified method.

    Args:
        data (np.ndarray): 2D array of spectral data.
        wavenumber (np.ndarray): 1D array of wavenumbers.
        method (str): Baseline correction method -> 'fft', 'bubblefill',
        or any method in pybaselines
        **kwargs: Additional arguments for the baseline method.
            For 'fft', specify 'frequency' and 'cutoff'.
            For 'bubblefill', specify 'min_bubble_widths'.
            For pybaselines methods, pass method-specific parameters
            (https://pybaselines.readthedocs.io/)

    Returns:
        np.ndarray: Baseline-corrected spectra.
    """
    output_data = np.zeros_like(data)

    for i, y_orig in enumerate(data):
        if method is None:
            output_data[i, :] = y_orig
        elif method == "fft":
            frequency = kwargs.get("frequency")
            cutoff = kwargs.get("cutoff")
            sig_fft = np.fft.fft(y_orig)
            freq = np.fft.fftfreq(len(y_orig), d=1.0 / frequency)
            sig_fft[np.abs(freq) > cutoff] = 0
            bsl = np.real(np.fft.ifft(sig_fft))
        elif method == "bubblefill":
            min_bubble_width = kwargs.get("min_bubble_widths")
            _, bsl = bubblefill(y_orig, min_bubble_widths=min_bubble_width)
        elif hasattr(pybaselines.Baseline, method):
            bsl_fitter = pybaselines.Baseline(wavenumber, check_finite=False)
            bsl_func = getattr(bsl_fitter, method)
            bsl, _ = bsl_func(y_orig, **kwargs)
        else:
            raise ValueError(f"Unknown baseline method: {method}")

        output_data[i, :] = y_orig - bsl

    return output_data


# ----- Filtering ----- #
def filtering(data: np.ndarray, method: str, params: List[dict]) -> List[np.ndarray]:
    """
    Applies filtering to spectral data using methods such as SG, FFT, FIR,
    Whittaker, or Wavelet.

    Args:
        data (np.ndarray): 2D array of spectral data.
        method (str): Filtering method -> 'sg', 'fft', 'fir', 'whittaker', 'wavelet'.
        params (List[dict]): List of parameter dictionaries to apply per filter method.
            For 'fft', specify 'frequency' and 'cutoff'.
            For 'fir', specify 'cutoff' and 'n_taps'.
            For 'whittaker', specify 'order' and 'lamb'.
            For 'wavelet', specify 'wavelet', 'level',
            'threshold_method' (universal, sure, bayes), and optionally 'bayes_scale'.

    Returns:
        List[np.ndarray]: List of filtered data arrays for each parameter combination.
    """

    # Filter function for wavelet denoising based on thresholding
    def wavelet_filter(
        y: np.ndarray,
        wavelet: str,
        level: int,
        threshold_method: str,
        bayes_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Applies wavelet-based denoising with thresholding (universal, SURE, or Bayes).

        Args:
            y (np.ndarray): Input 1D spectral signal.
            wavelet (str): Wavelet type.
            level (int): Level of decomposition.
            threshold_method (str): 'universal', 'sure', 'bayes'.
            bayes_scale (float): Scale factor for BayesShrink thresholding.

        Returns:
            np.ndarray: Filtered 1D signal.
        """
        coeffs = pywt.wavedec(y, wavelet=wavelet, level=level)

        if threshold_method == "universal":
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            thresh = sigma * np.sqrt(2 * np.log(len(y)))
            coeffs[1:] = [pywt.threshold(c, thresh, mode="soft") for c in coeffs[1:]]

        elif threshold_method == "sure":
            for i in range(1, len(coeffs)):
                c = coeffs[i]
                n = len(c)
                sorted_c = np.sort(np.abs(c))
                risks = [
                    (n - 2 * (j + 1) 
                     + np.sum(sorted_c[: j + 1] ** 2) 
                     + np.sum(sorted_c[j + 1 :] ** 2)
                     ) / n for j in range(n)
                ]
                sure_thresh = sorted_c[np.argmin(risks)]
                coeffs[i] = pywt.threshold(c, sure_thresh, mode="soft")
        elif threshold_method == "bayes":
            for i in range(1, len(coeffs)):
                c = coeffs[i]
                var = np.var(c)
                sigma = np.median(np.abs(c)) / 0.6745
                bayes_thresh = bayes_scale * (sigma**2 / np.sqrt(var))
                coeffs[i] = pywt.threshold(c, bayes_thresh, mode="soft")

        return pywt.waverec(coeffs, wavelet=wavelet)[: len(y)]

    output_data = []
    for param in params:
        filtered = []
        for y in data:
            if method == "sg":
                y_filt = savgol_filter(y, param["window_size"], param["order"])

            elif method == "fft":
                freq = np.fft.fftfreq(len(y), d=1.0 / param["frequency"])
                y_fft = np.fft.fft(y)
                y_fft[np.abs(freq) > param["cutoff"]] = 0
                y_filt = np.real(np.fft.ifft(y_fft))

            elif method == "whittaker":
                filt = scp.Filter(
                    method="whittaker", order=param["order"], lamb=param["lamb"]
                )
                y_filt = filt(y).data

            elif method == "fir":
                coeffs = firwin(param["n_taps"], param["cutoff"])
                y_filt = filtfilt(coeffs, 1.0, y)

            elif method == "wavelet":
                y_filt = wavelet_filter(
                    y,
                    param["wavelet"],
                    param["level"],
                    param["threshold_method"],
                    param.get("bayes_scale", 1.0),
                )

            else:
                raise ValueError(f"Unknown filtering method: {method}")

            filtered.append(y_filt)
        output_data.append(np.array(filtered))

    return output_data


# ----- Normalization ----- #
def normalize(data: np.ndarray, params: List[str]) -> np.ndarray:
    """
    Applies normalization to spectral data using methods such as MSC, SNV, UVN, Min-max.

    Args:
        data (np.ndarray): 2D array of spectral data.
        params (List[str]): List of normalization method names.
            Supported methods: 'msc', 'min_max', 'snv', 'uvn'.

    Returns:
        np.ndarray: Normalized spectral data.
    """

    # Multiplicative Scatter/Signal Correction (MSC)
    def msc(x: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
        x_centered = x - x.mean(axis=1, keepdims=True)
        ref = np.mean(x_centered, axis=0) if reference is None else reference
        return np.array(
            [(xi - np.polyfit(ref, xi, 1)[1]) / np.polyfit(ref, xi, 1)[0] for xi in x]
        )

    # Min-Max Normalization
    def min_max(x: np.ndarray) -> np.ndarray:
        return (x - x.min(axis=1, keepdims=True)) / (
            x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True)
        )

    # Standard Normal Variate (SNV)
    def snv(x: np.ndarray) -> np.ndarray:
        return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)

    # Unit Vector Normalization (UVN)
    def uvn(x: np.ndarray) -> np.ndarray:
        return x / np.sqrt(np.sum(x**2, axis=1, keepdims=True))

    methods = {"msc": msc, "min_max": min_max, "snv": snv, "uvn": uvn}

    for param in params:
        if param not in methods:
            raise ValueError(f"Unsupported normalization method: {param}")
        data = methods[param](data)

    return data


# ----- Derivative Calculation ----- #
def derivative(
    data: np.ndarray, wavenumber: np.ndarray, params: List[dict]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Computes Savitzky-Golay derivatives of spectral data and removes edge regions.

    Args:
        data (np.ndarray): 2D array of spectral data.
        wavenumber (np.ndarray): 1D array of wavenumbers.
        params (List[dict]): List of dictionaries with keys 
                            'window_size', 'order', and 'deriv'.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: List of tuples with trimmed wavenumber 
                                             and derivative arrays.

    """
    output_data = []

    # minimum gap length to identify regions (region_selection function)
    min_gap_length = 10

    for param in params:
        deriv_data = savgol_filter(
            data,
            window_length=param["window_size"],
            polyorder=param["order"],
            deriv=param["deriv"],
            axis=1,
        )
        num_edge_points = param["window_size"] // 2

        # Identify regions by gaps
        regions = []
        start = 0
        for i in range(1, len(wavenumber)):
            if wavenumber[i] - wavenumber[i - 1] > min_gap_length:
                regions.append((start, i))
                start = i
        regions.append((start, len(wavenumber)))

        drop_indices = set()
        for start, end in regions:
            drop_indices.update(range(start, start + num_edge_points))
            drop_indices.update(range(end - num_edge_points, end))

        keep_indices = sorted(set(range(len(wavenumber))) - drop_indices)
        output_data.append((wavenumber[keep_indices], deriv_data[:, keep_indices]))

    return output_data


# ----- Classification, ROC and PR Curve Creation ----- #
def create_roc_pr_curve(
    classifier,
    cv,
    data,
    diagnoses,
    output_folder_processed_data,
    baseline_method_name,
    region_selection_name,
    filtering_method_name,
    normalize_method_name,
    derivative_name,
    centering,
    clf_name,
    final_results,
    processing_index,
):
    """
    Performs cross-validated training and evaluation of a classifier and constructing ROC
    and Precision-Recall curves. Saves plotted performance curves and appends performance 
    metrics to `final_results`.

    Args:
        classifier: Scikit-learn classifier instance -> SVM, RandomForestClassifier,
                                                        PLSRegression.
        cv: Cross-validation strategy (e.g., StratifiedKFold).
        data (np.ndarray): 2D array of spectral data.
        diagnoses (np.ndarray): 1D array of binary labels (-1, 1).
        output_folder_processed_data (str): Path to save ROC and PR curve plots.
        baseline_method_name (str): Name of the baseline correction method and parameters.
        region_selection_name (str): Name of the spectral region selected.
        filtering_method_name (str): Name of the filtering method and parameters.
        normalize_method_name (str): Name of the normalization methodand parameters.
        derivative_name (str): Name of the derivative method and parameters.
        centering (bool): Whether mean-centering was applied.
        clf_name (str): Name of the classifier and parameters.
        final_results (list): A list to which final performance dictionaries will be appended.
        processing_index (int): Identifier for the current data processing configuration.

    Returns:
        None. Appends results to `final_results` and saves performance plot image.
    """

    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs, y_real, y_probs, pr_aucs = [], [], [], [], []
    true_labels_list, predicted_labels_list = [], []
    clf_parameters = []
    train_scores, test_scores = {"accuracy": [], "roc_auc": []}, {
        "accuracy": [],
        "roc_auc": [],
    }

    # Store classifier parameters
    if isinstance(classifier, SVC):
        clf_params = {
            "kernel": classifier.get_params()["kernel"],
            "C": classifier.get_params()["C"],
            "gamma": classifier.get_params()["gamma"],
        }
    elif isinstance(classifier, RandomForestClassifier):
        clf_params = {
            "criterion": classifier.get_params()["criterion"],
            "max_depth": classifier.get_params()["max_depth"],
            "max_features": classifier.get_params()["max_features"],
            "n_estimators": classifier.get_params()["n_estimators"],
        }
    elif isinstance(classifier, PLSRegression):
        clf_params = {
            "n_comp": classifier.get_params()["n_components"],
            "scale": False,
        }
    else:
        clf_params = classifier.get_params()

    clf_parameters.append(clf_params)
    os.makedirs(output_folder_processed_data, exist_ok=True)

    # Setup figure
    fig = plt.figure(constrained_layout=False, figsize=(20, 12.5))
    spec = gridspec.GridSpec(ncols=2, nrows=2, height_ratios=[2, 0.5], figure=fig)
    ax_roc, ax_pr, ax_info = (
        fig.add_subplot(spec[0, 0]),
        fig.add_subplot(spec[0, 1]),
        fig.add_subplot(spec[1, :]),
    )

    # Cross-validation loop
    for train_index, test_index in cv.split(data, diagnoses):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = diagnoses[train_index], diagnoses[test_index]

        classifier.fit(X_train, y_train)

        # Get predicted probabilities
        if clf_name == "PLS_DA":
            train_probs = expit(classifier.predict(X_train))
            test_probs = expit(classifier.predict(X_test))
        elif clf_name == "RF":
            train_probs = classifier.predict_proba(X_train)[:, 1]
            test_probs = classifier.predict_proba(X_test)[:, 1]
        elif clf_name == "SVM":
            train_probs = classifier.predict_proba(X_train)[:, 1]
            test_probs = classifier.predict_proba(X_test)[:, 1]

        else:
            raise ValueError(f"Unknown classifier name: {clf_name}")

        # binary prediction and evaluation
        y_pred = np.where(np.round(test_probs) == 0, -1, 1)
        true_labels_list.extend(y_test)
        predicted_labels_list.extend(y_pred)

        # metrics
        train_scores["accuracy"].append(accuracy_score(y_train, np.round(train_probs)))
        train_scores["roc_auc"].append(roc_auc_score(y_train, train_probs))
        test_scores["accuracy"].append(accuracy_score(y_test, y_pred))
        test_scores["roc_auc"].append(roc_auc_score(y_test, test_probs))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(fpr, tpr))
        ax_roc.plot(fpr, tpr, color="grey", lw=1, alpha=0.3)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, test_probs)
        y_real.append(y_test)
        y_probs.append(test_probs)
        pr_aucs.append(auc(recall, precision))
        ax_pr.plot(recall, precision, lw=1, alpha=0.3, color="grey")

    # Plot averaged ROC and PR curves
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    ax_roc.plot(
        mean_fpr,
        mean_tpr,
        color="k",
        label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
        lw=4,
        alpha=0.8,
    )
    ax_roc.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        color="#666666",
        alpha=0.2,
        label="± SD",
    )
    ax_roc.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)
    ax_roc.set_title("ROC Curve")
    ax_roc.set(
        title="ROC Curve",
        xlim=[-0.02, 1.02],
        ylim=[-0.02, 1.02],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    ax_roc.legend(loc="lower right")

    # PR Curve
    y_real = np.concatenate(y_real)
    y_probs = np.concatenate(y_probs)
    precision, recall, _ = precision_recall_curve(y_real, y_probs)
    pr_auc = average_precision_score(y_real, y_probs)
    ax_pr.plot(
        recall,
        precision,
        color="k",
        label=f"Precision-Recall (AUC = {pr_auc:.3f})",
        lw=4,
        alpha=0.8,
    )
    ax_pr.set(
        title="Precision-Recall Curve",
        xlim=[-0.02, 1.02],
        ylim=[-0.02, 1.02],
        xlabel="Recall",
        ylabel="Precision",
    )
    ax_pr.legend(loc="lower right")

    # Info panel
    ax_info.axis("off")
    info_text = f"""
    Processing Index: {processing_index}
    Baseline Method: {baseline_method_name}
    Region Selection: {region_selection_name}
    Filtering Method: {filtering_method_name}
    Centering: {'True' if centering else 'False'}
    Normalization Method: {normalize_method_name}
    Derivative Method: {derivative_name}
    Classifier: {clf_name}
    """
    ax_info.text(0.05, 0.5, info_text, fontsize=12, verticalalignment="center")

    plt.tight_layout()
    plt.savefig(
        f"{output_folder_processed_data}ROC_PR_average_{processing_index}_{clf_name}.png"
    )
    plt.close()

    # Confusion matrix and other metrics
    cm = confusion_matrix(true_labels_list, predicted_labels_list)
    tp, fp, fn, tn = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = (2 * ((precision * sensitivity) / (precision + sensitivity))
        if (precision + sensitivity) > 0
        else 0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    result = {
        "Processing index": processing_index,
        "Baseline": baseline_method_name,
        "Filtering": filtering_method_name,
        "Region": region_selection_name,
        "Normalization": normalize_method_name,
        "Derivative": derivative_name,
        "Centering": "True" if centering else "False",
        "Classifier": clf_name,
        "Best Parameters": clf_parameters,
        "Train ROC mean": np.mean(train_scores["roc_auc"]),
        "Train ROC std": np.std(train_scores["roc_auc"]),
        "Test ROC mean": np.mean(test_scores["roc_auc"]),
        "Test ROC std": np.std(test_scores["roc_auc"]),
        "Average auc(fpr,tpr)": mean_auc,
        "Average precision_recall_auc": np.mean(pr_aucs),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1-score": f1,
    }

    final_results.append(result)


# ----- PLS Gridsearch Scorer ----- #
def pls_gridsearch_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Custom scoring function for PLS model grid search using ROC AUC.

    This function converts continuous PLS predictions into class labels
    using a threshold at 0 (values > 0 are mapped to 1, others to -1),
    and then calculates the ROC AUC score based on these labels.

    Args:
        y_true (np.ndarray): True class labels (1 and -1).
        y_pred (np.ndarray): Predicted continuous values from the PLS model.

    Returns:
        float: ROC AUC score between the true and predicted labels.
    """
    y_pred_labels = np.where(y_pred > 0, 1, -1)
    return roc_auc_score(y_true, y_pred_labels)


# creation of average spectra (of all samples) with selected features
def plot_average_spectra_with_selected_features(
    data,
    variable_names,
    selected_features,
    wavenumber,
    output_folder_classification,
    processing_index,
    iteration,
    clf_name,
):
    """
    Plots the average spectrum across all samples and highlights the selected features.
    The plot includes the full average spectrum line and markers indicating the selected
    features. Saves the plot as a PNG file for documentation or analysis.

    Args:
        data (np.ndarray): 2D array of spectral data (samples × features).
        variable_names (list): List of feature names corresponding to spectral variables.
        selected_features (list): List of selected feature names to highlight on the plot.
        wavenumber (array-like): Spectral axis values (e.g., in cm⁻¹) corresponding to features.
        output_folder_classification (str): Path to save the generated plot image.
        processing_index (int or str): Identifier for the current processing step or config.
        iteration (int): Current iteration index (e.g., during cross-validation or resampling).
        clf_name (str): Name of the classifier or feature selection method used.

    Returns:
        None -> Saves the plotted average spectrum with selected features as a PNG file.
    """
    avg_spectrum = data.mean(axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(wavenumber, avg_spectrum, label="Average Spectrum")
    selected_indices = [variable_names.index(feature) for feature in selected_features]
    plt.scatter(
        np.array(wavenumber)[selected_indices],
        avg_spectrum[selected_indices],
        color="black",
        label="Selected Features",
        zorder=5,
    )
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity")
    plt.title(f"Iteration {iteration + 1}: Selected Features")
    plt.legend()
    plt.savefig(
        f"{output_folder_classification}average_spectra_with_selected_features_\
            {processing_index}_{clf_name}_iteration_{iteration + 1}.png",
        bbox_inches="tight",
    )
    plt.close()


# =============================== #
#     Utility & Preprocessing     #
# =============================== #

def already_processed(processing_index, output_folder):
    file_path = os.path.join(output_folder, f"processed_data_{processing_index}.csv")
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0


def load_metadata(indexed_file_path):
    if not os.path.exists(indexed_file_path):
        raise FileNotFoundError(f"Metadata file {indexed_file_path} not found.")
    return pd.read_csv(indexed_file_path, sep="\t")


def apply_baseline_correction(data, wavenumber, method, params):
    if method:
        return (
            baseline_correction(data, wavenumber, method, **params),
            f"{method}, {params}",
        )
    return data, "None"


def apply_filtering(data, method, params):
    if method:
        filtered, *_ = filtering(data=data, method=method, params=params)
        return filtered, f"{method}, {params}"
    return data, "None"


def apply_region_selection(wavenumber, data, regions_to_delete):
    if regions_to_delete is not None:
        wavenumber, X = region_selection(wavenumber, data, regions_to_delete)
        return wavenumber, X, f"{regions_to_delete}"
    return wavenumber, data, "None"


def apply_normalization(data, method):
    if method:
        return normalize(data=data, params=method), str(method)
    return data, "None"


def apply_derivative(data, params, wavenumber):
    if params:
        wavenumber_out, deriv_out = derivative(
            data=data, params=params, wavenumber=wavenumber
        )[0]
        return deriv_out, wavenumber_out, str(params)
    return data, wavenumber, "None"


def plot_average_spectra(df, wavenumber, params_text, save_path, min_gap_length=10):
    plt.figure(figsize=(15, 10))
    avg_0 = df[df["diagnosis"] == -1].iloc[:, 2:].mean()
    std_0 = df[df["diagnosis"] == -1].iloc[:, 2:].std()
    avg_1 = df[df["diagnosis"] == 1].iloc[:, 2:].mean()
    std_1 = df[df["diagnosis"] == 1].iloc[:, 2:].std()

    regions = []
    region_start = 0
    for i in range(1, len(wavenumber)):
        if wavenumber[i] - wavenumber[i - 1] > min_gap_length:
            regions.append((region_start, i))
            region_start = i
    regions.append((region_start, len(wavenumber)))

    for start, end in regions:
        plt.fill_between(
            wavenumber[start:end],
            avg_0[start:end] - std_0[start:end],
            avg_0[start:end] + std_0[start:end],
            color="#5A5AA5",
            alpha=0.2,
            label="cirrhosis ± SD" if start == 0 else None,
        )
        plt.fill_between(
            wavenumber[start:end],
            avg_1[start:end] - std_1[start:end],
            avg_1[start:end] + std_1[start:end],
            color="#ff4c4c",
            alpha=0.2,
            label="HCC ± SD" if start == 0 else None,
        )
        plt.plot(
            wavenumber[start:end],
            avg_0[start:end],
            color="blue",
            label="cirrhosis" if start == 0 else None,
        )
        plt.plot(
            wavenumber[start:end],
            avg_1[start:end],
            color="red",
            label="HCC" if start == 0 else None,
        )

    plt.text(
        0.025,
        0.975,
        params_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Intensity")
    plt.title("Average ROA spectra")
    plt.legend(bbox_to_anchor=(1, 1), prop={"size": 12})
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ================================ #
#     Model Training & Scoring     #
# ================================ #

def select_final_params(clf_name, 
                        results, 
                        threshold_train, 
                        threshold_sum):
    """
    Filters and selects the most suitable set of classifier parameters based on 
    performance metrics and predefined thresholds. If no configuration satisfies 
    the Conditions, defaults to the one with the highest mean test score.

    Args:
        clf_name (str): Name of the classifier ('PLS_DA', 'SVM', or 'RF').
        results (pd.DataFrame): Grid search results containing training and testing 
                                scores along with parameter combinations.
        threshold_train (float): Upper limit for acceptable mean training score.
        threshold_sum (float): Upper limit for the sum of mean and standard deviation 
                               of training scores.
    
    Returns:
        dict: The best parameter combination based on filtering and sorting criteria.
    """
    
    if clf_name == "PLS_DA":
        filtered = results[
            (results["mean_train_score"] <= threshold_train)
            & (
                results["mean_train_score"] + results["std_train_score"]
                <= threshold_sum
            )
            & ~((results["mean_train_score"] == 1) & (results["std_train_score"] == 0))
            & (results["param_n_components"] <= 10)
        ]
        sorter = [
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
            "std_train_score",
            "param_n_components",
        ]
    elif clf_name == "SVM":
        filtered = results[
            (results["mean_train_score"] <= threshold_train)
            & (
                results["mean_train_score"] + results["std_train_score"]
                <= threshold_sum
            )
            & (results["param_C"] >= 0.01)
            & (results["param_C"] <= 1000)
            & (results["param_gamma"] >= 0.001)
            & (results["param_gamma"] <= 10)
        ]
        sorter = [
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
            "std_train_score",
            "param_C",
            "param_gamma",
        ]
    elif clf_name == "RF":
        filtered = results[
            (results["mean_train_score"] <= threshold_train)
            & (
                results["mean_train_score"] + results["std_train_score"]
                <= threshold_sum
            )
        ]
        sorter = [
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
            "std_train_score",
            "param_max_depth",
        ]

    sorted_results = filtered.sort_values(
        by=sorter, ascending=[False, True, True, True] + [True] * (len(sorter) - 4)
    )
    if sorted_results.empty:
        print(
            f"No valid filtered parameters for {clf_name}, \
                using parameters with the highest mean_test_score."
        )
        sorted_results = results.sort_values(by="mean_test_score", ascending=False)

    return sorted_results.iloc[0]["params"]


def train_and_evaluate(
    X,
    y,
    classification_methods,
    cv,
    output_folder,
    processing_index,
    metadata_strs,
    threshold_train,
    threshold_sum,
    variable_names,
    centering,
):
    
    """
    Trains and evaluates multiple classification models using cross-validation and 
    custom parameter filtering. Saves model performance results and plots ROC/PR curves.

    Skips evaluation if output already exists. Uses a custom selection strategy to choose 
    the best parameters based on training score thresholds and variance. Final results 
    are saved in both separate and cumulative result files.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Binary target labels (e.g., -1 and 1).
        classification_methods (list): List of tuples (name, classifier, param_grid),
                                       e.g., [("SVM", SVC(), svm_grid)].
        cv: Cross-validation strategy (e.g., StratifiedKFold).
        output_folder (str): Path to the folder where output files will be saved.
        processing_index (int or str): Identifier for the current preprocessing run.
        metadata_strs (dict): Dictionary with metadata method names (e.g., normalization,
                              filtering, etc.) used to label saved results.
        threshold_train (float): Maximum allowed mean training score for filtering.
        threshold_sum (float): Maximum allowed sum of mean and std of training score.
        variable_names (list): List of feature names.
        centering (bool): Whether mean centering was applied to the data.

    Returns:
        None. Saves ROC/PR plots and CSV result files with evaluation metrics.
    """
    final_results = []

    for clf_name, clf, grid in classification_methods:

        roc_pr_filename = os.path.join(
            output_folder, f"ROC_PR_average_{processing_index}_{clf_name}.png"
        )
        if os.path.exists(roc_pr_filename):
            print(
                f"Skipping {clf_name} for processing index {processing_index} \
                    as it was already performed (ROC/PR curve already exists)"
            )
            continue

        filename = f"{output_folder}results_{processing_index}_{clf_name}.csv"
        scorer = (
            make_scorer(pls_gridsearch_scorer, needs_proba=False, needs_threshold=False)
            if clf_name == "PLS_DA"
            else "roc_auc"
        )
        grid_search = GridSearchCV(
            clf, grid, cv=cv, scoring=scorer, return_train_score=True, verbose=3
        )
        grid_search.fit(X, y)

        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results.to_csv(filename, sep="\t", index=False)

        best_params = select_final_params(
            clf_name, cv_results, threshold_train, threshold_sum
        )
        if isinstance(best_params, str):
            best_params = ast.literal_eval(best_params)

        if clf_name == "PLS_DA":
            final_clf = PLSRegression(n_components=best_params["n_components"])
        elif clf_name == "RF":
            final_clf = RandomForestClassifier(
                **best_params, random_state=0, class_weight="balanced"
            )
        elif clf_name == "SVM":
            final_clf = SVC(
                **best_params, probability=True, random_state=0, class_weight="balanced"
            )

        create_roc_pr_curve(
            classifier=final_clf,
            cv=cv,
            data=X,
            diagnoses=y,
            output_folder_processed_data=output_folder,
            centering=centering,
            clf_name=clf_name,
            processing_index=processing_index,
            final_results=final_results,
            **metadata_strs,
        )

        results_df = pd.DataFrame(final_results)
        results_df.to_csv(
            f"{output_folder}separate_result_{processing_index}_{clf_name}.csv",
            sep="\t",
            index=False,
        )
        cumulative_path = f"{output_folder}{clf_name}_mean_center_threshs_098_1.csv"
        results_df.to_csv(
            cumulative_path,
            mode="a" if os.path.exists(cumulative_path) else "w",
            sep="\t",
            index=False,
            header=not os.path.exists(cumulative_path),
        )


# ===================== #
#     Main Function     #
# ===================== #

def process_data(
    output_folder,
    wavenumber,
    data,
    sample_id,
    diagnoses,
    baseline_method,
    baseline_params,
    region_selection_params,
    filtering_method,
    filtering_params,
    normalize_method,
    derivative_params,
    classification_methods,
    cv,
    random_state,
    processing_index,
    centering=True,
    threshold_train=1,
    threshold_sum=1,
):

    """
    Executes a full data processing and classification pipeline for a specific configuration.

    This includes baseline correction, filtering, region selection, normalization, and 
    derivative calculation. After pre-processing, the data is optionally mean-centered, 
    saved, visualized, and passed to the classification pipeline with cross-validation. 
    Results and plots are saved in the specified output folder.

    Args:
        output_folder (str): Directory where all outputs (processed data, plots, results) are saved.
        wavenumber (array-like): Original spectral axis values.
        data (np.ndarray): Raw spectral data.
        sample_id (pd.Series): Series of sample identifiers.
        diagnoses (pd.Series or np.ndarray): Binary class labels (-1 and 1).
        baseline_method (str): Method name for baseline correction.
        baseline_params (dict): Parameters for the baseline correction method.
        region_selection_params (dict): Parameters for selecting specific spectral regions.
        filtering_method (str): Method name for signal filtering.
        filtering_params (dict): Parameters for the filtering method.
        normalize_method (str): Method name for normalization.
        derivative_params (dict): Parameters for derivative transformation.
        classification_methods (list): List of tuples (name, classifier, param_grid).
        cv: Cross-validation strategy (e.g., StratifiedKFold instance).
        random_state (int): Seed for reproducibility (not directly used here but assumed elsewhere).
        processing_index (int or str): Unique identifier for the processing configuration.
        centering (bool): Whether mean-centering is applied to the processed data.
        threshold_train (float): Maximum allowed mean training score for model selection.
        threshold_sum (float): Maximum allowed sum of mean and std of training score.

    Returns:
        None. Saves processed data, spectrum plots, and model evaluation results to disk.
    """

    processing_index = int(processing_index)
    metadata_df = load_metadata(
        os.path.join(output_folder, "indexed_processing_combinations.csv")
    )
    metadata_row = metadata_df[metadata_df["processing_index"] == processing_index]

    processing_index = int(processing_index)
    metadata_df = load_metadata(
        os.path.join(output_folder, "indexed_processing_combinations.csv")
    )
    metadata_row = metadata_df[
        metadata_df["processing_index"] == processing_index
    ].iloc[0]

    baseline_str = metadata_row["baseline"]
    filtering_str = metadata_row["filtration"]
    region_str = metadata_row["region"]
    normalize_str = metadata_row["normalization"]
    deriv_str = metadata_row["derivative"] 
        
    if already_processed(processing_index, output_folder):
        print(f"Skipping already processed index {processing_index}")
        df = pd.read_csv(
            os.path.join(output_folder, f"processed_data_{processing_index}.csv"),
            sep="\t",
        )
        sample_id = df["id"].reset_index(drop=True)
        diagnoses = df["diagnosis"].astype(float).reset_index(drop=True).to_numpy()
        X = df.drop(columns=["id", "diagnosis"]).reset_index(drop=True).to_numpy()
    
    else:
        print(f"Processing index {processing_index}...")

        X, baseline_str = apply_baseline_correction(
            data, wavenumber, baseline_method, baseline_params
        )
        X, filtering_str = apply_filtering(X, filtering_method, filtering_params)
        wavenumber, X, region_str = apply_region_selection(
            wavenumber, X, region_selection_params
        )
        X, normalize_str = apply_normalization(X, normalize_method)
        X, wavenumber, deriv_str = apply_derivative(X, derivative_params, wavenumber)

        df = pd.DataFrame(X, columns=wavenumber)
        df.insert(0, "id", sample_id.reset_index(drop=True))
        df.insert(1, "diagnosis", diagnoses)
        
        df.to_csv(
            os.path.join(output_folder, f"processed_data_{processing_index}.csv"),
            sep="\t",
            index=False,
        )

        params_text = f"Baseline: {baseline_str}\nFiltering: {filtering_str}\
            \nRegion: {region_str}\nNormalization: {normalize_str}\nDerivative: {deriv_str}"
        plot_average_spectra(
            df,
            wavenumber,
            params_text,
            os.path.join(output_folder, f"average_spectra_{processing_index}.png"),
        )

    if centering:
        X = X - np.mean(X, axis=0)

    y = np.asarray(diagnoses)
    variable_names = wavenumber if isinstance(wavenumber, list) else wavenumber.tolist()

    train_and_evaluate(
        X,
        y,
        classification_methods,
        cv,
        output_folder,
        processing_index,
        {
            "baseline_method_name": baseline_str,
            "region_selection_name": region_selection_params,
            "filtering_method_name": filtering_str,
            "normalize_method_name": normalize_str,
            "derivative_name": deriv_str,
        },
        threshold_train,
        threshold_sum,
        variable_names,
        centering,
    )

# ============================================ #
# Sample names for diagnosis assignment
# only related to the dataset used in the paper https://doi.org/10.1016/j.saa.2025.126261
samples_hcc = ["SM35",	"SM36",	"SM37",	"SM38",	"SM39",	"SM40",	"SM41",	
               "SM42",	"SM43",	"SM44", "SM45",	"SM46",	"SM50",	"SM59",	
               "SM69",	"SM72",	"SM86",	"SM87",	"H005",	"H012", "H023",
               "H024",	"H029",	"H033",	"H034",	"H037",	"H038",	"H040",	
               "H042",	"H043", "H046",	"H047",	"H049",	"H052",	"H056",	
               "H057",	"H058",	"H059",	"H061",	"H062",	"H063",	"H064",	
               "H066",	"H067",	"H068",	"H069",	"H072",	"H077",	"H083",	
               "H086",  "H088",	"H090",	"H091",	"H094",	"H096",	"H097",
               "H098",	"H099",	"H100",	"H101",	"H102",	"H103",	"H104",	
               "H105",	"H106",	"H107",	"H108",	"H110",	"H111",	"H118",
               "H124",	"H127"]
samples_cir = ["SM47",	"SM48",	"SM51",	"SM52",	"SM53",	"SM54",	"SM55",
               "SM56",	"SM57",	"SM58",	"SM60",	"SM63",	"SM64",	"SM65",
               "SM66",	"SM67",	"SM68",	"SM70",	"SM71",	"SM73",	"SM75",	
               "SM76",	"SM77",	"SM78",	"SM80",	"SM81",	"SM82",	"SM85",	
               "SM88",	"SM89","SM90",	"SM91",	"SM92",	"H002",	"H004",	
               "H006",	"H007",	"H008",	"H009",	"H010",	"H011",	"H013",	
               "H015",	"H016",	"H017",	"H018",	"H019",	"H020",	"H021",	
               "H022",  "H026",	"H027",	"H028",	"H030",	"H031",	"H032",	
               "H035",	"H036",	"H041",	"H045",	"H048",	"H050",	"H051",	
               "H053",	"H054",	"H060",	"H065",	"H070",	"H071",	"H073",	
               "H074",	"H075",	"H076",	"H078",	"H079",	"H080",	"H081",	
               "H082",	"H084",	"H085",	"H089",	"H092",	"H093",	"H095",	
               "H109",	"H112",	"H114",	"H115",	"H116",	"H117", "H119",	
               "H120",	"H121",	"H122",	"H123",	"H125",	"H126",	"H128",	
               "H129",	"H130",	"H131",	"H132",	"H133"]


# Diagnoses assignment
def assign_diagnosis(sample_id):
    sample_id_str = str(sample_id)
    for sample, dg in zip([samples_hcc, samples_cir], [1, -1]):
        if any(id in sample_id_str for id in sample):
            return dg
    "not_found"

from pathlib import Path
from typing import Any

import dask
import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb
from dask.diagnostics import ProgressBar


def get_stats(array, name: str) -> dict[str, Any]:
    return {f"{name}_mean": np.nanmean(array), f"{name}_med": np.nanmedian(array), f"{name}_std": np.nanstd(array)}


def feature_extraction(signal, sampling_freq: int) -> dict[str, Any]:
    """Feature extraction from raw ECG signal."""
    # Check if signal is inverted (and correct it).
    ecg_signal, _ = nk.ecg_invert(signal, sampling_rate=sampling_freq)

    # Filter ECG signal.
    cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=sampling_freq, method="neurokit")

    # Standardization.
    cleaned_ecg = (cleaned_ecg - np.mean(cleaned_ecg)) / np.std(cleaned_ecg)

    # Find R peaks.
    _, rpeaks_info = nk.ecg_peaks(cleaned_ecg, sampling_rate=sampling_freq)
    rpeaks = rpeaks_info["ECG_R_Peaks"]

    # Find P, Q, S and T peaks.
    _, waves_peak = nk.ecg_delineate(cleaned_ecg, rpeaks, sampling_rate=sampling_freq, method="prominence")

    # ==================================================
    # Morphological features...
    # ==================================================
    # ... for R peaks.
    features = get_stats(cleaned_ecg[rpeaks], "R")

    # ... for P peaks.
    p_peaks = np.array(waves_peak["ECG_P_Peaks"])
    wave_indices = p_peaks[~np.isnan(p_peaks)].astype(int)
    features.update(get_stats(cleaned_ecg[wave_indices], "P"))

    # ... for Q peaks.
    q_peaks = np.array(waves_peak["ECG_Q_Peaks"])
    wave_indices = q_peaks[~np.isnan(q_peaks)].astype(int)
    features.update(get_stats(cleaned_ecg[wave_indices], "Q"))

    # ... for S peaks.
    s_peaks = np.array(waves_peak["ECG_S_Peaks"])
    wave_indices = s_peaks[~np.isnan(s_peaks)].astype(int)
    features.update(get_stats(cleaned_ecg[wave_indices], "S"))

    # ... for T peaks.
    t_peaks = np.array(waves_peak["ECG_T_Peaks"])
    wave_indices = t_peaks[~np.isnan(t_peaks)].astype(int)
    features.update(get_stats(cleaned_ecg[wave_indices], "T"))

    # ==================================================
    # Time interval features
    # ==================================================
    # R-R interval.
    features.update(get_stats(np.diff(rpeaks), "RR"))

    # QRS interval.
    features.update(get_stats(q_peaks - s_peaks, "QRS"))

    # S-T interval.
    features.update(get_stats(t_peaks - s_peaks, "ST"))

    # P-Q interval.
    features.update(get_stats(p_peaks - q_peaks, "PQ"))

    # ==================================================
    # HRV features
    # ==================================================
    hrv_time = (
        nk.hrv_time(rpeaks, sampling_rate=sampling_freq)
        .drop(columns=["HRV_SDANN1", "HRV_SDNNI1", "HRV_SDANN2", "HRV_SDNNI2", "HRV_SDANN5", "HRV_SDNNI5"])
        .to_dict("records")[0]
    )
    features.update(hrv_time)

    hrv_freq = (
        nk.hrv_frequency(rpeaks, sampling_rate=sampling_freq)
        .drop(columns=["HRV_ULF", "HRV_VLF", "HRV_LF", "HRV_LFHF", "HRV_LFn"])
        .to_dict("records")[0]
    )
    features.update(hrv_freq)

    return features


@dask.delayed
def process_single_file(obs_path: Path) -> dict[str, Any]:
    """Loads a signal, extracts features, and returns a dictionary."""
    signal, meta = wfdb.rdsamp(obs_path)
    features = {"record_name": obs_path.stem}
    features.update(feature_extraction(signal.flatten(), meta["fs"]))
    return features


def get_dataset(path: str | Path, to_csv_params: dict[str, Any]):
    path = Path(path)
    # Load labels.
    y = None
    labels_path = path / "REFERENCE.csv"
    if labels_path.exists():
        y = pd.read_csv(labels_path, header=None, names=["record_name", "class_label"], index_col="record_name")

        if y.empty:
            raise ValueError(f"Label file '{labels_path}' is empty.")
        if y["class_label"].isna().any():
            raise ValueError(f"Label file '{labels_path}' contains missing (NaN) values.")

    # Load features.
    # Check if the features were already computed.
    features_path = path / "features.csv"
    if features_path.exists():
        print("Loading features from cache.")
        X = pd.read_csv(features_path, index_col="record_name", **to_csv_params)
    else:
        print("No feature cache found. Starting feature extraction...")
        observations = sorted([obs.with_suffix("") for obs in path.iterdir() if obs.suffix == ".mat"])
        with ProgressBar():
            X = dask.compute(*[process_single_file(obs) for obs in observations])
        X = pd.DataFrame(X).set_index("record_name")

        # Save features.
        X.to_csv(features_path, **to_csv_params)

    if X.empty:
        raise ValueError("Feature set (X) is empty.")

    if y is not None:
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Mismatch between number of features and labels. X has {X.shape[0]} rows, but y has {y.shape[0]} rows."
            )
        y = y.reindex(X.index)

        return X, y["class_label"]

    return X

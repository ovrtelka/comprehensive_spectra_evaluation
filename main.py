import numpy as np
import pandas as pd
import os
import time
import traceback
from itertools import product
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from utils import (
    assign_diagnosis,
    process_data,
)
from scipy.interpolate import CubicSpline


# ====================== #
#      Configuration     #
# ====================== #


INPUT_PATH = "path/to/data/"
OUTPUT_FOLDER = "path/to/output/folder/"
ERROR_LOG_PATH = os.path.join(OUTPUT_FOLDER, "error_log_processing_loop.txt")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

RANDOM_STATE = 100
CV_STRATEGY = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10, 
    random_state=RANDOM_STATE
)

WAVENUM_MIN = 580
WAVENUM_MAX = 1850
INTERPOLATION_STEP = 1


# ============================== #
#      Load and Prepare Data     #
# ============================== #

df = pd.read_csv(INPUT_PATH, delimiter="\t", decimal=".", header=None)
df.insert(1, "dg", df.iloc[:, 0].apply(assign_diagnosis))

X_raw = -df.iloc[1:, 2:].astype(float).values
y = df.iloc[1:, 1].astype(float).values
sample_id = df.iloc[1:, 0:1]
wavenumber_raw = df.iloc[0, 2:].astype(float).values

x_new = np.linspace(
    WAVENUM_MIN, WAVENUM_MAX, int((WAVENUM_MAX - WAVENUM_MIN) / INTERPOLATION_STEP + 1)
)
y_news = [CubicSpline(wavenumber_raw, row, bc_type="natural")(x_new) for row in X_raw]
X = np.array(y_news)
wavenumber = x_new


# ========================== #
#      Define Parameters     #
# ========================== #

carotenoid_removal = [(0, 809), (1000, 1030), (1140, 1175), (1490, 1540), (1716, 2000)]
edge_removal = [(0, 809), (1716, 2000)]
full_range = [(0, 0)]

all_region_selections = [
    full_range,
    edge_removal,
    carotenoid_removal,
]


baseline_params = [
    ("drpls", {"lam": [1e7,1e8,],
               "eta": [0.1, 0.5, 0.9],
               "diff_order": [2]},
    ),
]

filtration_params = [
    ( "fir", {"n_taps": [51, 91],
            "cutoff": [0.06,]},
    ),
]
normalize_methods = ["uvn", "msc"]

derivative_params = [({"window_size": [9],
                       "order": [2],
                       "deriv": [1],}
    ),
]

CLASSIFICATION_METHODS = [
    ("SVM", SVC(random_state=RANDOM_STATE, class_weight="balanced"),
        {"C": [0.01, 0.01, 0.1, 1, 10],
        "gamma": [0.01, 0.01, 0.1, 1, 10],
        "kernel": ["rbf", "poly"],
        },
    ),
    ("RF", RandomForestClassifier( random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
        {"n_estimators": [250],
         "max_depth": [3, 6],
         "max_features": ["sqrt", "log2"],
         "min_samples_split": [10],
         "min_samples_leaf": [5],
        },
    ),
    ("PLS_DA", PLSRegression(scale=False), {"n_components": range(1, 11)}),
]


# ==================================== #
#      Pre-processing Combinations     #
# ==================================== #

all_baseline_combinations = [
    (model, dict(zip(params.keys(), vals)))
    for model, params in baseline_params
    for vals in product(*params.values())
]
all_filtration_combinations = [
    (method, [dict(zip(params.keys(), vals))])
    for method, params in filtration_params
    for vals in product(*params.values())
]
all_normalize_combinations = [[method] for method in normalize_methods]
all_derivative_combinations = [
    [dict(zip(params.keys(), vals))]
    for params in derivative_params
    for vals in product(*params.values())
]

all_methods_combinations = list(
    product(
        [None] + all_baseline_combinations,
        all_region_selections,
        [None] + all_filtration_combinations,
        [None] + all_normalize_combinations,
        [None] + all_derivative_combinations,
    )
)

print(
    f" NUMBER OF POSSIBLE PRE-ROCESSING COMBINATIONS: {len(all_methods_combinations)}"
)
zero_padding_length = len(str(len(all_methods_combinations))) + 1
indexed_combinations = [
    {"processing_index": f"{idx:0{zero_padding_length}d}", "combination": combination}
    for idx, combination in enumerate(all_methods_combinations, start=1)
]

comb_df = pd.DataFrame(indexed_combinations)
comb_df_details = pd.DataFrame(
    comb_df["combination"].tolist(),
    columns=["baseline", "region", "filtration", "normalization", "derivative"],
)
df_combined = pd.concat([comb_df["processing_index"], comb_df_details], axis=1)
df_combined.to_csv(
    os.path.join(OUTPUT_FOLDER, "indexed_processing_combinations.csv"),
    sep="\t",
    index=False,
)

# sleep to assure the file is written before starting multiprocessing
time.sleep(5)

def already_processed(index, clf_name):
    path = os.path.join(OUTPUT_FOLDER, f"ROC_PR_average_{index}_{clf_name}.png")
    return os.path.exists(path)


# ================================================================================= #

#"""
# ===============================
#        SINGLE PROCESSING                 
# ===============================   

start_time = time.time()
total = len(all_methods_combinations) * len(CLASSIFICATION_METHODS)

for idx, (baseline_method, region_params, filt_method, norm_method, deriv_params) \
    in enumerate(all_methods_combinations):
    index_str = f'{idx + 1:0{zero_padding_length}d}'

    if already_processed(index_str, OUTPUT_FOLDER):
        print(f'Skipping combination {index_str} (already processed).')
        continue

    for clf_name, clf_model, clf_params in CLASSIFICATION_METHODS:
        try:
            process_data(
                output_folder=OUTPUT_FOLDER,
                processing_index=index_str,
                wavenumber=wavenumber,
                data=X/100000,
                sample_id=sample_id,
                diagnoses=y,
                baseline_method=baseline_method[0] if baseline_method else None,
                baseline_params=baseline_method[1] if baseline_method else None,
                region_selection_params=region_params,
                filtering_method=filt_method[0] if filt_method else None,
                filtering_params=filt_method[1] if filt_method else {},
                normalize_method=norm_method,
                derivative_params=deriv_params,
                classification_methods=[(clf_name, clf_model, clf_params)],
                cv=CV_STRATEGY,
                random_state=RANDOM_STATE,
                centering=True
            )

        except Exception as e:
            err_msg = (
                f'Error in combination {index_str}: {e}\n'
                f'Traceback:\n{traceback.format_exc()}\n'
            )
            print(err_msg)
            with open(ERROR_LOG_PATH, 'a') as f:
                f.write(err_msg)

elapsed = time.time() - start_time
print(f'\nFinished in {elapsed:.2f} seconds')
print(f'Processed results saved in: {os.path.abspath(OUTPUT_FOLDER)}')

"""


#"""
# ========================== #
#      MULTI PROCESSING      #
# ========================== #

from joblib import Parallel, delayed

def process_data_wrapper(args):
    return process_data(**args)

def main(num_cores=None):
    jobs = []
    total_combinations = len(all_methods_combinations) * len(CLASSIFICATION_METHODS)

    for idx, (baseline, region, filtering, normalization, derivative) in enumerate(
        all_methods_combinations
    ):
        index_str = f"{idx + 1:0{zero_padding_length}d}"

        for clf_name, clf_model, clf_params in CLASSIFICATION_METHODS:
            if already_processed(index_str, clf_name):
                print(f"Skipping {index_str} | {clf_name} (already processed)")
                continue

            current_combination = (
                idx * len(CLASSIFICATION_METHODS)
                + CLASSIFICATION_METHODS.index((clf_name, clf_model, clf_params))
                + 1
            )
            percentage_complete = (current_combination / total_combinations) * 100
            print(
                f"Processing combination {current_combination} out of {total_combinations} \
                    ({percentage_complete:.3f}% complete)"
            )

            args = {
                "output_folder": OUTPUT_FOLDER,
                "processing_index": index_str,
                "wavenumber": wavenumber,
                "data": X / 1000000,
                "sample_id": sample_id,
                "diagnoses": y,
                "baseline_method": baseline[0] if baseline else None,
                "baseline_params": baseline[1] if baseline else None,
                "region_selection_params": region,
                "filtering_method": filtering[0] if filtering else None,
                "filtering_params": filtering[1] if filtering else {},
                "normalize_method": normalization if normalization else None,
                "derivative_params": derivative if derivative else None,
                "classification_methods": [(clf_name, clf_model, clf_params)],
                "cv": CV_STRATEGY,
                "random_state": RANDOM_STATE,
                "threshold_train": 0.98,
                "threshold_sum": 1,
                "centering": True,
            }

            jobs.append(args)

    # Run parallel jobs
    num_cores = num_cores or os.cpu_count()
    print(f"Running on {num_cores} cores")
    start_time = time.time()

    results = Parallel(n_jobs=num_cores, backend="loky")(
        delayed(process_data_wrapper)(job) for job in jobs
    )

    duration = time.time() - start_time
    print(f"Finished in {duration:.2f} seconds")


if __name__ == "__main__":
    main(num_cores=4)

#""""

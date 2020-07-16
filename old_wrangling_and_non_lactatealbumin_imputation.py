import os, copy, re, sys, operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import dmatrices
from scipy import stats
from statsmodels.imputation import mice
from statsmodels.discrete import discrete_model as dm
from typing import Tuple, Dict, List
from progressbar import progressbar as pb
from sklearn.preprocessing import RobustScaler, OneHotEncoder, QuantileTransformer
from sklearn.linear_model import LogisticRegression

sys.path.append("")
from nelarisk.constants import RANDOM_SEED
from nelarisk.helpers import split_into_folds, save_object, winsorize
from nelarisk.explore import con_data, cat_data, missingness_perc
from nelarisk.impute import CategoricalImputer

get_ipython().run_line_magic("matplotlib", "inline")


df = pd.read_pickle(
    os.path.join("data", "df_after_univariate_wrangling_new_indications.pkl")
)


# ## Preprocessing


cont_vars = [
    "S01AgeOnArrival",
    "S03SerumCreatinine",
    "S03PreOpArterialBloodLactate",
    "S03PreOpLowestAlbumin",
    "S03Sodium",
    "S03Potassium",
    "S03Urea",
    "S03WhiteCellCount",
    "S03Pulse",
    "S03SystolicBloodPressure",
    "S03GlasgowComaScore",
]

cat_vars = [
    "S03ASAScore",
    "S03CardiacSigns",
    "S03RespiratorySigns",
    "S03WhatIsTheOperativeSeverity",
    "S03DiagnosedMalignancy",
    "S03Pred_Peritsoil",
    "S03NCEPODUrgency",
    "S02PreOpCTPerformed",
    "S03ECG",
]

indications = [c for c in df.columns if "S05Ind_" in c]
# indications


def select_vars(
    df: pd.DataFrame,
    cont_vars: List[str] = cont_vars,
    cat_vars: List[str] = cat_vars,
    indications: List[str] = indications,
    target: str = "Target",
) -> pd.DataFrame:
    """Select only the variables of interest, dropping the rest."""
    return df[cont_vars + cat_vars + indications + [target]]


# The categories for `S03ECG` are:
#
# - 1 = no abnormalities
# - 4 = AF, rate 60-90
# - 8 = AF (rate > 90) or any other abnormal rhythm including pacing
#
# We will combine 4 and 8 together to allow easier user input to the model (a single checkbox) to signify arrhythmia. We also capture rate information in the heart rate variable, so we don't lose info by combining these categories.


def combine_categories(
    df: pd.DataFrame, combine: Dict[str, Dict[float, float]]
) -> pd.DataFrame:
    """Combines values of categorical variables. Propogates missing
        values. Example key-value pair in combine is
        'S03ECG' : {1.0: 0.0, 4.0: 1.0, 8.0: 1.0} which combines
        the two 'abnormal ecg' categories together together."""
    drop = []

    for v, mapping in combine.items():
        temp = f"{v}_temp"
        df[temp] = df[v].copy()
        df[v] = np.nan

        for old, new in mapping.items():
            df.loc[df[temp] == old, v] = new

        drop.append(temp)

    return df.drop(drop, axis=1)


# ### Meaningful missingness
#
# The missingness is low or none in all considered variables except lactate and albumin. **Decision to treat all variables except lactate and albumin as MAR**.
#
# Lactate is more likely to be missing in patients not considered sick/unstable by the treating team. Especially in this latter group, lactate not being measured may be associated with lower mortality, and the unmeasured lactates may have been lower than the emergency lap. population average. **Decision to make a binary 'lactate measured' feature for use in the mortality risk model**. This won't help us predict better values for the unmeasured lactates, but it will allow the mortality model to adjust. Note that this adjustment may be less successful if there are various reasons for unmeasured lactates - e.g. no lactate measurement is more likely in patients recently admitted to hospital, lacate measurement more likely in recent patients as gas machines more widely available.
#
# We will treat albumin similarly to lactate. Unmeasured LFTs are probably more likely to be normal.


def add_missingness_indicators(
    df: pd.DataFrame,
    cols: list = ["S03PreOpArterialBloodLactate", "S03PreOpLowestAlbumin"],
) -> pd.DataFrame:
    """Adds a missingness indicator column for each of the
        specified variables (cols)."""
    for c in cols:
        c_missing = f"{c}_missing"
        df[c_missing] = np.zeros(df.shape[0])
        df.loc[df[c].isnull(), c_missing] = 1.0
    return df


# ### Label encode categorical variables


def make_single_indications_variable(
    df: pd.DataFrame, indications: Tuple[str] = indications
) -> pd.DataFrame:
    """Changes encoding of indications from one-hot to a single
        variable with multiple integer labels."""
    df["Indication"] = df[indications].idxmax(axis=1)
    return df.drop(indications, axis=1)


multi_cat_levels = {
    "S03ASAScore": (1.0, 2.0, 3.0, 4.0, 5.0),
    "S03CardiacSigns": (1.0, 2.0, 4.0, 8.0),
    "S03RespiratorySigns": (1.0, 2.0, 4.0, 8.0),
    "S03DiagnosedMalignancy": (1.0, 2.0, 4.0, 8.0),
    "S03Pred_Peritsoil": (1.0, 2.0, 4.0, 8.0),
    "S03NCEPODUrgency": (1.0, 2.0, 3.0, 8.0),
    "Indication": indications,
}

binary_vars = list(set(cat_vars) - set(multi_cat_levels.keys()))


def label_encode(df: pd.DataFrame, multi_cat_levels=multi_cat_levels) -> pd.DataFrame:
    """Encode labels for each categorical variable as integers, with
        missingness support."""
    for c, levels in multi_cat_levels.items():
        if c is not "Indication":
            df[c] = df[c].astype(float)
            df[c] = [np.nan if np.isnan(x) else levels.index(x) for x in df[c].values]
            df[c] = df[c].astype(float)
        else:
            df[c] = [
                np.nan if x == "S05Ind_Missing" else levels.index(x)
                for x in df[c].values
            ]
    return df


# ### Run preprocessing


def preprocess_data(
    data: pd.DataFrame,
) -> (
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    Dict[str, Tuple[float, float]],
):
    """Split NELA data in train and test folds, and preprocess each."""
    df = data.copy()

    df = select_vars(df)
    df = combine_categories(df, combine={"S03ECG": {1.0: 0.0, 4.0: 1.0, 8.0: 1.0}})
    df = add_missingness_indicators(df)
    df = make_single_indications_variable(df)
    df = label_encode(df)

    X_train_df, y_train, X_test_df, y_test = split_into_folds(df)

    X_train_df, winsor_thresholds = winsorize(
        X_train_df,
        cont_vars=cont_vars,
        include={
            "S01AgeOnArrival": [False, True],
            "S03GlasgowComaScore": [False, False],
        },
    )
    X_test_df, _ = winsorize(X_test_df, winsor_thresholds)

    return (X_train_df, y_train, X_test_df, y_test, winsor_thresholds)


(X_train_df, y_train, X_test_df, y_test, winsor_thresholds) = preprocess_data(df)


# ## Calculate number of imputations needed
#
# For this calculation, we include variables which we will be imputing later (lactate and albumin).


def calculate_mice_imputations(df: pd.DataFrame) -> (int, float):
    """White et al recommend using 100 * f MICE imputations,
        where f is the fraction of incomplete cases in the
        DataFrame."""
    f = 1 - (df.dropna(how="any").shape[0] / df.shape[0])
    n_imputations = int(np.ceil(f * 100))
    print(
        f"{np.round(f * 100, 3)}% incomplete cases",
        f"so running {n_imputations} MICE imputations",
    )
    return n_imputations, f


n_mice_imputations = {}

for fold, Xdf in (("train", X_train_df), ("test", X_test_df)):
    print(fold)
    n_mice_imputations[fold], _ = calculate_mice_imputations(Xdf)
    print("\n")


# ## Imputation excluding lactate and albumin


def drop_add_lactate_albumin_cols(
    df: pd.DataFrame,
    lac_alb_df: pd.DataFrame = None,
    drop: bool = True,
    lac_alb_cols: List[str] = [
        "S03PreOpArterialBloodLactate",
        "S03PreOpLowestAlbumin",
        "S03PreOpArterialBloodLactate_missing",
        "S03PreOpLowestAlbumin_missing",
    ],
) -> pd.DataFrame:
    """If drop is True, simply drops lac_alb_cols from df. If
        drop is False, adds lac_alb_cols from lac_alb_df to df.
        
        We need to temporarily remove variables related to lactate
        and albumin, as we will be imputing these later using GAMs
        and we don't want to use the missingness indicators to
        inform MICE on the other variables."""
    if drop:
        return df.drop(lac_alb_cols, axis=1)
    else:
        return pd.concat([df, lac_alb_df[lac_alb_cols]], axis=1)


def drop_multi_cat_cols(
    df: pd.DataFrame, multi_cat_cols: List[str] = list(multi_cat_levels.keys())
) -> pd.DataFrame:
    """Drops non-binary categorical variables prior to
        statsmodels MICE. These will be added back later by
        CategoricalImputer."""
    return df.drop(multi_cat_cols, axis=1)


def run_mice(
    df: pd.DataFrame,
    n_mice_imps: int,
    binary_vars: List[str] = binary_vars,
    n_burn_in: int = 10,
    n_skip: int = 3,
) -> (List[pd.DataFrame], mice.MICEData):
    """Imputes missing continuous and binary variables using MICE
        with linear regression / binary logistic regression. Uses
        the default burn-in and skip settings from statsmodels."""
    imp = mice.MICEData(df)

    for v in binary_vars:
        imp.set_imputer(v, model_class=dm.Logit)

    mice_imps = []

    # Initial MICE 'burn in' imputations which we discard
    for _ in pb(range(n_burn_in), prefix="MICE burn in:"):
        imp.update_all()

    for i in pb(range(n_mice_imps), prefix="MICE:"):
        if i:
            # skip some MICE imputations between the ones we keep
            imp.update_all(n_skip + 1)
        mice_imps.append(imp.data.copy())

    return mice_imps, imp


mice_cont_vars = copy.deepcopy(cont_vars)
mice_cont_vars.remove("S03PreOpArterialBloodLactate")
mice_cont_vars.remove("S03PreOpLowestAlbumin")


def impute(
    data: pd.DataFrame,
    n_mice_imps: int = None,
    mice_cont_vars: List[str] = mice_cont_vars,
    binary_vars: List[str] = binary_vars,
    multi_cat_vars: List[str] = list(multi_cat_levels.keys()),
) -> (mice.MICEData, CategoricalImputer):
    """Imputes missing continuous and binary variables using
        linear regression / binary logistic regression implemented
        in statsmodels MICE. Then, imputes each non-binary
        categorical variable in turn using sklearn.
        
        We do the imputation in two stages as principled imputation
        of non-binary categorical variables is tricky using
        statsmodels MICE. See issue 3858 on their github.
        
        Note that, as a consequence of this two-stage imputation,
        all missing values are imputed without reference to the
        non-binary categorical variables."""
    df = data.copy()
    df_no_lac_alb = drop_add_lactate_albumin_cols(df, drop=True)
    mice_df = df_no_lac_alb.copy()

    if n_mice_imps is None:
        n_mice_imps, _ = calculate_mice_imputations(mice_df)

    mice_df = drop_multi_cat_cols(mice_df)
    mice_imps, imp = run_mice(mice_df, n_mice_imps)

    cat_imp = CategoricalImputer(
        df_no_lac_alb,
        mice_imps,
        mice_cont_vars,
        binary_vars,
        multi_cat_vars,
        RANDOM_SEED,
    )
    cat_imp.impute_all()

    return imp, cat_imp


# ### Run imputation and put output in dict

# TODO: Consider using the all data from the test fold trusts for imputation, then restricting to just the test fold cases thereafter.


output = {}

for fold_name, X_df, y in (("train", X_train_df, y_train), ("test", X_test_df, y_test)):
    print(fold_name.upper())
    imp_results = impute(X_df, n_mice_imputations[fold_name])
    output[fold_name] = {
        "X_df": X_df,
        "y": y,
        "imp": {"MICEData": imp_results[0], "CategoricalImputer": imp_results[1]},
    }


# ### Add back in albumin- and lactate-related columns


def add_lactate_albumin_cols_multidf(
    dfs: List[pd.DataFrame], lac_alb_df: pd.DataFrame
) -> List[pd.DataFrame]:
    """Adds back lactate- and albumin-related columns to a
        list of DataFrames."""
    dfs = copy.deepcopy(dfs)
    for i in range(len(dfs)):
        dfs[i] = drop_add_lactate_albumin_cols(dfs[i], lac_alb_df, drop=False)
    return dfs


"""We can't pickle some of the imputation objects, so we
    just extract the imputed dataframes to save for use
    elsewhere."""
pkl_output = {}

for fold_name in ("train", "test"):
    pkl_output[fold_name] = copy.deepcopy(output[fold_name])
    pkl_output[fold_name]["imp"] = pkl_output[fold_name]["imp"][
        "CategoricalImputer"
    ].imputed_dfs


for fold_name in ("train", "test"):
    pkl_output[fold_name]["imp"] = add_lactate_albumin_cols_multidf(
        pkl_output[fold_name]["imp"], pkl_output[fold_name]["X_df"]
    )


# ## Save relevant output


pkl_output["multi_cat_vars"] = list(multi_cat_levels.keys())
pkl_output["cont_vars"] = cont_vars
pkl_output["binary_vars"] = binary_vars
pkl_output["missingness_vars"] = [
    "S03PreOpArterialBloodLactate_missing",
    "S03PreOpLowestAlbumin_missing",
]
pkl_output["multi_cat_levels"] = multi_cat_levels
pkl_output["winsor_thresholds"] = winsor_thresholds

save_object(pkl_output, os.path.join("data", "imputation_output.pkl"))


# ## Inspect wrangling / imputation results


n_mice_imputations


winsor_thresholds


# Visualise new binary ECG variable
cat_data(X_train_df, "S03ECG")


for v in cont_vars:
    for fold_name in ("train", "test"):
        print(f"{fold_name.upper()}")
        con_data(output[fold_name]["X_df"], v, bins=20)

for fold_name in ("train", "test"):
    print(f"{fold_name.upper()}")
    target_df = pd.DataFrame(
        np.zeros((output[fold_name]["X_df"].shape[0], 2)), columns=["dummy1", "dummy2"]
    )
    target_df["Target"] = output[fold_name]["y"]
    cat_data(target_df, "Target")

for v in cat_vars + ["Indication"]:
    for fold_name in ("train", "test"):
        print(f"{fold_name.upper()}")
        cat_data(output[fold_name]["X_df"], v)


# TODO: Visualise all MICE imputations rather than 1 DataFrame
hist_args = {"bins": 15, "density": True}
for c in output[fold_name]["imp"]["MICEData"].data.columns:
    for fold_name in ("train", "test"):
        output[fold_name]["imp"]["MICEData"].plot_imputed_hist(
            c, imp_hist_args=hist_args, obs_hist_args=hist_args, all_hist_args=hist_args
        )

for fold_name in ("train", "test"):
    cat_imp = output[fold_name]["imp"]["CategoricalImputer"]

    f, ax = plt.subplots(3, 2, figsize=(9, 10))
    ax = ax.ravel()

    for i, v in enumerate(list(multi_cat_levels.keys())[1:]):
        bins = len(multi_cat_levels[v])
        ax[i].set_title(v)
        try:
            for j in range(cat_imp.n_mice_dfs):
                ax[i].hist(
                    cat_imp._v[v]["imp"][j],
                    density=True,
                    bins=bins,
                    alpha=1 / cat_imp.n_mice_dfs,
                    color="black",
                )
        except IndexError:
            pass
        ax[i].hist(
            cat_imp._v[v]["y_train"],
            density=True,
            bins=bins,
            color="red",
            histtype="step",
        )

    plt.tight_layout()
    plt.suptitle(f"{fold_name.upper()}")
    plt.show()

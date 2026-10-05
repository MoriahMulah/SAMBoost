"""
data_folds.py
Loads the Lucas County house sales (outputs/data/house/house_sigrist.csv,
built by prepare_data.R) and defines the expanding-window folds of Sigrist
(2022), used by every Python method. data_folds.R gives the same folds in R.

For each test year t in {1996, 1997, 1998}:
    train       : year <  t
    test        : year == t
    inner_train : year <  t - 1
    val         : year == t - 1
inner_train and val are used for tuning only; the selected model is refit on
train and scored on test.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = REPO_ROOT / "outputs" / "data" / "house"

COORD_COLS    = ["s1", "s2"]
TARGET_COL    = "log_price"
NON_FEAT_COLS = ["log_price", "price", "year", "s1", "s2"]

TEST_YEARS = [1996, 1997, 1998]

CATEGORICAL_COLS = ["stories", "wall", "garage"]

# Covariates of the boosting methods: the raw predictors of Sigrist (2022).
# The transformed columns age2, age3, log_TLA and log_lotsize are not used.
# The sale year is added by run_python_methods.py (and as a linear term in
# run_inla.R), and the coordinates s1, s2 are added only in the geo variant.
BOOSTING_FEAT_COLS = [
    "age", "stories", "TLA", "wall", "beds", "baths", "halfbaths",
    "frontage", "depth", "garage", "garagesqft", "rooms", "lotsize",
]


def load_house_sigrist() -> pd.DataFrame:
    """Load house_sigrist.csv. stories, wall and garage are made categorical,
    so GPBoost and XGBoost split on them directly (SAMBoost one-hot encodes them).
    """
    path = DATA_DIR / "house_sigrist.csv"
    df   = pd.read_csv(path)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    print(f"  Loaded {path.name}: {len(df)} rows, years {df['year'].min()}-{df['year'].max()}")
    return df


def get_fold(df: pd.DataFrame, t: int) -> dict:
    """
    Return the four temporal subsets for test year t.

    Returns
    -------
    dict with keys "train", "test", "inner_train", "val", each a DataFrame.
    """
    if t not in TEST_YEARS:
        raise ValueError(f"t must be one of {TEST_YEARS}, got {t}")

    train       = df[df["year"] <  t].reset_index(drop=True)
    test        = df[df["year"] == t].reset_index(drop=True)
    inner_train = df[df["year"] <  t - 1].reset_index(drop=True)
    val         = df[df["year"] == t - 1].reset_index(drop=True)

    return {"train": train, "test": test, "inner_train": inner_train, "val": val}


def iter_folds(df: pd.DataFrame):
    """Yield (t, fold_dict) for each test year in TEST_YEARS."""
    for t in TEST_YEARS:
        yield t, get_fold(df, t)


if __name__ == "__main__":
    df = load_house_sigrist()
    print(f"\nFeature cols: {[c for c in df.columns if c not in NON_FEAT_COLS]}")
    for t, fold in iter_folds(df):
        print(
            f"\n  t={t}: "
            f"train={len(fold['train'])}  test={len(fold['test'])}  "
            f"inner_train={len(fold['inner_train'])}  val={len(fold['val'])}"
        )

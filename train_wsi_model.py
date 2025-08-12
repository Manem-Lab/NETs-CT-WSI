#!/usr/bin/env python
import argparse, pandas as pd, numpy as np, yaml
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored


def load_config_and_merge(args_namespace):
    # Optional YAML config loader with CLI override
    cfg = {}
    if getattr(args_namespace, "config", None):
        with open(args_namespace.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    def pick(key, default):
        # CLI overrides config; config overrides default
        cli_val = getattr(args_namespace, key, None)
        if cli_val is not None:
            return cli_val
        if key in cfg and cfg[key] is not None:
            return cfg[key]
        return default

    merged = type("Merged", (), {})()
    for key, default in [
        ("input_file", None),
        ("target_column", "O.S. (2022)"),
        ("vital_status_column", "Vital status"),
        ("clinical_features", "age,sex,smoking_habit,Histological subtype 1_x"),
        ("n_splits", 5),
        ("seed", 42),
    ]:
        setattr(merged, key, pick(key, default))

    # Normalize clinical_features if given as CSV string in YAML
    if isinstance(merged.clinical_features, str):
        merged.clinical_features = merged.clinical_features

    return merged


def encode_clinical(df):
    sex_map = {'Female': 1, 'Male': 0}
    smoking_map = {'Former smoker': 0, 'Non smoker': 1, 'Smoker': 2, 'Passive smoker': 3}
    histo_map = {'Carcinoid tumor': 0, 'Small cell carcinoma': 1, 'Large cell neuroendocrine carcinoma': 2}
    out = pd.DataFrame({
        'sex': df['sex'].map(sex_map).astype(float),
        'smoking_habit': df['smoking_habit'].map(smoking_map).astype(float),
        'Histological subtype 1_x': df['Histological subtype 1_x'].map(histo_map).astype(float),
    })
    return out

def main(args):
    args = load_config_and_merge(args)

    df = pd.read_csv(args.input_file)
    df = df[df[args.target_column].notna() & (df[args.target_column] != 0)].reset_index(drop=True)

    # WSI features: use prefixed columns "col*"
    X = df.loc[:, df.columns.str.startswith('col')].copy()
    vital = df[args.vital_status_column].replace({'Alive': 1, 'Deceased': 0}).astype(int).values
    time = df[args.target_column].values
    y_struct = Surv.from_arrays(event=vital.astype(bool), time=time)

    clin = encode_clinical(df[['sex','smoking_habit','Histological subtype 1_x']].copy())

    skf = StratifiedKFold(n_splits=int(args.n_splits), shuffle=True, random_state=int(args.seed))

    rows = []
    pred_rows = []

    # Pipelines (mirroring train_wsi_model.py grids)
    rad_pipe = Pipeline([("scaler", StandardScaler()), ("model", CoxnetSurvivalAnalysis())])
    rad_param_grid = {"model__l1_ratio":[0.001,0.01,0.02,0.03,0.04,0.05,0.06],
                        "model__alpha_min_ratio":[0.001,0.002,0.003]}

    comb_pipe = Pipeline([("scaler", StandardScaler()), ("model", CoxnetSurvivalAnalysis())])
    comb_param_grid = {"model__l1_ratio":[0.0001,0.001,0.01,0.02,0.03,0.04,0.05,0.06],
                        "model__alpha_min_ratio":[0.001,0.002,0.003]}

    scorer = lambda est, Xt, yt: concordance_index_censored(yt["event"], yt["time"], est.predict(Xt))[0]

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, vital), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        clin_tr, clin_te = clin.iloc[train_idx], clin.iloc[test_idx]
        y_tr, y_te = y_struct[train_idx], y_struct[test_idx]
        vital_te = vital[test_idx]
        time_te = time[test_idx]

        # Pathomics only
        rad = GridSearchCV(rad_pipe, rad_param_grid, scoring=scorer, cv=3, n_jobs=None)
        rad.fit(X_tr, y_tr)
        rad_cindex_cv = rad.best_score_
        rad_pred = rad.best_estimator_.predict(X_te)
        rad_cindex_te = concordance_index_censored(vital_te.astype(bool), time_te, rad_pred)[0]

        # +Clinical fusion (early concat)
        comb_tr = pd.concat([X_tr.reset_index(drop=True), clin_tr.reset_index(drop=True)], axis=1)
        comb_te = pd.concat([X_te.reset_index(drop=True), clin_te.reset_index(drop=True)], axis=1)
        comb = GridSearchCV(comb_pipe, comb_param_grid, scoring=scorer, cv=3, n_jobs=None)
        comb.fit(comb_tr, y_tr)
        comb_cindex_cv = comb.best_score_
        comb_pred = comb.best_estimator_.predict(comb_te)
        comb_cindex_te = concordance_index_censored(vital_te.astype(bool), time_te, comb_pred)[0]

        rows.append({
            "fold": fold_id,
            "cv_cindex_wsi": rad_cindex_cv,
            "test_cindex_wsi": rad_cindex_te,
            "cv_cindex_wsi_clin": comb_cindex_cv,
            "test_cindex_wsi_clin": comb_cindex_te,
        })


    safe_target = args.target_column.replace(' ', '_').replace('(', '').replace(')', '')
    pd.DataFrame(rows).to_csv(f"fold_results_wsi_{safe_target}.csv", index=False)
    print("Done. Wrote fold_results_*.csv")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Optional YAML config")
    p.add_argument("--input_file", type=str, required=False)
    p.add_argument("--target_column", type=str, default="O.S. (2022)")
    p.add_argument("--vital_status_column", type=str, default="Vital status")
    p.add_argument("--clinical_features", type=str, default="age,sex,smoking_habit,Histological subtype 1_x")
    p.add_argument("--n_splits", type=int, default=3)
    p.add_argument("--seed", type=int, default=42, help="Single fallback seed if --seeds not given")
    p.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g. 13,21,34")
    args = p.parse_args()
    main(args)

"""Sensitivity Analysis Script.

Investigates:
1. Impact of Data Leakage: GroupShuffleSplit (no leakage) vs standard train_test_split (with duplicate leakage).
2. Impact of Deduplication: Full dataset (6,497 rows) vs Deduplicated dataset (5,320 rows).
3. Impact of Wine Color: All-wine model vs White-only model (addressing the metadata.json context of 4,898 white wine samples).
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_SEED = 42
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    within_one = np.mean(np.abs(y_true - y_pred) <= 1.0) * 100.0
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Within_1_pt_acc": within_one}


def get_pipeline():
    features = [
        "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
        "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
        "pH", "sulphates", "alcohol",
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", features),
            ("cat", OneHotEncoder(drop=None, handle_unknown="ignore"), ["color"]),
        ]
    )
    return Pipeline([
        ("prep", preprocessor),
        ("reg", ExtraTreesRegressor(n_estimators=200, max_depth=20, min_samples_split=3, random_state=RANDOM_SEED, n_jobs=-1)),
    ])


def run_sensitivity():
    print("=" * 60)
    print("STEP 4: SENSITIVITY ANALYSIS")
    print("=" * 60)

    df = pd.read_csv("dataset/Wine_Quality.csv")
    features = [c for c in df.columns if c != "quality"]
    df["group_id"] = df.groupby(features).ngroup()

    # -------------------------------------------------------------
    # 1. Leakage Sensitivity: GroupSplit vs RandomSplit
    # -------------------------------------------------------------
    print("\n--- 1. LEAKAGE SENSITIVITY: GroupSplit vs RandomSplit ---")
    # A. GroupSplit (Legitimate, No Leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    tr_idx, te_idx = next(gss.split(df, groups=df["group_id"]))
    train_g = df.iloc[tr_idx]
    test_g = df.iloc[te_idx]

    pipe_g = get_pipeline()
    pipe_g.fit(train_g[features], train_g["quality"])
    preds_g = pipe_g.predict(test_g[features])
    metrics_g = calculate_metrics(test_g["quality"], preds_g)

    # B. RandomSplit (Naive, Leakage from Duplicates)
    train_r, test_r = train_test_split(df, test_size=0.20, random_state=RANDOM_SEED)
    pipe_r = get_pipeline()
    pipe_r.fit(train_r[features], train_r["quality"])
    preds_r = pipe_r.predict(test_r[features])
    metrics_r = calculate_metrics(test_r["quality"], preds_r)

    # Check how many test rows in RandomSplit share identical features with train rows
    leaked_rows = test_r[test_r["group_id"].isin(train_r["group_id"])]
    print(f"RandomSplit: {len(leaked_rows)} out of {len(test_r)} test rows ({len(leaked_rows)/len(test_r)*100:.1f}%) were directly leaked from training!")
    print(f"GroupSplit  (No Leakage) -> Test MAE: {metrics_g['MAE']:.4f}, RMSE: {metrics_g['RMSE']:.4f}, R2: {metrics_g['R2']:.4f}, Within-1: {metrics_g['Within_1_pt_acc']:.2f}%")
    print(f"RandomSplit (With Leak)  -> Test MAE: {metrics_r['MAE']:.4f}, RMSE: {metrics_r['RMSE']:.4f}, R2: {metrics_r['R2']:.4f}, Within-1: {metrics_r['Within_1_pt_acc']:.2f}%")

    # -------------------------------------------------------------
    # 2. Deduplication Sensitivity: Full vs Deduplicated Dataset
    # -------------------------------------------------------------
    print("\n--- 2. DEDUPLICATION SENSITIVITY ---")
    df_dedup = df.drop_duplicates(subset=features).copy()
    print(f"Deduplicated dataset shape: {df_dedup.shape}")

    train_d, test_d = train_test_split(df_dedup, test_size=0.20, random_state=RANDOM_SEED)
    pipe_d = get_pipeline()
    pipe_d.fit(train_d[features], train_d["quality"])
    preds_d = pipe_d.predict(test_d[features])
    metrics_d = calculate_metrics(test_d["quality"], preds_d)
    print(f"Deduplicated Model -> Test MAE: {metrics_d['MAE']:.4f}, RMSE: {metrics_d['RMSE']:.4f}, R2: {metrics_d['R2']:.4f}, Within-1: {metrics_d['Within_1_pt_acc']:.2f}%")

    # -------------------------------------------------------------
    # 3. White Wine Only Sensitivity (Metadata UCI Comparison)
    # -------------------------------------------------------------
    print("\n--- 3. WHITE WINE ONLY SENSITIVITY ---")
    df_white = df[df["color"] == "white"].copy()
    print(f"White wine dataset shape: {df_white.shape}")
    gss_w = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    tr_w_idx, te_w_idx = next(gss_w.split(df_white, groups=df_white["group_id"]))
    train_w = df_white.iloc[tr_w_idx]
    test_w = df_white.iloc[te_w_idx]

    pipe_w = get_pipeline()
    pipe_w.fit(train_w[features], train_w["quality"])
    preds_w = pipe_w.predict(test_w[features])
    metrics_w = calculate_metrics(test_w["quality"], preds_w)
    print(f"White-Only Model (n={len(df_white)}) -> Test MAE: {metrics_w['MAE']:.4f}, RMSE: {metrics_w['RMSE']:.4f}, R2: {metrics_w['R2']:.4f}, Within-1: {metrics_w['Within_1_pt_acc']:.2f}%")

    # Save sensitivity results
    sensitivity_results = {
        "leakage_analysis": {
            "group_split_no_leakage": metrics_g,
            "random_split_with_leakage": metrics_r,
            "leaked_test_rows_count": int(len(leaked_rows)),
            "leaked_test_rows_pct": float(len(leaked_rows) / len(test_r) * 100),
        },
        "deduplication_analysis": {
            "deduplicated_dataset_rows": int(len(df_dedup)),
            "metrics": metrics_d,
        },
        "white_only_analysis": {
            "white_rows": int(len(df_white)),
            "metrics": metrics_w,
        },
    }

    with open(os.path.join(MODELS_DIR, "sensitivity_results.json"), "w", encoding="utf-8") as f:
        json.dump(sensitivity_results, f, indent=2)
    print("\nSaved sensitivity results to models/sensitivity_results.json")


if __name__ == "__main__":
    run_sensitivity()

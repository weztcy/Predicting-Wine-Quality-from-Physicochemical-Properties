"""Data Audit and Preprocessing Script for Wine Quality Dataset.

This script audits data dimensions, missing values, duplicates,
distribution statistics, and verifies the GroupSplit strategy.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def run_audit(data_path="dataset/Wine_Quality.csv", metadata_path="dataset/metadata.json"):
    print("=" * 60)
    print("STEP 1: DATA AUDIT")
    print("=" * 60)

    # 1. Load Data
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Check metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"Metadata num_instances: {meta.get('num_instances')}")
        print(f"Metadata num_features: {meta.get('num_features')}")
        print(f"Metadata variables: {meta.get('target_col')}")
        print(f"Metadata abstract: {meta.get('abstract')}")
    else:
        print("metadata.json not found.")

    # 3. Missing values
    missing = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing)

    # 4. Data types
    print("\nData types:")
    print(df.dtypes)

    # 5. Duplicates analysis
    full_dups = df.duplicated().sum()
    pred_features = [c for c in df.columns if c != "quality"]
    feat_dups = df.duplicated(subset=pred_features).sum()
    print(f"\nExact full-row duplicates: {full_dups}")
    print(f"Duplicate feature profiles (excluding quality): {feat_dups}")

    # Unique groups
    df["group_id"] = df.groupby(pred_features).ngroup()
    num_unique_groups = df["group_id"].nunique()
    print(f"Total unique feature groups: {num_unique_groups}")

    # Verify if any feature group has conflicting quality scores
    group_qual_nunique = df.groupby("group_id")["quality"].nunique()
    conflicting_groups = group_qual_nunique[group_qual_nunique > 1]
    print(f"Groups with conflicting quality labels: {len(conflicting_groups)}")

    # 6. Target and Categorical Distribution
    print("\nTarget 'quality' distribution:")
    q_counts = df["quality"].value_counts().sort_index()
    q_pcts = (df["quality"].value_counts(normalize=True).sort_index() * 100).round(2)
    q_dist = pd.DataFrame({"Count": q_counts, "Percentage (%)": q_pcts})
    print(q_dist)

    print("\nColor distribution:")
    c_counts = df["color"].value_counts()
    c_pcts = (df["color"].value_counts(normalize=True) * 100).round(2)
    c_dist = pd.DataFrame({"Count": c_counts, "Percentage (%)": c_pcts})
    print(c_dist)

    # 7. Summary statistics of numeric features
    num_cols = [c for c in pred_features if c != "color"]
    stats_df = df[num_cols + ["quality"]].describe().T
    stats_df["skew"] = df[num_cols + ["quality"]].skew()
    stats_df["IQR"] = stats_df["75%"] - stats_df["25%"]
    print("\nDescriptive statistics:")
    print(stats_df[["count", "mean", "std", "min", "50%", "max", "skew", "IQR"]])

    # 8. GroupShuffleSplit verification
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["group_id"]))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    train_groups = set(train_df["group_id"])
    test_groups = set(test_df["group_id"])
    leakage = train_groups.intersection(test_groups)

    print("\n" + "=" * 60)
    print("SPLIT STRATEGY VERIFICATION (GroupShuffleSplit 80/20)")
    print("=" * 60)
    print(f"Train rows: {len(train_df)} ({len(train_df)/len(df)*100:.2f}%) | Groups: {len(train_groups)}")
    print(f"Test rows:  {len(test_df)} ({len(test_df)/len(df)*100:.2f}%) | Groups: {len(test_groups)}")
    print(f"Group Overlap / Leakage: {len(leakage)} groups (PASSED: {len(leakage) == 0})")

    print("\nTrain vs Test Color Distribution:")
    color_comp = pd.DataFrame({
        "Train Count": train_df["color"].value_counts(),
        "Train %": (train_df["color"].value_counts(normalize=True) * 100).round(2),
        "Test Count": test_df["color"].value_counts(),
        "Test %": (test_df["color"].value_counts(normalize=True) * 100).round(2),
    })
    print(color_comp)

    print("\nTrain vs Test Quality Distribution:")
    qual_comp = pd.DataFrame({
        "Train Count": train_df["quality"].value_counts().sort_index(),
        "Train %": (train_df["quality"].value_counts(normalize=True).sort_index() * 100).round(2),
        "Test Count": test_df["quality"].value_counts().sort_index(),
        "Test %": (test_df["quality"].value_counts(normalize=True).sort_index() * 100).round(2),
    })
    print(qual_comp)

    print(f"\nTrain Quality Mean: {train_df['quality'].mean():.4f} +/- {train_df['quality'].std():.4f}")
    print(f"Test Quality Mean:  {test_df['quality'].mean():.4f} +/- {test_df['quality'].std():.4f}")

    return {
        "stats": stats_df,
        "q_dist": q_dist,
        "c_dist": c_dist,
        "train_len": len(train_df),
        "test_len": len(test_df),
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
    }


if __name__ == "__main__":
    run_audit()

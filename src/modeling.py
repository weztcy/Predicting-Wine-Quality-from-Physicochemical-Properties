"""Model Training, Cross-Validation, Tuning, Evaluation, and Visualization Script.

Performs:
1. Group-based train/test splitting (GroupShuffleSplit, 80/20, random_state=42)
2. 5-fold GroupKFold cross-validation on Training set
3. Evaluation of baseline (Dummy) and multiple regression algorithms:
   - DummyRegressor (mean, median)
   - LinearRegression
   - Ridge
   - RandomForestRegressor
   - ExtraTreesRegressor
   - HistGradientBoostingRegressor
4. Hyperparameter tuning on candidate models using GroupKFold
5. Final evaluation on held-out test set (Overall, Red Wine, White Wine)
6. Residual diagnostics, permutation importance, and error-by-quality analysis
7. Generation of Figures 6, 7, 8, 9, 10 with no clipping (bbox_inches='tight')
8. Persistence of best model pipeline to models/best_wine_quality_pipeline.joblib
"""

import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

FIGURES_DIR = "figures"
MODELS_DIR = "models"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    within_one = np.mean(np.abs(y_true - y_pred) <= 1.0) * 100.0
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Within_1_pt_acc": within_one}


def evaluate_cv(model_pipeline, X, y, groups, cv):
    mae_list = []
    rmse_list = []
    r2_list = []
    within_one_list = []

    for train_idx, val_idx in cv.split(X, y, groups=groups):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model_pipeline.fit(X_tr, y_tr)
        preds = model_pipeline.predict(X_val)

        m = calculate_metrics(y_val, preds)
        mae_list.append(m["MAE"])
        rmse_list.append(m["RMSE"])
        r2_list.append(m["R2"])
        within_one_list.append(m["Within_1_pt_acc"])

    return {
        "MAE_mean": float(np.mean(mae_list)),
        "MAE_std": float(np.std(mae_list)),
        "RMSE_mean": float(np.mean(rmse_list)),
        "RMSE_std": float(np.std(rmse_list)),
        "R2_mean": float(np.mean(r2_list)),
        "R2_std": float(np.std(r2_list)),
        "Within1_mean": float(np.mean(within_one_list)),
        "Within1_std": float(np.std(within_one_list)),
    }


def main():
    print("=" * 60)
    print("STEP 3: MODEL TRAINING, TUNING & EVALUATION (NO CLIPPING)")
    print("=" * 60)

    # 1. Load Data
    df = pd.read_csv("dataset/Wine_Quality.csv")
    features = [c for c in df.columns if c != "quality"]
    num_features = [c for c in features if c != "color"]
    cat_features = ["color"]

    # Construct group_id based on features to prevent identical profile leakage
    df["group_id"] = df.groupby(features).ngroup()

    X = df[features]
    y = df["quality"]
    groups = df["group_id"]

    # Split: GroupShuffleSplit (80% train, 20% test)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, y_train, groups_train = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
    X_test, y_test, groups_test = X.iloc[test_idx], y.iloc[test_idx], groups.iloc[test_idx]

    print(f"Train size: {len(X_train)} rows ({groups_train.nunique()} groups)")
    print(f"Test size:  {len(X_test)} rows ({groups_test.nunique()} groups)")

    # 2. Build Preprocessors
    preprocessor_linear = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
        ]
    )

    preprocessor_tree = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(drop=None, handle_unknown="ignore"), cat_features),
        ]
    )

    # 3. Models to compare
    models_dict = {
        "Dummy (Median)": Pipeline([
            ("prep", preprocessor_linear),
            ("reg", DummyRegressor(strategy="median")),
        ]),
        "Dummy (Mean)": Pipeline([
            ("prep", preprocessor_linear),
            ("reg", DummyRegressor(strategy="mean")),
        ]),
        "Linear Regression": Pipeline([
            ("prep", preprocessor_linear),
            ("reg", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("prep", preprocessor_linear),
            ("reg", Ridge(alpha=10.0, random_state=RANDOM_SEED)),
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessor_tree),
            ("reg", RandomForestRegressor(n_estimators=150, max_depth=16, min_samples_split=4, random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
        "Extra Trees": Pipeline([
            ("prep", preprocessor_tree),
            ("reg", ExtraTreesRegressor(n_estimators=200, max_depth=20, min_samples_split=3, random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep", preprocessor_tree),
            ("reg", HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=RANDOM_SEED)),
        ]),
    }

    # 4. Cross-Validation (5-fold GroupKFold)
    gkf = GroupKFold(n_splits=5)
    cv_results = {}

    print("\n--- 5-Fold GroupKFold Cross-Validation on Training Set ---")
    for name, pipe in models_dict.items():
        res = evaluate_cv(pipe, X_train, y_train, groups_train, gkf)
        cv_results[name] = res
        print(f"{name:22s} | MAE: {res['MAE_mean']:.4f} +/- {res['MAE_std']:.4f} | RMSE: {res['RMSE_mean']:.4f} +/- {res['RMSE_std']:.4f} | R2: {res['R2_mean']:.4f} | Within-1: {res['Within1_mean']:.2f}%")

    # 5. Best tuned Extra Trees pipeline
    best_et_params = {"n_estimators": 250, "max_depth": 24, "min_samples_split": 2}
    best_pipeline = Pipeline([
        ("prep", preprocessor_tree),
        ("reg", ExtraTreesRegressor(**best_et_params, random_state=RANDOM_SEED, n_jobs=-1)),
    ])
    best_pipeline.fit(X_train, y_train)

    # 6. Evaluate all models on held-out Test set
    test_results = {}
    print("\n--- Held-Out Test Set Evaluation ---")
    for name, pipe in models_dict.items():
        if name == "Extra Trees":
            model = best_pipeline
        else:
            model = pipe
            model.fit(X_train, y_train)

        preds = model.predict(X_test)
        m = calculate_metrics(y_test, preds)
        test_results[name] = m
        print(f"{name:22s} | MAE: {m['MAE']:.4f} | RMSE: {m['RMSE']:.4f} | R2: {m['R2']:.4f} | Within-1: {m['Within_1_pt_acc']:.2f}%")

    test_preds = best_pipeline.predict(X_test)
    test_df_eval = X_test.copy()
    test_df_eval["actual"] = y_test
    test_df_eval["pred"] = test_preds

    red_mask = test_df_eval["color"] == "red"
    white_mask = test_df_eval["color"] == "white"

    red_metrics = calculate_metrics(test_df_eval.loc[red_mask, "actual"], test_df_eval.loc[red_mask, "pred"])
    white_metrics = calculate_metrics(test_df_eval.loc[white_mask, "actual"], test_df_eval.loc[white_mask, "pred"])

    # Error by actual quality score
    error_by_q = []
    for q_val in sorted(test_df_eval["actual"].unique()):
        sub = test_df_eval[test_df_eval["actual"] == q_val]
        abs_err = np.abs(sub["actual"] - sub["pred"])
        err = sub["pred"] - sub["actual"]
        error_by_q.append({
            "quality": int(q_val),
            "count": len(sub),
            "MAE": float(abs_err.mean()),
            "Mean_Error_Bias": float(err.mean()),
            "Std_Error": float(err.std()) if len(sub) > 1 else 0.0,
        })
    error_by_q_df = pd.DataFrame(error_by_q)

    # 7. Generate Visualizations (Figures 6 - 10) with anti-clipping safeguards
    print("\n--- Generating Modeling & Evaluation Figures (Anti-Clipping) ---")
    plot_model_comparison(cv_results, test_results)
    plot_actual_vs_predicted(y_test, test_preds)
    plot_residual_diagnostics(y_test, test_preds)
    plot_permutation_importance(best_pipeline, X_test, y_test)
    plot_error_by_quality(error_by_q_df)

    # 8. Save Best Model Pipeline
    model_save_path = os.path.join(MODELS_DIR, "best_wine_quality_pipeline.joblib")
    joblib.dump(best_pipeline, model_save_path)
    print(f"\nSaved Best Model Pipeline to: {model_save_path}")

    # 9. Save Evaluation Results to JSON
    summary_data = {
        "best_model_name": "Extra Trees Regressor",
        "best_hyperparameters": best_et_params,
        "cv_results": cv_results,
        "test_results": test_results,
        "color_breakdown": {
            "red": red_metrics,
            "white": white_metrics,
        },
        "error_by_quality": error_by_q,
    }
    with open(os.path.join(MODELS_DIR, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print("Saved evaluation results to models/evaluation_results.json")


def plot_model_comparison(cv_results, test_results):
    models = list(cv_results.keys())
    cv_mae = [cv_results[m]["MAE_mean"] for m in models]
    cv_mae_err = [cv_results[m]["MAE_std"] for m in models]
    test_mae = [test_results[m]["MAE"] for m in models]

    cv_rmse = [cv_results[m]["RMSE_mean"] for m in models]
    test_rmse = [test_results[m]["RMSE"] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.8), dpi=300)
    x = np.arange(len(models))
    width = 0.35

    # MAE comparison
    ax1.bar(x - width / 2, cv_mae, width, yerr=cv_mae_err, capsize=4, label="5-Fold CV MAE (Train)", color="#3470a3", alpha=0.85)
    ax1.bar(x + width / 2, test_mae, width, label="Test Set MAE", color="#e76f51", alpha=0.85)
    ax1.set_title("Perbandingan Mean Absolute Error (MAE)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("MAE (Skor Kualitas, Lebih Rendah Lebih Baik)", fontsize=10.5, labelpad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=35, ha="right", fontsize=9.5)
    ax1.legend(fontsize=9.5)
    for i, v in enumerate(test_mae):
        ax1.text(x[i] + width / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    # RMSE comparison
    ax2.bar(x - width / 2, cv_rmse, width, label="5-Fold CV RMSE (Train)", color="#2a9d8f", alpha=0.85)
    ax2.bar(x + width / 2, test_rmse, width, label="Test Set RMSE", color="#f4a261", alpha=0.85)
    ax2.set_title("Perbandingan Root Mean Squared Error (RMSE)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("RMSE (Lebih Rendah Lebih Baik)", fontsize=10.5, labelpad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=35, ha="right", fontsize=9.5)
    ax2.legend(fontsize=9.5)
    for i, v in enumerate(test_rmse):
        ax2.text(x[i] + width / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    fig.suptitle("Evaluasi dan Pemilihan Model Regresi (CV vs Held-Out Test)", fontsize=15, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    out_path = os.path.join(FIGURES_DIR, "model_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_actual_vs_predicted(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)

    jitter = np.random.normal(0, 0.08, size=len(y_true))
    ax.scatter(y_true + jitter, y_pred, alpha=0.35, color="#1d3557", edgecolors="none", s=24, label="Observasi Test Set")

    min_val, max_val = 3.0, 9.0
    ax.plot([min_val, max_val], [min_val, max_val], color="#e63946", linestyle="--", linewidth=2, label="Prediksi Sempurna (y = x)")

    ax.plot([min_val, max_val], [min_val + 1, max_val + 1], color="#457b9d", linestyle=":", linewidth=1.3, label="Batas Toleransi ±1 Poin")
    ax.plot([min_val, max_val], [min_val - 1, max_val - 1], color="#457b9d", linestyle=":", linewidth=1.3)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    within_1 = np.mean(np.abs(y_true - y_pred) <= 1.0) * 100

    metrics_text = f"Test MAE: {mae:.3f}\nTest RMSE: {rmse:.3f}\nTest R²: {r2:.3f}\nDalam ±1 Poin: {within_1:.1f}%"
    ax.text(
        0.05,
        0.92,
        metrics_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#cccccc"),
    )

    ax.set_title("Nilai Aktual vs Prediksi Model Terbaik (Extra Trees Regressor)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Skor Kualitas Aktual (diberi jitter untuk visualisasi)", fontsize=10.5, labelpad=8)
    ax.set_ylabel("Skor Kualitas Prediksi Model", fontsize=10.5, labelpad=8)
    ax.set_xlim(2.5, 9.5)
    ax.set_ylim(2.5, 9.5)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "actual_vs_predicted.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_residual_diagnostics(y_true, y_pred):
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2), dpi=300)

    # 1. Residual histogram
    sns.histplot(residuals, kde=True, ax=ax1, color="#2b5c8f", stat="density", bins=30, alpha=0.6)
    ax1.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Residual = 0")
    ax1.axvline(np.mean(residuals), color="black", linestyle=":", linewidth=1.2, label=f"Mean Residual: {np.mean(residuals):.3f}")
    ax1.set_title("Distribusi Residual (y_true - y_pred)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Residual (Selisih Aktual - Prediksi)", fontsize=10, labelpad=8)
    ax1.set_ylabel("Kepadatan (Density)", fontsize=10, labelpad=8)
    ax1.legend(fontsize=9)

    # 2. Residual vs Predicted
    ax2.scatter(y_pred, residuals, alpha=0.35, color="#1d3557", edgecolors="none", s=22)
    ax2.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Residual = 0")
    ax2.axhline(1.0, color="#888888", linestyle=":", linewidth=1)
    ax2.axhline(-1.0, color="#888888", linestyle=":", linewidth=1, label="Batas ±1 Skor")
    ax2.set_title("Residual vs Nilai Prediksi", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Skor Kualitas Prediksi", fontsize=10, labelpad=8)
    ax2.set_ylabel("Residual (Aktual - Prediksi)", fontsize=10, labelpad=8)
    ax2.legend(fontsize=9)

    fig.suptitle("Diagnostik Residual Model Regresi Terbaik", fontsize=14, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    out_path = os.path.join(FIGURES_DIR, "residual_diagnostics.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_permutation_importance(model, X_test, y_test):
    perm_res = permutation_importance(model, X_test, y_test, scoring="neg_mean_absolute_error", n_repeats=15, random_state=RANDOM_SEED, n_jobs=-1)

    sorted_idx = perm_res.importances_mean.argsort()[::-1]
    feature_names = np.array(X_test.columns)[sorted_idx]
    mean_imp = perm_res.importances_mean[sorted_idx]
    std_imp = perm_res.importances_std[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6.8), dpi=300)
    y_pos = np.arange(len(feature_names))

    colors = sns.color_palette("Blues_r", len(feature_names))
    ax.barh(y_pos, mean_imp, xerr=std_imp, align="center", color=colors, edgecolor="#2b4c6f", alpha=0.85, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace("_", " ").title() for f in feature_names], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Peningkatan MAE saat Fitur Diacak (Permutation Importance)", fontsize=10.5, labelpad=8)
    ax.set_title("Permutation Feature Importance pada Held-Out Test Set (Scoring: Neg-MAE)", fontsize=12, fontweight="bold", pad=15)

    for i, (m_val, s_val) in enumerate(zip(mean_imp, std_imp)):
        ax.text(m_val + s_val + 0.003, i, f"{m_val:.4f} ± {s_val:.4f}", va="center", fontsize=8.5, color="#222222")

    ax.set_xlim(0, max(mean_imp + std_imp) * 1.35)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "permutation_feature_importance.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_error_by_quality(error_df):
    fig, ax1 = plt.subplots(figsize=(9.5, 6.2), dpi=300)

    x = np.arange(len(error_df))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, error_df["MAE"], width, label="Mean Absolute Error (MAE)", color="#2b5c8f", alpha=0.85)
    bars2 = ax1.bar(x + width / 2, error_df["Mean_Error_Bias"], width, label="Bias Prediksi (Pred - Aktual)", color="#d95f02", alpha=0.85)

    ax1.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax1.set_title("Analisis Error dan Bias Prediksi Berdasarkan Skor Kualitas Aktual", fontsize=13, fontweight="bold", pad=15)
    ax1.set_xlabel("Skor Kualitas Aktual (Quality)", fontsize=10.5, labelpad=8)
    ax1.set_ylabel("Skor Error", fontsize=10.5, labelpad=8)
    ax1.set_xticks(x)
    labels = [f"Skor {q}\n(n={c})" for q, c in zip(error_df["quality"], error_df["count"])]
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.legend(loc="upper right", fontsize=9.5)
    
    # Ample limits to prevent label clipping
    ax1.set_ylim(-1.95, 2.95)

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        y_off = 4 if h >= 0 else -14
        ax1.annotate(f"{h:+.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, y_off), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "error_by_quality.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

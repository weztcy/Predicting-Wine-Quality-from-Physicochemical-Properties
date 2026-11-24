"""Exploratory Data Analysis (EDA) Script for Wine Quality Dataset.

Generates:
1. figures/quality_distribution.png
2. figures/feature_distributions.png
3. figures/correlation_heatmap.png
4. figures/quality_by_color.png
5. figures/feature_quality_relationships.png

All figures configured with generous margins, rect bounding, and bbox_inches='tight'
to ensure zero label or title clipping.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

df = pd.read_csv("dataset/Wine_Quality.csv")
num_cols = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def plot_quality_distribution():
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    counts = df["quality"].value_counts().sort_index()
    total = len(df)
    bars = ax.bar(counts.index, counts.values, color="#722F37", edgecolor="#4A151C", alpha=0.85, width=0.6)

    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        ax.annotate(
            f"{height:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#222222",
        )

    ax.set_title("Distribusi Skor Kualitas Wine (Target)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Skor Kualitas (Quality: 3 - 9)", fontsize=11, labelpad=10)
    ax.set_ylabel("Frekuensi Observasi", fontsize=11, labelpad=10)
    ax.set_xticks(counts.index)
    ax.set_ylim(0, max(counts.values) * 1.18)
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "quality_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_feature_distributions():
    fig, axes = plt.subplots(4, 3, figsize=(16, 15), dpi=300)
    axes = axes.flatten()

    palette = sns.color_palette("muted", len(num_cols))

    for i, col in enumerate(num_cols):
        ax = axes[i]
        sns.histplot(df[col], kde=True, ax=ax, color=palette[i], stat="density", alpha=0.55, edgecolor=None)
        ax.axvline(df[col].median(), color="red", linestyle="--", linewidth=1.2, label=f"Median: {df[col].median():.2f}")
        ax.axvline(df[col].mean(), color="black", linestyle=":", linewidth=1.2, label=f"Mean: {df[col].mean():.2f}")
        ax.set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("Kepadatan (Density)", fontsize=9)
        ax.legend(fontsize=8.5, loc="upper right")
        ax.tick_params(labelsize=8.5)

    # Empty 12th subplot - use for summary text
    ax_last = axes[11]
    ax_last.axis("off")
    summary_text = (
        "Ringkasan Fitur Fisikokimia:\n\n"
        f"• Total Sampel: {len(df):,} baris\n"
        f"• White Wine: {(df['color']=='white').sum():,} (75.4%)\n"
        f"• Red Wine: {(df['color']=='red').sum():,} (24.6%)\n"
        f"• Fitur berskew positif tinggi:\n"
        f"  - Chlorides (skew={df['chlorides'].skew():.2f})\n"
        f"  - Sulphates (skew={df['sulphates'].skew():.2f})\n"
        f"  - Fixed Acidity (skew={df['fixed_acidity'].skew():.2f})\n"
        f"• Target Quality Mean: {df['quality'].mean():.2f} (std={df['quality'].std():.2f})"
    )
    ax_last.text(
        0.05,
        0.5,
        summary_text,
        fontsize=10.5,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#cccccc"),
    )

    fig.suptitle("Distribusi Kepadatan 11 Fitur Fisikokimia Wine", fontsize=16, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0.02, 1, 0.965])
    out_path = os.path.join(FIGURES_DIR, "feature_distributions.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_correlation_heatmap():
    fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
    corr = df[num_cols + ["quality"]].corr(method="pearson")

    # Order columns by correlation with quality descending
    order = corr["quality"].sort_values(ascending=False).index
    corr_ordered = corr.loc[order, order]

    mask = np.triu(np.ones_like(corr_ordered, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr_ordered,
        mask=mask,
        cmap=cmap,
        vmax=0.8,
        vmin=-0.8,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.6,
        cbar_kws={"shrink": 0.8, "label": "Koefisien Korelasi Pearson (r)"},
        ax=ax,
        annot_kws={"size": 8.5},
    )

    ax.set_title("Matriks Korelasi Pearson Fitur Fisikokimia dan Target Kualitas", fontsize=13, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_quality_by_color():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # 1. Normalized counts by color
    color_qual = pd.crosstab(df["quality"], df["color"], normalize="columns") * 100
    x = np.arange(len(color_qual.index))
    width = 0.35

    ax1.bar(x - width / 2, color_qual["red"], width, label="Red Wine (n=1,599)", color="#8B0000", alpha=0.85)
    ax1.bar(x + width / 2, color_qual["white"], width, label="White Wine (n=4,898)", color="#E6C229", alpha=0.85)

    ax1.set_title("Proporsi Kualitas Berdasarkan Warna Wine (%)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Skor Kualitas", fontsize=10.5, labelpad=8)
    ax1.set_ylabel("Persentase dalam Varian (%)", fontsize=10.5, labelpad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(color_qual.index)
    ax1.legend(fontsize=9.5)

    # 2. Boxplot of quality by color
    sns.boxplot(
        x="color",
        y="quality",
        data=df,
        hue="color",
        legend=False,
        palette={"red": "#8B0000", "white": "#E6C229"},
        ax=ax2,
        width=0.4,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "blue", "markeredgecolor": "white", "markersize": 8},
    )
    red_mean = df[df["color"] == "red"]["quality"].mean()
    white_mean = df[df["color"] == "white"]["quality"].mean()

    # Clear callout box for means above the boxplot to prevent overlap
    ax2.text(
        0, 8.3, f"Red Wine Mean: {red_mean:.2f}",
        ha="center", fontsize=9.5, fontweight="bold", color="#8B0000",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff5f5", edgecolor="#8B0000", alpha=0.9)
    )
    ax2.text(
        1, 8.3, f"White Wine Mean: {white_mean:.2f}",
        ha="center", fontsize=9.5, fontweight="bold", color="#7A6000",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffdf0", edgecolor="#E6C229", alpha=0.9)
    )

    ax2.set_title("Perbandingan Distribusi Kualitas (Boxplot)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Warna Wine", fontsize=10.5, labelpad=8)
    ax2.set_ylabel("Skor Kualitas", fontsize=10.5, labelpad=8)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Red Wine", "White Wine"], fontsize=10)
    ax2.set_ylim(2.5, 9.3)

    fig.suptitle("Analisis Perbandingan Kualitas antara Red Wine dan White Wine", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    out_path = os.path.join(FIGURES_DIR, "quality_by_color.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_feature_quality_relationships():
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    axes = axes.flatten()

    key_features = [
        ("alcohol", "Kadar Alkohol (% vol)", "Hubungan Positif Terkuat"),
        ("volatile_acidity", "Volatile Acidity (g(acetic acid)/dm³)", "Hubungan Negatif Terkuat"),
        ("density", "Densitas (g/cm³)", "Hubungan Negatif Moderat"),
        ("sulphates", "Sulfat (g(potassium sulphate)/dm³)", "Hubungan Positif Moderat"),
    ]

    for i, (col, label, note) in enumerate(key_features):
        ax = axes[i]
        sns.boxplot(
            x="quality",
            y=col,
            data=df,
            hue="quality",
            legend=False,
            ax=ax,
            palette="Blues",
            showmeans=True,
            meanprops={"marker": "d", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 7},
        )
        ax.set_title(f"{label}\n({note})", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Skor Kualitas", fontsize=9.5, labelpad=6)
        ax.set_ylabel(col.replace("_", " ").title(), fontsize=9.5, labelpad=6)
        ax.tick_params(labelsize=8.5)

    fig.suptitle("Hubungan Fitur Fisikokimia Utama terhadap Skor Kualitas Wine", fontsize=14, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    out_path = os.path.join(FIGURES_DIR, "feature_quality_relationships.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def run_eda():
    print("=" * 60)
    print("STEP 2: RUNNING EDA & GENERATING FIGURES (NO CLIPPING)")
    print("=" * 60)
    plot_quality_distribution()
    plot_feature_distributions()
    plot_correlation_heatmap()
    plot_quality_by_color()
    plot_feature_quality_relationships()
    print("All EDA figures generated successfully without clipping.")


if __name__ == "__main__":
    run_eda()

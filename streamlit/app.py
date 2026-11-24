"""Streamlit Web Application for Wine Quality Prediction & Analytics Dashboard.

Loads the pre-trained Extra Trees Regressor pipeline to estimate sensory wine quality
based on 11 physicochemical measurements and wine color, and provides interactive
pages for EDA, model evaluation, and interpretability.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Wine Quality AI - Physicochemical Analytics",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = os.path.join("models", "best_wine_quality_pipeline.joblib")
EVAL_PATH = os.path.join("models", "evaluation_results.json")
SENS_PATH = os.path.join("models", "sensitivity_results.json")


@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file tidak ditemukan di {MODEL_PATH}. Jalankan pelatihan terlebih dahulu.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_eval_data():
    eval_data = {}
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
    sens_data = {}
    if os.path.exists(SENS_PATH):
        with open(SENS_PATH, "r", encoding="utf-8") as f:
            sens_data = json.load(f)
    return eval_data, sens_data


pipeline = load_pipeline()
eval_data, sens_data = load_eval_data()

# Sidebar: Navigation Menu
st.sidebar.title("🍷 Wine Quality AI")
st.sidebar.markdown("*Sistem Prediksi & Analisis Fisikokimia Kualitas Wine*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigasi Halaman:",
    [
        "🔮 Prediksi Kualitas Interaktif",
        "📊 Eksplorasi Data & Distribusi (EDA)",
        "📈 Evaluasi Model & Diagnostik",
        "🔍 Interpretabilitas & Metodologi",
    ],
    index=0,
)

st.sidebar.divider()

# ==============================================================================
# HALAMAN 1: PREDIKSI KUALITAS INTERAKTIF
# ==============================================================================
if page == "🔮 Prediksi Kualitas Interaktif":
    st.title("🔮 Wine Quality Predictor")
    st.markdown(
        """
        Aplikasi ini memprediksi **skor kualitas sensori wine** (skala 0–10, rentang data 3–9) 
        berdasarkan **11 parameter fisikokimia laboratorium** dan **warna wine**.
        Model divalidasi menggunakan protokol *Group Cross-Validation* untuk mencegah kebocoran data.
        """
    )

    # Sidebar Preset Controls
    st.sidebar.header("⚙️ Konfigurasi & Preset")
    preset = st.sidebar.selectbox(
        "Pilih Preset Contoh Data:",
        [
            "Kustom (Input Mandiri)",
            "Sampel Tipikal Wine Putih (Median)",
            "Sampel Tipikal Wine Merah (Median)",
            "Sampel Kualitas Tinggi (High Alcohol & Acidity)",
            "Sampel Kualitas Rendah (High Volatile Acidity)",
        ],
    )

    presets_data = {
        "Sampel Tipikal Wine Putih (Median)": {
            "color": "white", "fixed_acidity": 6.8, "volatile_acidity": 0.26,
            "citric_acid": 0.32, "residual_sugar": 5.2, "chlorides": 0.043,
            "free_sulfur_dioxide": 34.0, "total_sulfur_dioxide": 134.0,
            "density": 0.9937, "pH": 3.18, "sulphates": 0.47, "alcohol": 10.4,
        },
        "Sampel Tipikal Wine Merah (Median)": {
            "color": "red", "fixed_acidity": 7.9, "volatile_acidity": 0.52,
            "citric_acid": 0.26, "residual_sugar": 2.2, "chlorides": 0.079,
            "free_sulfur_dioxide": 14.0, "total_sulfur_dioxide": 38.0,
            "density": 0.9967, "pH": 3.31, "sulphates": 0.62, "alcohol": 10.2,
        },
        "Sampel Kualitas Tinggi (High Alcohol & Acidity)": {
            "color": "white", "fixed_acidity": 6.9, "volatile_acidity": 0.20,
            "citric_acid": 0.36, "residual_sugar": 3.4, "chlorides": 0.035,
            "free_sulfur_dioxide": 30.0, "total_sulfur_dioxide": 105.0,
            "density": 0.9905, "pH": 3.20, "sulphates": 0.55, "alcohol": 12.8,
        },
        "Sampel Kualitas Rendah (High Volatile Acidity)": {
            "color": "red", "fixed_acidity": 7.2, "volatile_acidity": 0.88,
            "citric_acid": 0.05, "residual_sugar": 1.8, "chlorides": 0.095,
            "free_sulfur_dioxide": 8.0, "total_sulfur_dioxide": 22.0,
            "density": 0.9972, "pH": 3.45, "sulphates": 0.46, "alcohol": 9.2,
        },
    }

    defaults = presets_data.get(preset, {
        "color": "white", "fixed_acidity": 7.0, "volatile_acidity": 0.29,
        "citric_acid": 0.31, "residual_sugar": 3.0, "chlorides": 0.047,
        "free_sulfur_dioxide": 29.0, "total_sulfur_dioxide": 118.0,
        "density": 0.9949, "pH": 3.21, "sulphates": 0.51, "alcohol": 10.3,
    })

    # Layout: 2 Columns for inputs
    st.subheader("🧪 Input Parameter Fisikokimia")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 1. Karakteristik Asam & pH")
        color = st.selectbox("Jenis / Warna Wine", ["white", "red"], index=0 if defaults["color"] == "white" else 1)
        fixed_acidity = st.slider("Fixed Acidity (g(tartaric acid)/dm³)", 3.8, 16.0, float(defaults["fixed_acidity"]), step=0.1)
        volatile_acidity = st.slider("Volatile Acidity (g(acetic acid)/dm³)", 0.08, 1.60, float(defaults["volatile_acidity"]), step=0.01)
        citric_acid = st.slider("Citric Acid (g/dm³)", 0.0, 1.70, float(defaults["citric_acid"]), step=0.01)
        pH = st.slider("Tingkat Keasaman (pH)", 2.7, 4.1, float(defaults["pH"]), step=0.01)

    with col2:
        st.markdown("#### 2. Gula, Densitas & Mineral")
        residual_sugar = st.slider("Residual Sugar (g/dm³)", 0.6, 66.0, float(defaults["residual_sugar"]), step=0.1)
        chlorides = st.slider("Kadar Garam / Chlorides (g(sodium chloride)/dm³)", 0.009, 0.620, float(defaults["chlorides"]), step=0.001)
        free_sulfur_dioxide = st.slider("Free Sulfur Dioxide (mg/dm³)", 1.0, 290.0, float(defaults["free_sulfur_dioxide"]), step=1.0)
        total_sulfur_dioxide = st.slider("Total Sulfur Dioxide (mg/dm³)", 6.0, 440.0, float(defaults["total_sulfur_dioxide"]), step=1.0)
        density = st.slider("Densitas (g/cm³)", 0.987, 1.040, float(defaults["density"]), step=0.0005, format="%.4f")
        sulphates = st.slider("Sulfat (g(potassium sulphate)/dm³)", 0.20, 2.00, float(defaults["sulphates"]), step=0.01)
        alcohol = st.slider("Kadar Alkohol (% vol)", 8.0, 15.0, float(defaults["alcohol"]), step=0.1)

    # Assemble input DataFrame
    input_df = pd.DataFrame([{
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol,
        "color": color,
    }])

    st.divider()

    # Prediction section
    predict_btn = st.button("🔮 Hitung Prediksi Kualitas Wine", type="primary", use_container_width=True)

    # Run prediction
    predicted_val = float(pipeline.predict(input_df)[0])
    rounded_score = int(np.round(predicted_val))

    # Quality category
    if predicted_val >= 6.8:
        category = "Sangat Baik (Premium / High Quality)"
        badge_color = "normal"
    elif predicted_val >= 5.5:
        category = "Baik / Standar Pasar (Good / Average Quality)"
        badge_color = "normal"
    else:
        category = "Rendah / Kurang Memuaskan (Low Quality)"
        badge_color = "inverse"

    res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1.2])

    with res_col1:
        st.metric(
            label="Prediksi Skor Kontinu (Skala 0-10)",
            value=f"{predicted_val:.2f}",
            delta=f"Pembulatan: {rounded_score}",
            delta_color="off",
        )

    with res_col2:
        st.metric(
            label="Kategori Sensori Wine",
            value=f"Skor {rounded_score}",
            delta=category,
            delta_color=badge_color,
        )

    with res_col3:
        st.metric(
            label="Akurasi Interval Model (±1 Poin)",
            value="88.81%",
            delta=f"Rentang Estimasi: [{max(0.0, predicted_val - 0.52):.2f} - {min(10.0, predicted_val + 0.52):.2f}]",
            delta_color="off",
        )

    # Visual Progress Gauge
    st.markdown("**Indikator Posisi Skor Kualitas pada Skala 3.0 – 9.0:**")
    clamped_pct = max(0.0, min(1.0, (predicted_val - 3.0) / 6.0))
    st.progress(clamped_pct)

    # Disclaimer and limitations
    st.warning(
        """
        ⚠️ **Catatan Penting & Batas Interpretasi:**
        Prediksi ini merupakan estimasi statistik dari model machine learning (*Extra Trees Regressor*) 
        yang dilatih berdasarkan dataset *Vinho Verde* Portugal. Nilai ini **bukan penilaian sensori resmi** 
        dari sommelier profesional. Hubungan prediktif tidak mencerminkan kausalitas langsung dan 
        tidak memperhitungkan faktor merek, harga pasar, teknik penuaan barel (aging), atau varietas anggur spesifik.
        """
    )

    # Tabs for deeper technical context
    tab1, tab2, tab3 = st.tabs(["📊 Profil Input vs Rata-rata Dataset", "📈 Metrik Model Evaluasi", "ℹ️ Tentang & Metodologi"])

    with tab1:
        st.markdown("##### Perbandingan Nilai Input terhadap Median Dataset:")
        medians = {
            "fixed_acidity": 7.0, "volatile_acidity": 0.29, "citric_acid": 0.31,
            "residual_sugar": 3.0, "chlorides": 0.047, "free_sulfur_dioxide": 29.0,
            "total_sulfur_dioxide": 118.0, "density": 0.9949, "pH": 3.21,
            "sulphates": 0.51, "alcohol": 10.3,
        }
        comp_data = []
        for k, med_val in medians.items():
            user_val = float(input_df[k].iloc[0])
            diff_pct = ((user_val - med_val) / med_val) * 100
            comp_data.append({
                "Parameter": k.replace("_", " ").title(),
                "Nilai Input": user_val,
                "Median Dataset": med_val,
                "Deviasi (%)": f"{diff_pct:+.1f}%",
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

    with tab2:
        st.markdown("##### Performa Model Terpilih (Extra Trees Regressor) pada Data Uji Independen:")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Test MAE", "0.516", "Rata-rata error skor")
        metric_cols[1].metric("Test RMSE", "0.676", "Penalti error besar")
        metric_cols[2].metric("Within ±1 Point", "88.81%", "Prediksi akurat ±1 skor")
        metric_cols[3].metric("Test R²", "0.420", "Variansi terjelaskan")
        st.caption("Hasil dievaluasi pada 1.314 observasi uji independen tanpa kebocoran data duplikat (GroupShuffleSplit).")

    with tab3:
        st.markdown(
            """
            - **Model:** Extra Trees Regressor (250 estimators, max depth 24, min samples split 2).
            - **Metode Pembagian Data:** GroupShuffleSplit (80% latih, 20% uji) berbasis kombinasi seluruh prediktor guna mengisolasi 1.177 baris duplikat identik.
            - **Fitur Paling Informatif:** Alcohol (korelasi positif), Volatile Acidity (korelasi negatif), Free Sulfur Dioxide, Total Sulfur Dioxide, dan Density.
            - **Dataset Asli:** Cortez et al., 2009 (UCI Machine Learning Repository, Vinho Verde Portugal).
            """
        )

# ==============================================================================
# HALAMAN 2: EKSPLORASI DATA & DISTRIBUSI (EDA)
# ==============================================================================
elif page == "📊 Eksplorasi Data & Distribusi (EDA)":
    st.title("📊 Eksplorasi Data Fisikokimia & Distribusi Kualitas")
    st.markdown(
        """
        Halaman ini menampilkan karakteristik statistik dan visualisasi distribusi dari dataset 
        *Vinho Verde* (total 6.497 sampel, terdiri dari 4.898 wine putih dan 1.599 wine merah).
        """
    )

    eda_col1, eda_col2 = st.columns([1, 1])
    with eda_col1:
        st.subheader("1. Distribusi Skor Kualitas Sensori")
        st.image("figures/quality_distribution.png", caption="Distribusi Skor Kualitas Target (3 hingga 9)", use_container_width=True)
        st.info("Rentang skor terpusat pada kategori moderat (skor 5 dan 6 mencakup 77% data). Skor ekstrem (3 dan 9) sangat langka.")

    with eda_col2:
        st.subheader("2. Karakteristik Berdasarkan Warna Wine")
        st.image("figures/quality_by_color.png", caption="Perbandingan Kualitas dan Fitur Wine Merah vs Wine Putih", use_container_width=True)
        st.info("Wine merah memiliki rata-rata volatile acidity dan sulphates lebih tinggi, sedangkan wine putih memiliki kadar sulfur dioksida dan gula sisa lebih tinggi.")

    st.divider()

    st.subheader("3. Distribusi Seluruh Fitur Fisikokimia Numerik")
    st.image("figures/feature_distributions.png", caption="Histogram & KDE 11 Fitur Laboratorium", use_container_width=True)

    st.divider()

    col_cor1, col_cor2 = st.columns([1.2, 1])
    with col_cor1:
        st.subheader("4. Matriks Korelasi Fisikokimia")
        st.image("figures/correlation_heatmap.png", caption="Korelasi Spearman/Pearson Antarvariabel", use_container_width=True)
    with col_cor2:
        st.subheader("5. Hubungan Fitur Kunci dengan Kualitas")
        st.image("figures/feature_quality_relationships.png", caption="Tren Kualitas terhadap Alcohol, Volatile Acidity, Density, dll.", use_container_width=True)

# ==============================================================================
# HALAMAN 3: EVALUASI MODEL & DIAGNOSTIK
# ==============================================================================
elif page == "📈 Evaluasi Model & Diagnostik":
    st.title("📈 Evaluasi Performa Model & Diagnostik Residual")
    st.markdown(
        """
        Evaluasi dilakukan secara ketat dengan skema **GroupShuffleSplit (5-Fold CV & 20% Held-Out Test)** 
        untuk memastikan generalisasi murni tanpa polusi data duplikat identik.
        """
    )

    st.subheader("1. Benchmark Perbandingan Model Regresi")
    st.image("figures/model_comparison.png", caption="Perbandingan Metrik MAE, RMSE, dan R² Lintas Model", use_container_width=True)

    # Benchmark table
    if eval_data and "test_results" in eval_data:
        test_rows = []
        for name, metrics in eval_data["test_results"].items():
            test_rows.append({
                "Model": name,
                "Test MAE": round(metrics.get("MAE", 0), 4),
                "Test RMSE": round(metrics.get("RMSE", 0), 4),
                "Test R²": round(metrics.get("R2", 0), 4),
                "Akurasi ±1 Poin (%)": f"{metrics.get('Within_1_pt_acc', 0):.2f}%",
            })
        st.dataframe(pd.DataFrame(test_rows).sort_values("Test MAE"), use_container_width=True)

    st.divider()

    col_ev1, col_ev2 = st.columns(2)
    with col_ev1:
        st.subheader("2. Aktual vs Prediksi")
        st.image("figures/actual_vs_predicted.png", caption="Perbandingan Skor Sebenarnya vs Estimasi Model", use_container_width=True)
    with col_ev2:
        st.subheader("3. Diagnostik Residual")
        st.image("figures/residual_diagnostics.png", caption="Pola Sebaran Error dan Uji Normalitas Residual", use_container_width=True)

    st.divider()

    col_diag1, col_diag2 = st.columns([1.2, 1])
    with col_diag1:
        st.subheader("4. Analisis Error per Tingkat Kualitas")
        st.image("figures/error_by_quality.png", caption="Rata-rata Error Absolut pada Tiap Skor Kualitas", use_container_width=True)
    with col_diag2:
        st.subheader("5. Bukti Kebocoran Data (Data Leakage Impact)")
        if sens_data and "leakage_analysis" in sens_data:
            leak = sens_data["leakage_analysis"]
            st.metric("Test MAE (Group Split / Jujur)", f"{leak['group_split_no_leakage']['MAE']:.3f}", "Evaluasi Valid")
            st.metric("Test MAE (Random Split / Bocor)", f"{leak['random_split_with_leakage']['MAE']:.3f}", "-22.5% Error Semu (Kebocoran)", delta_color="inverse")
            st.metric("Baris Bocor pada Test Set", f"{leak['leaked_test_rows_count']} ({leak['leaked_test_rows_pct']:.1f}%)")
            st.caption("Pemisahan acak biasa menyebabkan 27.6% data uji identik dengan data latih, memicu overestimasi akurasi.")

# ==============================================================================
# HALAMAN 4: INTERPRETABILITAS & METODOLOGI
# ==============================================================================
elif page == "🔍 Interpretabilitas & Metodologi":
    st.title("🔍 Interpretabilitas Model & Metodologi Validasi")
    st.markdown(
        """
        Penjelasan faktor-faktor kimiawi penentu kualitas wine berdasarkan uji *Permutation Importance* 
        dan dokumentasi metodologis pencegahan bias.
        """
    )

    col_imp1, col_imp2 = st.columns([1.3, 1])

    with col_imp1:
        st.subheader("1. Permutation Feature Importance")
        st.image("figures/permutation_feature_importance.png", caption="Pengaruh Pengacakan Fitur terhadap Peningkatan MAE", use_container_width=True)

    with col_imp2:
        st.subheader("2. Interpretasi Temuan Kimiawi")
        st.markdown(
            """
            - **Alkohol (% vol):** Parameter paling dominan positif. Kadar alkohol lebih tinggi (11–13%) konsisten dengan skor kualitas lebih tinggi karena kematangan anggur dan fermentasi optimal.
            - **Volatile Acidity (Asam Asetat):** Parameter paling dominan negatif. Peningkatan asam volatil memicu aroma cuka yang tidak diinginkan dan menurunkan skor sensori drastis.
            - **Free & Total Sulfur Dioxide:** Berperan menjaga kesegaran wine dan mencegah oksidasi. Keseimbangan yang tepat penting untuk stabilitas sensori.
            - **Sulphates (Sulfat):** Bekerja sinergis dengan SO2 sebagai antimikroba alami, berkorelasi positif terutama pada wine merah.
            """
        )

    st.divider()

    st.subheader("3. Metodologi Ketat & Audit Data")
    st.markdown(
        """
        | Aspek Metodologi | Implementasi dalam Project | Tujuan Ilmiah |
        | :--- | :--- | :--- |
        | **Pencegahan Leakage** | `GroupShuffleSplit` berbasis hash 12 prediktor | Mencegah 1.177 baris duplikat terbagi silang ke train & test |
        | **Pipeline Utuh** | `ColumnTransformer` + `StandardScaler` + `OneHotEncoder` | Transformasi data hanya fit pada data latih di setiap lipatan CV |
        | **Metrik Primer** | MAE (Mean Absolute Error) & Akurasi $\\pm 1$ Poin | Relevan secara operasional karena skor kualitas berskala 0–10 |
        | **Uji Sensitivitas** | Komparasi group split vs random split & deduplikasi | Mengonfirmasi kekokohan model di berbagai skenario data |
        """
    )

    st.divider()

    st.subheader("4. Keterbatasan dan Rekomendasi Penggunaan")
    st.markdown(
        """
        1. **Domain Data Terbatas:** Model dilatih secara eksklusif pada varietas *Vinho Verde* Portugal. Penggunaan pada varietas atau terroir lain (seperti Bordeaux, Napa, dsb.) memerlukan kalibrasi ulang.
        2. **Keterbatasan Fisikokimiawi:** 11 parameter laboratorium tidak mencakup profil aromatik volatil (terpen, ester, pirazin), senyawa fenolik kompleks, atau teknik penuaan dalam tong kayu ek (*oak barrel aging*).
        3. **Peran Sommelier Manusia:** Model ini dirancang sebagai instrumen *screening* cepat laboratorium untuk pengendalian mutu (*quality control*), bukan sebagai pengganti evaluasi sensorik panel ahli manusia.
        """
    )

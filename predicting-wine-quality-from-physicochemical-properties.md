# Portfolio Project Data — Predicting Wine Quality from Physicochemical Properties

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/predicting_wine_quality_from/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image15.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "predicting-wine-quality-from-physicochemical-properties",
  "code": "PRO-23 // ML-06",
  "title": "Predicting Wine Quality from Physicochemical Properties",
  "category": "machine-learning",
  "categoryLabel": "Supervised Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Predicting-Wine-Quality-from-Physicochemical-Properties",
  "license": "MIT License",
  "banner": "/images/projects/predicting_wine_quality_from/banner.webp",
  "summary": "Leakage-free supervised regression predicting sensory wine quality from 11 physicochemical lab assays and color via Extra Trees and Streamlit.",
  "projectDescription": "Supervised tabular regression pipeline predicting Vinho Verde sensory wine quality from 11 laboratory physicochemical measurements and color. Uses group-aware splitting to eliminate 1,177 duplicate-row leaks, benchmarks seven models, and serves interactive predictions via Streamlit.",
  "architectureSummary": "Group-aware tabular regression pipeline using GroupShuffleSplit, ColumnTransformer, ExtraTreesRegressor, permutation diagnostics, and Streamlit UI.",
  "techStack": [
    "Python",
    "Scikit-Learn",
    "Pandas",
    "NumPy",
    "Matplotlib",
    "Seaborn",
    "Joblib",
    "Streamlit"
  ],
  "telemetryMatrix": {
    "projectType": "Supervised Tabular Regression Pipeline",
    "targetPlatform": "Python CLI & Streamlit Web Dashboard",
    "primaryDomain": "Enology & Physicochemical Analytics",
    "architecturePipeline": "Group Split -> ExtraTrees -> Permutation",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Tabular Regression & Analytics Pipeline",
    "learningParadigm": "Supervised Learning",
    "modelDomain": "Enology Physicochemical Quality",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "Leakage Free Splitting",
      "Extra Trees Ensemble",
      "Permutation Explainability"
    ]
  },
  "mlProblemContext": "Traditional wine quality grading relies on human sensory panels that are costly, slow, and prone to inter-panelist subjectivity. Routine laboratory chemical assays provide repeatable measurements, yet duplicate profiles in combined wine records cause severe leakage under naive random train-test splits.",
  "mlDatasetOverview": {
    "profileBadge": "UCI Vinho Verde",
    "source": "Cortez et al. (2009) / UCI ML Repository ID 186 (dataset/original/Wine_Quality.csv)",
    "modality": "Tabular",
    "scale": "6,497 Rows (4,898 White, 1,599 Red) / 5,320 Unique Profiles",
    "inputShape": "N x 12 Features (11 Float64 + Color)",
    "dataFormat": "CSV",
    "description": "Combined Portuguese Vinho Verde dataset spanning 11 continuous physicochemical measurements and wine color (white or red) with zero missing values, partitioned 80/20 via GroupShuffleSplit into 5,183 training rows and 1,314 independent held-out test rows.",
    "provenanceNote": "Public UCI Machine Learning Repository dataset (DOI 10.24432/C56S3T); actual CSV expands upon the 4,898-row white-wine scope noted in metadata.json."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "12 Physicochemical & Color Inputs",
    "spaceNote": "11 laboratory continuous assays plus categorical red/white variant indicator",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "quality",
      "type": "Ordinal Continuous Score",
      "distribution": "Integer scores 3 to 9 (mean 5.82, std 0.87; 76.56% concentrated in scores 5 and 6)"
    },
    "inputSignals": [
      {
        "name": "alcohol",
        "type": "Numerical",
        "importanceOrRole": "Rank #1 (+0.1860 MAE impact)",
        "description": "Alcohol volume percentage (8.00-14.90% vol), exhibiting the strongest positive correlation (r = +0.44) with sensory quality."
      },
      {
        "name": "volatile_acidity",
        "type": "Numerical",
        "importanceOrRole": "Rank #2 (+0.0730 MAE impact)",
        "description": "Acetic acid concentration (0.08-1.58 g/dm3) associated with undesirable vinegar aroma and strong negative quality correlation (r = -0.27)."
      },
      {
        "name": "free_sulfur_dioxide",
        "type": "Numerical",
        "importanceOrRole": "Rank #3 (+0.0414 MAE impact)",
        "description": "Unbound SO2 concentration (1.0-289.0 mg/dm3) protecting wine against oxidation and microbial spoilage."
      },
      {
        "name": "color",
        "type": "Categorical",
        "importanceOrRole": "Rank #4 (+0.0324 MAE impact)",
        "description": "Binary wine variant indicator (white 75.39%, red 24.61%) capturing distinct chemical baseline profiles."
      },
      {
        "name": "total_sulfur_dioxide",
        "type": "Numerical",
        "importanceOrRole": "Rank #5 (+0.0225 MAE impact)",
        "description": "Combined free and bound SO2 (6.0-440.0 mg/dm3), strongly collinear with free SO2 (r = +0.72)."
      },
      {
        "name": "sulphates",
        "type": "Numerical",
        "importanceOrRole": "Rank #6 (+0.0210 MAE impact)",
        "description": "Potassium sulphate additive concentration (0.22-2.00 g/dm3; skewness +1.80) supporting antimicrobial preservation."
      },
      {
        "name": "residual_sugar",
        "type": "Numerical",
        "importanceOrRole": "Rank #7 (+0.0177 MAE impact)",
        "description": "Post-fermentation sugar concentration (0.60-65.80 g/dm3) interacting non-linearly with acidity and density."
      },
      {
        "name": "density",
        "type": "Numerical",
        "importanceOrRole": "Rank #9 (+0.0138 MAE impact)",
        "description": "Specific gravity (0.9871-1.0390 g/cm3), inversely tied to alcohol (r = -0.69) and positively linked to residual sugar (r = +0.55)."
      }
    ],
    "dataCharacteristicsSummary": "All 13 columns contain zero missing values, while chlorides (+5.40), sulphates (+1.80), and fixed acidity (+1.72) show pronounced positive skew. Exactly 1,177 duplicate rows span 5,320 unique profiles with zero label conflicts across identical chemical records."
  },
  "mlLearningApproach": {
    "paradigm": "Supervised Tabular Regression",
    "methodologyDescription": "Formulates sensory quality estimation as continuous regression rather than discrete classification to preserve ordinal distance and penalize magnitude errors directly. Seven pipelines spanning dummy baselines, linear models, and tree ensembles were compared under 5-fold GroupKFold.",
    "justification": "Extra Trees Regressor handles skewed chemical distributions, multi-feature collinearity, and non-linear sensory interactions without requiring monotonic feature scaling, while randomized split thresholds reduce variance relative to standard bagging and linear models."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Data Audit & Duplicate Grouping",
      "description": "Verified zero null values across 6,497 rows and grouped identical 12-feature profiles via df.groupby(features).ngroup() into 5,320 unique groups with zero conflicting target labels.",
      "badge": "Integrity Audit"
    },
    {
      "step": "02",
      "title": "Leakage-Free Group Partitioning",
      "description": "Applied GroupShuffleSplit (80/20, random_state=42) on predictor group_id, producing 5,183 training rows (4,256 groups) and 1,314 test rows (1,064 groups) with zero group overlap.",
      "badge": "Zero Leakage"
    },
    {
      "step": "03",
      "title": "Model-Specific ColumnTransformer",
      "description": "Built separate pipelines applying StandardScaler plus dropped-first OneHotEncoder for linear models, and numeric passthrough plus full OneHotEncoder(handle_unknown='ignore') for tree ensembles.",
      "badge": "Fold Safe"
    }
  ],
  "mlDataPreparationNote": "All ColumnTransformer scalers and encoders are fit exclusively on training partitions inside each 5-fold GroupKFold split and the final training set, eliminating train-test leakage.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "Extra Trees Regressor",
    "architectureSummary": "Scikit-learn Pipeline combining a ColumnTransformer preprocessor (numeric passthrough and OneHotEncoder for wine color) with an Extremely Randomized Trees ensemble of 250 unpruned-up-to-depth-24 randomized decision trees.",
    "configurationBadge": "250 Randomized Trees",
    "hyperparameters": [
      {
        "name": "n_estimators",
        "value": "250",
        "description": "Number of randomized decision trees averaged in the ensemble"
      },
      {
        "name": "max_depth",
        "value": "24",
        "description": "Maximum tree depth balancing non-linear capacity and variance control"
      },
      {
        "name": "min_samples_split",
        "value": "2",
        "description": "Minimum samples required to split an internal decision node"
      },
      {
        "name": "random_state",
        "value": "42",
        "description": "Fixed seed ensuring reproducible threshold sampling and tree splits"
      }
    ],
    "objectiveCriterion": "Mean Squared Error (L2 Split Criterion)",
    "optimizationSolver": "Randomized Threshold Tree Splitting",
    "trainingProcess": "All seven candidate models were cross-validated on the 5,183-row training partition using 5-fold GroupKFold. The selected Extra Trees configuration (250 estimators, depth 24, min split 2) was fit once on the full training split and serialized via Joblib."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "5-Fold GroupKFold & 20% GroupShuffleSplit Holdout",
    "description": "Model selection relied strictly on 5-fold GroupKFold cross-validation over training groups before single-pass evaluation on 1,314 unseen test rows, complemented by color-stratified checks and random-split leakage sensitivity analysis.",
    "metrics": [
      {
        "label": "TEST MAE",
        "value": "0.5163",
        "benchmark": "Baseline < 0.6492"
      },
      {
        "label": "TEST RMSE",
        "value": "0.6765",
        "benchmark": "Baseline < 0.8900"
      },
      {
        "label": "TEST R2",
        "value": "0.4198",
        "benchmark": "Target > 0.40"
      },
      {
        "label": "WITHIN 1 PT",
        "value": "88.81%",
        "benchmark": "Target > 85.0%"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Interactive Prediction Interface (Low-Quality Red Wine)",
      "badge": "Streamlit UI",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image01.webp",
      "telemetryNote": "11 lab sliders + color selector configured for low-quality preset (volatile acidity 0.88, alcohol 9.20)"
    },
    {
      "title": "Prediction Output & Input vs. Median Profile",
      "badge": "Inference Output",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image02.webp",
      "telemetryNote": "Predicted continuous score 4.72 (interval [4.20 - 5.24]) with +203.4% volatile acidity deviation"
    },
    {
      "title": "Held-Out Test Evaluation Metrics Tab",
      "badge": "Benchmark View",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image03.webp",
      "telemetryNote": "In-app metrics card displaying Test MAE 0.516, RMSE 0.676, Within +/-1 Acc 88.81%, and R2 0.420"
    },
    {
      "title": "Model Architecture & Methodology Summary Tab",
      "badge": "System Metadata",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image04.webp",
      "telemetryNote": "Documents Extra Trees 250-tree configuration, GroupShuffleSplit 80/20 isolation, and UCI provenance"
    },
    {
      "title": "Exploratory Data Analysis & Variant Distribution Page",
      "badge": "EDA Dashboard",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image05.webp",
      "telemetryNote": "Shows central concentration in scores 5-6 (76.6%) and red vs. white wine variant comparisons"
    },
    {
      "title": "Model Evaluation & Diagnostics Dashboard Page",
      "badge": "Model Benchmark",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image06.webp",
      "telemetryNote": "Interactive Streamlit benchmark ranking all seven regressors by held-out Test MAE and RMSE"
    },
    {
      "title": "Interpretability & Leakage Audit Page",
      "badge": "Audit & XAI",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image07.webp",
      "telemetryNote": "Combines permutation feature importance rankings with chemical interpretation and GroupShuffleSplit audit"
    },
    {
      "title": "Actual vs. Predicted Scatter & Quality Error Breakdown",
      "badge": "Error Analysis",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image08.webp",
      "telemetryNote": "88.8% within +/-1 point overall; low error on scores 5-7 (0.36-0.61 MAE), higher bias on extremes 3 and 8"
    },
    {
      "title": "Residual Distribution & Residual vs. Predicted Plots",
      "badge": "Residual Check",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image09.webp",
      "telemetryNote": "Symmetric zero-centered residual histogram (mean 0.010) with discrete-target diagonal bands"
    },
    {
      "title": "Physicochemical Feature Density Distributions",
      "badge": "Feature EDA",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image10.webp",
      "telemetryNote": "11-panel KDE and histogram grid highlighting right skew in chlorides (+5.40) and sulphates (+1.80)"
    },
    {
      "title": "Key Physicochemical Features vs. Quality Boxplots",
      "badge": "Trend Analysis",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image11.webp",
      "telemetryNote": "Monotonic quality increase with alcohol alongside decreasing volatile acidity and density"
    },
    {
      "title": "Permutation Feature Importance (Held-Out Test Set)",
      "badge": "Feature Importance",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image12.webp",
      "telemetryNote": "15-repeat Neg-MAE permutation led by alcohol (+0.1860) and volatile acidity (+0.0730)"
    },
    {
      "title": "Quality Distribution & Red vs. White Comparison",
      "badge": "Target Balance",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image13.webp",
      "telemetryNote": "Red wine mean 5.64 (n=1,599) vs. white wine mean 5.88 (n=4,898) across 6,497 total samples"
    },
    {
      "title": "5-Fold GroupKFold CV vs. Test Model Comparison",
      "badge": "Model Selection",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image14.webp",
      "telemetryNote": "Consistent generalization from CV (MAE 0.525) to Test (MAE 0.516) outperforming linear and boosting models"
    },
    {
      "title": "Pearson Correlation Matrix Heatmap",
      "badge": "Collinearity Map",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image15.webp",
      "telemetryNote": "Quality correlates +0.44 with alcohol and -0.31 with density; density correlates -0.69 with alcohol"
    }
  ],
  "mlVisualizationTelemetryNote": "Extra Trees generalizes consistently from 5-fold GroupKFold CV (MAE 0.5251, RMSE 0.6829) to the held-out test set (MAE 0.5163, RMSE 0.6765), whereas naive random splitting falsely lowers MAE to 0.4016 via 27.6% duplicate leakage.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Permutation importance on the held-out test set identifies alcohol, volatile acidity, free sulfur dioxide, and wine color as primary predictive signals. Ensemble predictions align closely with sensory ratings across core market tiers while compressing toward the mean on rare extreme grades.",
    "keyDrivers": [
      {
        "factor": "Alcohol (% vol)",
        "impact": "+0.1860 +/- 0.0077 MAE degradation upon permutation; strongest positive predictor (r = +0.44)"
      },
      {
        "factor": "Volatile Acidity",
        "impact": "+0.0730 +/- 0.0042 MAE degradation; elevated acetic acid sharply lowers predicted sensory score"
      },
      {
        "factor": "Free Sulfur Dioxide",
        "impact": "+0.0414 +/- 0.0027 MAE degradation; non-linear oxidation protection optimum"
      },
      {
        "factor": "Wine Color Variant",
        "impact": "+0.0324 +/- 0.0018 MAE degradation; captures red vs. white baseline shifts (MAE 0.5069 vs. 0.5190)"
      }
    ],
    "limitations": [
      {
        "title": "Extreme Quality Score Compression",
        "description": "Rare classes score 3 (n=30) and score 9 (n=5) cause regression-to-the-mean bias on test extremes (score 3 MAE 2.40, score 8 MAE 1.48)."
      },
      {
        "title": "Unmeasured Aromatic & Vinification Factors",
        "description": "Lab assays exclude grape cultivar, vintage year, oak barrel aging, and complex phenolic or ester aromatics driving ~58% unexplained variance."
      },
      {
        "title": "Vinho Verde Regional Specificity",
        "description": "All 6,497 records originate from northern Portugal's Vinho Verde region, requiring domain recalibration before applying to external terroirs."
      }
    ],
    "futureImprovements": [
      {
        "title": "Sample-Weighted or Ordinal Loss Calibration",
        "target": "Reduce MAE compression bias on underrepresented quality tiers (scores 3, 4, and 8)"
      },
      {
        "title": "Enriched Viticultural & Barrel Metadata",
        "target": "Increase explained variance (R2) beyond 0.42 by integrating cultivar, vintage, and aging features"
      },
      {
        "title": "Multi-Region Domain Validation",
        "target": "Benchmark calibration stability across non-Vinho Verde red and white wine appellations"
      }
    ],
    "discoveryNote": "Permutation importance quantifies predictive reliance rather than chemical causality; artificially spiking ethanol will not reproduce holistic sensory balance."
  },
  "mlModelArtifacts": [
    {
      "name": "best_wine_quality_pipeline.joblib",
      "format": "JOBLIB"
    },
    {
      "name": "evaluation_results.json",
      "format": "JSON"
    },
    {
      "name": "sensitivity_results.json",
      "format": "JSON"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Supervised Learning",
    "modelArchitecture": "Extra Trees Regressor",
    "coreFramework": "Scikit-Learn",
    "validationMethod": "5-Fold GroupKFold CV"
  }
}
```

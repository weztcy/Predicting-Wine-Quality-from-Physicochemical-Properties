# Portfolio Project Data — Predicting Wine Quality from Physicochemical Properties

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/predicting_wine_quality_from/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image15.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "predicting-wine-quality-from-physicochemical-properties",
  "code": "PRO-21 // ML-06",
  "title": "Predicting Wine Quality from Physicochemical Properties",
  "category": "machine-learning",
  "categoryLabel": "Supervised Learning",
  "year": "2023",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Predicting-Wine-Quality-from-Physicochemical-Properties",
  "license": "MIT License",
  "banner": "/images/projects/predicting_wine_quality_from/banner.webp",
  "summary": "Leakage-free supervised regression pipeline predicting Vinho Verde sensory wine quality from 11 physicochemical lab properties and wine color.",
  "projectDescription": "Built a leakage-free supervised regression pipeline on 6,497 Vinho Verde wine samples using group-aware splitting, 5-fold GroupKFold benchmarking across seven models, permutation explainability, and an interactive Streamlit dashboard achieving 88.81% within-one-point accuracy.",
  "architectureSummary": "CSV Ingestion -> Feature-Group Hashing -> GroupShuffleSplit (80/20) -> 5-Fold GroupKFold & ColumnTransformer -> Tuned Extra Trees -> Joblib Artifact -> Streamlit App",
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
    "projectType": "Supervised Tabular Regression",
    "targetPlatform": "Python CLI & Streamlit Web Dashboard",
    "primaryDomain": "Oenology & Physicochemical Analytics",
    "architecturePipeline": "GroupSplit -> ColumnTransformer -> ExtraTrees",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2023"
  },
  "mlProjectInfo": {
    "projectType": "Supervised Regression Pipeline",
    "learningParadigm": "Supervised Learning",
    "modelDomain": "Physicochemical Wine Quality Analytics",
    "projectStatus": "Completed",
    "developmentYear": "2023",
    "capabilityBadges": [
      "Leakage-Free Group Splitting",
      "Extra Trees Ensemble",
      "Streamlit Analytics UI"
    ]
  },
  "mlProblemContext": "Sensory wine grading by sommelier panels is costly, slow, and prone to human subjectivity and fatigue. Automating quality estimation from 11 objective lab measurements requires preventing severe data leakage from 1,177 duplicate rows while capturing non-linear chemical interactions.",
  "mlDatasetOverview": {
    "profileBadge": "6,497 Vinho Verde Samples",
    "source": "UCI Wine Quality Dataset (Cortez et al., 2009)",
    "modality": "Tabular",
    "scale": "6,497 Rows (4,898 White, 1,599 Red)",
    "inputShape": "N x 12 Features (11 Numeric + Color)",
    "dataFormat": "CSV & JSON",
    "description": "Contains 6,497 Portuguese Vinho Verde records across 5,320 unique chemical profiles with zero missing values, partitioned via GroupShuffleSplit into 5,183 training rows (4,256 groups) and 1,314 held-out test rows (1,064 groups).",
    "provenanceNote": "UCI ML Repository ID 186 (DOI: 10.24432/C56S3T); CSV ground truth retains both white and red variants with color feature."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "12 Predictors / Ordinal Target",
    "spaceNote": "11 continuous float64 lab assays plus categorical wine color with zero null entries.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "quality",
      "type": "Ordinal Discrete Score (Regression)",
      "distribution": "Range 3-9; mean 5.82 (std 0.87); 76.56% concentrated in scores 5 and 6"
    },
    "inputSignals": [
      {
        "name": "alcohol",
        "type": "Numerical",
        "importanceOrRole": "Rank #1 (0.1860 MAE Drop)",
        "description": "Alcohol percentage by volume (% vol, 8.00-14.90); strongest positive linear and non-linear predictor of sensory quality."
      },
      {
        "name": "volatile_acidity",
        "type": "Numerical",
        "importanceOrRole": "Rank #2 (0.0730 MAE Drop)",
        "description": "Acetic acid concentration (g/dm3, 0.08-1.58); excess levels impart undesirable vinegar aroma and lower quality scores."
      },
      {
        "name": "free_sulfur_dioxide",
        "type": "Numerical",
        "importanceOrRole": "Rank #3 (0.0414 MAE Drop)",
        "description": "Free SO2 (mg/dm3, 1.0-289.0) acting as an antimicrobial and antioxidant stabilizer with non-linear optimal thresholds."
      },
      {
        "name": "color",
        "type": "Categorical",
        "importanceOrRole": "Rank #4 (0.0324 MAE Drop)",
        "description": "Binary wine variant (white: 75.39%, red: 24.61%) capturing baseline chemical matrix differences."
      },
      {
        "name": "total_sulfur_dioxide",
        "type": "Numerical",
        "importanceOrRole": "Rank #5 (0.0225 MAE Drop)",
        "description": "Combined free and bound SO2 (mg/dm3, 6.0-440.0), strongly correlated with free SO2 (r = +0.72)."
      },
      {
        "name": "sulphates",
        "type": "Numerical",
        "importanceOrRole": "Rank #6 (0.0210 MAE Drop)",
        "description": "Potassium sulphate additive (g/dm3, 0.22-2.00) supporting microbial stability, especially in red wine."
      }
    ],
    "dataCharacteristicsSummary": "Exploratory statistics reveal strong right-skewness in chlorides (+5.40), sulphates (+1.80), and fixed acidity (+1.72), alongside collinearity between density and alcohol (r = -0.69). Extreme tail scores 3 (0.46%) and 9 (0.08%) are scarce compared to central scores 5 and 6."
  },
  "mlLearningApproach": {
    "paradigm": "Supervised Tabular Regression with Group-Aware Validation",
    "methodologyDescription": "Quality is modeled as continuous regression rather than multi-class classification to preserve ordinal distance and penalize error magnitude directly. Seven pipelines spanning dummy baselines, linear/ridge models, bagging forests, and boosting are benchmarked under 5-fold GroupKFold.",
    "justification": "Non-linear tree ensembles handle heavily skewed physicochemical distributions and cross-color interactions without distribution normality assumptions, while Extra Trees randomized split thresholds further reduce variance on noisy sensory labels."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Data Audit & Duplicate Profiling",
      "description": "Verified zero missing values across 13 columns, identified 1,177 exact duplicate rows spanning 5,320 unique chemical profiles, and confirmed zero conflicting quality labels within identical groups.",
      "badge": "Data Audit"
    },
    {
      "step": "02",
      "title": "Group-Aware 80/20 Splitting",
      "description": "Assigned group_id via groupby across all 12 predictor columns and applied GroupShuffleSplit (test_size=0.20, seed=42), isolating 5,183 train rows and 1,314 held-out test rows with zero group overlap.",
      "badge": "Zero Leakage"
    },
    {
      "step": "03",
      "title": "Fold-Isolated ColumnTransformer",
      "description": "Built distinct preprocessing pipelines applying StandardScaler and OneHotEncoder(drop='first') for linear models while passing raw numeric scales with OneHotEncoder for tree-based ensembles.",
      "badge": "ColumnTransformer"
    }
  ],
  "mlDataPreparationNote": "All transformers fit strictly inside each training fold and on the training partition alone; sensitivity testing confirms naive random splitting leaks 27.6% of test rows from training profiles.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "Extra Trees Regressor",
    "architectureSummary": "Ensemble of 250 extremely randomized regression trees (max_depth=24, min_samples_split=2) averaging predictions across randomized feature split thresholds on one-hot encoded color and raw physicochemical features.",
    "configurationBadge": "250 Randomized Trees",
    "hyperparameters": [
      {
        "name": "n_estimators",
        "value": "250",
        "description": "Number of randomized decision trees averaged in the ensemble."
      },
      {
        "name": "max_depth",
        "value": "24",
        "description": "Maximum tree depth balancing non-linear expressiveness and overfitting control."
      },
      {
        "name": "min_samples_split",
        "value": "2",
        "description": "Minimum samples required to split an internal decision node."
      },
      {
        "name": "random_state",
        "value": "42",
        "description": "Fixed random seed ensuring reproducible splits, tree construction, and permutation tests."
      }
    ],
    "objectiveCriterion": "Mean Squared Error (Squared Error Split Criterion)",
    "optimizationSolver": "Randomized Recursive Binary Partitioning",
    "trainingProcess": "All seven candidate pipelines were first benchmarked via 5-fold GroupKFold cross-validation on the 5,183-row training set. Tuned Extra Trees parameters (250 trees, depth 24, split 2) were then fitted on the full training partition and locked prior to single-pass held-out test evaluation."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "5-Fold GroupKFold CV & 20% Held-Out GroupShuffleSplit Test",
    "description": "Model selection relied exclusively on 5-fold GroupKFold CV over the 5,183 training rows, followed by final verification on 1,314 unseen test rows stratified across red wine, white wine, and individual quality scores.",
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
        "benchmark": "Baseline > 0.3181"
      },
      {
        "label": "WITHIN 1-PT",
        "value": "88.81%",
        "benchmark": "Target > 85.00%"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Streamlit Predictor Input Parameter Panel",
      "badge": "Inference UI",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image01.webp",
      "telemetryNote": "Two-column interactive slider interface for 11 physicochemical assays and wine color presets."
    },
    {
      "title": "Prediction Output & Median Deviation Profile",
      "badge": "Inference Output",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image02.webp",
      "telemetryNote": "Displays continuous score 4.72, rounded Skor 5, 88.81% ±1-pt interval, and median deviation table."
    },
    {
      "title": "Streamlit Model Evaluation Metrics Tab",
      "badge": "Metrics Card",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image03.webp",
      "telemetryNote": "Surfaces held-out Test MAE (0.516), RMSE (0.676), Within ±1 Point (88.81%), and R2 (0.420)."
    },
    {
      "title": "Streamlit Methodology & Provenance Tab",
      "badge": "Documentation UI",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image04.webp",
      "telemetryNote": "Summarizes Extra Trees hyperparameters, GroupShuffleSplit leakage guard, and UCI Cortez et al. source."
    },
    {
      "title": "Streamlit Exploratory Data Analysis Dashboard",
      "badge": "EDA Dashboard",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image05.webp",
      "telemetryNote": "Combines overall sensory score distribution with red-vs-white wine comparative analytics."
    },
    {
      "title": "Streamlit Model Evaluation & Diagnostics Page",
      "badge": "Benchmark UI",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image06.webp",
      "telemetryNote": "Pairs 7-model MAE/RMSE bar comparisons with a sorted held-out performance summary table."
    },
    {
      "title": "Streamlit Interpretability & Audit Methodology Page",
      "badge": "Explainability UI",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image07.webp",
      "telemetryNote": "Integrates permutation importance chart with chemical findings and anti-leakage audit summary."
    },
    {
      "title": "Actual vs. Predicted & Per-Quality Error Analysis",
      "badge": "Error Analysis",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image08.webp",
      "telemetryNote": "88.8% of predictions fall inside ±1.0 tolerance; MAE is lowest on score 6 (0.36) and highest on score 3 (2.40)."
    },
    {
      "title": "Residual Distribution & Residuals vs. Predicted",
      "badge": "Residual Diagnostics",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image09.webp",
      "telemetryNote": "Symmetric zero-centered residual histogram (mean 0.010) confirming unbiased central-score calibration."
    },
    {
      "title": "Density Distributions of 11 Physicochemical Features",
      "badge": "Feature EDA",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image10.webp",
      "telemetryNote": "Highlights strong right skew in chlorides (+5.40), sulphates (+1.80), and fixed acidity (+1.72)."
    },
    {
      "title": "Key Physicochemical Relationships vs. Quality",
      "badge": "Bivariate Boxplots",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image11.webp",
      "telemetryNote": "Shows monotonic alcohol increase and volatile acidity decrease from low to high quality scores."
    },
    {
      "title": "Permutation Feature Importance (Held-Out Test)",
      "badge": "Permutation Importance",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image12.webp",
      "telemetryNote": "Alcohol (0.1860), volatile acidity (0.0730), free SO2 (0.0414), and color (0.0324) lead MAE impact."
    },
    {
      "title": "Quality Distribution by Color & Target Histogram",
      "badge": "Variant Analysis",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image13.webp",
      "telemetryNote": "White wine averages 5.88 vs. 5.64 for red wine; scores 5 and 6 represent 76.6% of all records."
    },
    {
      "title": "7-Model Regression Benchmark (CV vs. Test)",
      "badge": "Model Comparison",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image14.webp",
      "telemetryNote": "Extra Trees outperforms Random Forest, HistGradientBoosting, Ridge, Linear, and Dummy baselines."
    },
    {
      "title": "Pearson Correlation Heatmap of Lab Features",
      "badge": "Correlation Matrix",
      "imageUrl": "/images/projects/predicting_wine_quality_from/image15.webp",
      "telemetryNote": "Shows alcohol (+0.44) and density (-0.31) correlations with quality plus free/total SO2 collinearity (+0.72)."
    }
  ],
  "mlVisualizationTelemetryNote": "5-fold GroupKFold CV metrics (MAE 0.5251, RMSE 0.6829) align closely with held-out test metrics (MAE 0.5163, RMSE 0.6765), confirming zero duplicate leakage.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Held-out permutation importance (15 repeats) shows the Extra Trees model relies predominantly on alcohol, volatile acidity, free sulfur dioxide, and wine color to separate quality tiers. These patterns reflect statistical predictive associations rather than direct causal winemaking prescriptions.",
    "keyDrivers": [
      {
        "factor": "Alcohol (% vol)",
        "impact": "Top positive driver (+0.1860 MAE increase when permuted; Pearson r = +0.44 with quality)"
      },
      {
        "factor": "Volatile Acidity",
        "impact": "Primary negative driver (+0.0730 MAE increase when permuted; excess acetic acid degrades sensory score)"
      },
      {
        "factor": "Free Sulfur Dioxide",
        "impact": "Third key predictor (+0.0414 MAE increase; non-linear oxidation preservation balance)"
      },
      {
        "factor": "Wine Color",
        "impact": "Fourth predictor (+0.0324 MAE increase; differentiates red and white baseline chemistry)"
      }
    ],
    "limitations": [
      {
        "title": "Vinho Verde Regional Scope",
        "description": "Trained solely on northern Portuguese Vinho Verde samples; generalization to other grape cultivars, terroirs, or vintages requires recalibration."
      },
      {
        "title": "Extreme Score Compression Bias",
        "description": "Severe scarcity of scores 3 (n=30) and 9 (n=5) causes regression toward the mean, producing +2.40 bias on score 3 and -1.48 bias on score 8."
      },
      {
        "title": "Unobserved Sensory & Enological Factors",
        "description": "Excludes volatile aromatic esters, phenolic tannins, oak barrel aging, grape variety, and subjective human panelist variance."
      }
    ],
    "futureImprovements": [
      {
        "title": "Conformal Prediction Intervals",
        "target": "Deliver calibrated sample-specific uncertainty intervals alongside point estimates"
      },
      {
        "title": "Ordinal & Cost-Sensitive Objectives",
        "target": "Reduce prediction compression bias on tail quality scores 3, 4, 8, and 9"
      },
      {
        "title": "Expanded Enological Metadata",
        "target": "Incorporate grape cultivar, vintage year, and phenolic assays to lift R2 beyond 0.42"
      }
    ],
    "discoveryNote": "Equally consistent test MAE across red wine (0.5069) and white wine (0.5190) validates retaining the unified multi-color dataset."
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
    "validationMethod": "5-Fold GroupKFold"
  }
}
```

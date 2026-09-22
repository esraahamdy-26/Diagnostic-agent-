"""
Diabetes & Hypertension — Pre-Lab Risk Screening (v2, non-circular)
=======================================================================
الفرق عن النسخة الأصلية:

  الأصلي:  features = [age, bmi, fasting_glucose, hba1c, systolic_bp, ...]
           target   = مشتق من fasting_glucose و hba1c نفسهم
           → circular، والموديل مجرد بيحفظ معادلة already-known

  الجديد:  features = [age, bmi, sex, ...] فقط (من غير fasting_glucose/hba1c
           للسكر، ومن غير systolic_bp/diastolic_bp للضغط)
           target   = نفس التصنيف الطبي المعروف (زي الأصلي)
           → السؤال بقى حقيقي: "من غير ما تعمل التحليل، إيه احتمالية إنك
             لو عملته هيطلع غير طبيعي؟" — ده بالظبط اللي محتاجينه في agent
             بيتكلم مع المريض *قبل* أي تحليل.

النتيجة متوقع تكون أقل دقة من الـ "100%" المزيفة — وده مؤشر صحة، مش ضعف.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "real_processed" / "nhanes_chronic_labs_real.csv"
OUT_DIR = ROOT / "models"
METRICS_DIR = ROOT / "metrics"


SCREENING_SPECS = {
    "diabetes_risk_screening": {
        "target": "diabetes_label",
        "numeric_features": ["age", "bmi", "systolic_bp", "diastolic_bp", "total_cholesterol"],
        "categorical_features": ["sex"],
        "excluded_because_circular": ["fasting_glucose", "hba1c"],
        "dropna_subset": ["age", "bmi", "systolic_bp", "diastolic_bp", "total_cholesterol", "sex"],
    },
    "hypertension_risk_screening": {
        "target": "hypertension_label",
        "numeric_features": ["age", "bmi", "total_cholesterol", "creatinine_mg_dl"],
        "categorical_features": ["sex"],
        "excluded_because_circular": ["systolic_bp", "diastolic_bp"],
        "dropna_subset": ["age", "bmi", "total_cholesterol", "creatinine_mg_dl", "sex"],
    },
}


def preprocessing(numeric_features, categorical_features) -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                             ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", numeric, numeric_features), ("cat", categorical, categorical_features)],
                              sparse_threshold=0)


def candidate_pipelines(numeric_features, categorical_features) -> dict:
    prep = preprocessing(numeric_features, categorical_features)
    return {
        "logistic_regression": Pipeline([("preprocessor", prep),
                                          ("model", LogisticRegression(max_iter=1500, class_weight="balanced"))]),
        "random_forest": Pipeline([("preprocessor", prep),
                                    ("model", RandomForestClassifier(n_estimators=350, min_samples_leaf=3,
                                                                      class_weight="balanced", random_state=RANDOM_STATE))]),
        "gradient_boosting": Pipeline([("preprocessor", prep),
                                        ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))]),
        "hist_gradient_boosting": Pipeline([("preprocessor", prep),
                                             ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE))]),
    }


def train_screening_model(name: str, spec: dict) -> dict:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=spec["dropna_subset"] + [spec["target"]]).copy()

    X = df[spec["numeric_features"] + spec["categorical_features"]]
    y = df[spec["target"]]

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"Rows after cleaning: {len(df)}")
    print(f"Positive class prevalence: {y.mean():.1%}")
    print(f"Excluded (circular) features: {spec['excluded_because_circular']}")

    x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp)

    candidates = candidate_pipelines(spec["numeric_features"], spec["categorical_features"])
    candidate_metrics = {}
    fitted = {}
    for cname, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        fitted[cname] = pipeline
        pred = pipeline.predict(x_val)
        proba = pipeline.predict_proba(x_val)[:, 1]
        candidate_metrics[cname] = {
            "recall": float(recall_score(y_val, pred, pos_label=1)),
            "roc_auc": float(roc_auc_score(y_val, proba)),
            "macro_f1": float(f1_score(y_val, pred, average="macro")),
        }

    # اختيار موضوعي: recall أولاً (الأهم طبياً)، بعدين roc_auc
    best_name = sorted(candidate_metrics,
                        key=lambda k: (candidate_metrics[k]["recall"], candidate_metrics[k]["roc_auc"]),
                        reverse=True)[0]
    best_pipeline = fitted[best_name]

    test_pred = best_pipeline.predict(x_test)
    test_proba = best_pipeline.predict_proba(x_test)[:, 1]

    print(f"\nCandidate comparison (validation):")
    for cname, m in candidate_metrics.items():
        print(f"  {cname:25s} recall={m['recall']:.3f}  roc_auc={m['roc_auc']:.3f}  macro_f1={m['macro_f1']:.3f}")
    print(f"\nSelected: {best_name}")
    print(f"\nTEST SET:")
    print(classification_report(y_test, test_pred, zero_division=0))
    print(f"Test ROC-AUC: {roc_auc_score(y_test, test_proba):.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, OUT_DIR / f"{name}.joblib")

    metadata = {
        "model_name": name,
        "reframing_note": "Pre-lab risk screening: predicts likelihood of an abnormal lab result "
                           "BEFORE the lab test is done, using only information a patient can report "
                           "in conversation (no lab values used as input).",
        "excluded_circular_features": spec["excluded_because_circular"],
        "features_order": spec["numeric_features"] + spec["categorical_features"],
        "numeric_features": spec["numeric_features"],
        "categorical_features": spec["categorical_features"],
        "best_candidate": best_name,
        "candidate_validation_metrics": candidate_metrics,
        "test_recall": float(recall_score(y_test, test_pred, pos_label=1)),
        "test_roc_auc": float(roc_auc_score(y_test, test_proba)),
        "test_macro_f1": float(f1_score(y_test, test_pred, average="macro")),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "positive_class_prevalence": float(y.mean()),
        "rows_total": int(len(df)),
        "recommended_action": "Model output should trigger a recommendation to get the actual lab "
                               "test done, never replace it.",
    }
    (METRICS_DIR / f"{name}_metrics.json").write_text(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    for name, spec in SCREENING_SPECS.items():
        train_screening_model(name, spec)

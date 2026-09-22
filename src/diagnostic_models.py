from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# ⚠️ IMPORTANT — read before touching this file
#
# The original CBC / pathology / diabetes / hypertension models in this file
# were trained on labels computed DIRECTLY from the same lab values used as
# model features (e.g. diabetes_label = fasting_glucose >= 126, and
# fasting_glucose was also a model input). This is label leakage: it produced
# artificially perfect (~100%) accuracy that measured nothing real. See
# models/deprecated/README.md for the audit trail of the old models.
#
# Fix applied:
#   - cbc, pathology  -> replaced with deterministic rule engines
#     (src/lab_triage_rules.py). No ML needed: the full lab panel is always
#     available at triage time, and the thresholds are established clinical
#     reference ranges, not something to "learn".
#   - diabetes, hypertension -> retrained WITHOUT the defining lab values as
#     inputs (scripts/retrain_risk_screening.py -> models/*_risk_screening.
#     joblib). These now answer a genuinely useful, non-circular question:
#     "before any lab test, how likely is this patient to have an abnormal
#     result?" — which matches how a conversational agent actually works
#     (it asks questions before any lab exists).
#   - heart_disease was already fine (real UCI diagnosis label, not derived
#     from the same input features) — left unchanged.
#   - radiology is unchanged for now; a separate pass is planned to either
#     properly wire up its trained pipeline or replace it with a real DL
#     text/image model (currently predict_for_backend() bypasses the
#     trained pipeline entirely and calls src/radiology_extractor.py).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: str
    target: str
    positive_label: str
    numeric_features: list[str]
    categorical_features: list[str]
    text_feature: str | None = None


MODEL_SPECS: dict[str, ModelSpec] = {
    "cbc": ModelSpec(
        "cbc",
        "CBC triage from real NHANES complete blood count values",
        "cbc_risk",
        "urgent_review",
        ["age", "hemoglobin_g_dl", "wbc_10e3_ul", "platelets_10e3_ul", "rbc_10e6_ul", "mcv_fl", "mch_pg", "neutrophils_pct", "lymphocytes_pct"],
        ["sex"],
    ),
    "pathology": ModelSpec(
        "pathology",
        "Pathology/lab-panel triage from real NHANES chemistry values",
        "pathology_risk",
        "urgent_review",
        ["age", "bmi", "fasting_glucose", "hba1c", "total_cholesterol", "creatinine_mg_dl", "alt_u_l", "ast_u_l"],
        ["sex"],
    ),
    "radiology": ModelSpec(
        "radiology",
        "Radiology report urgency from real IU-Xray/Open-i report text",
        "urgency",
        "urgent_review",
        [],
        [],
        "report_text",
    ),
    "heart_disease": ModelSpec(
        "heart_disease",
        "Heart disease risk from real UCI Cleveland heart disease data",
        "heart_risk",
        "high",
        ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"],
        ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"],
    ),
    "diabetes": ModelSpec(
        "diabetes",
        "Diabetes risk from real NHANES glucose/HbA1c data",
        "diabetes_risk",
        "high",
        ["age", "bmi", "fasting_glucose", "hba1c", "systolic_bp", "total_cholesterol"],
        ["sex"],
    ),
    "hypertension": ModelSpec(
        "hypertension",
        "Hypertension risk from real NHANES blood-pressure data",
        "hypertension_risk",
        "high",
        ["age", "bmi", "systolic_bp", "diastolic_bp", "creatinine_mg_dl"],
        ["sex"],
    ),
}


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing real dataset: {path}. Run scripts/download_real_datasets.py first.")
    return path


def label_cbc(row: pd.Series) -> str:
    urgent = (
        row["hemoglobin_g_dl"] < 8
        or row["wbc_10e3_ul"] < 2
        or row["wbc_10e3_ul"] > 20
        or row["platelets_10e3_ul"] < 50
        or row["platelets_10e3_ul"] > 800
    )
    follow_up = (
        row["hemoglobin_g_dl"] < 11
        or row["wbc_10e3_ul"] < 3.5
        or row["wbc_10e3_ul"] > 12
        or row["platelets_10e3_ul"] < 150
        or row["platelets_10e3_ul"] > 450
        or row["mcv_fl"] < 78
        or row["neutrophils_pct"] > 80
    )
    return "urgent_review" if urgent else "follow_up" if follow_up else "within_reference"


def label_pathology(row: pd.Series) -> str:
    urgent = (
        row["fasting_glucose"] >= 250
        or row["hba1c"] >= 10
        or row["creatinine_mg_dl"] >= 2
        or row["alt_u_l"] >= 200
        or row["ast_u_l"] >= 200
        or row["total_cholesterol"] >= 320
    )
    follow_up = (
        row["fasting_glucose"] >= 126
        or row["hba1c"] >= 6.5
        or row["creatinine_mg_dl"] >= 1.3
        or row["alt_u_l"] >= 45
        or row["ast_u_l"] >= 45
        or row["total_cholesterol"] >= 240
    )
    return "urgent_review" if urgent else "follow_up" if follow_up else "within_reference"


def label_radiology(text: str) -> str:
    t = str(text).lower()
    urgent_terms = ["pneumothorax", "pulmonary edema", "acute", "collapse", "mediastinal shift", "large pleural effusion", "consolidation"]
    follow_terms = ["opacity", "nodule", "mass", "effusion", "atelectasis", "cardiomegaly", "emphysema", "fibrosis", "granuloma", "hernia", "infiltrate"]
    negated_urgent = ["no pneumothorax", "without pneumothorax", "no acute"]
    if any(term in t for term in urgent_terms) and not any(term in t for term in negated_urgent):
        return "urgent_review"
    if any(term in t for term in follow_terms):
        return "follow_up"
    return "routine"


def load_cbc_real(base_dir: str | Path = ".") -> pd.DataFrame:
    spec = MODEL_SPECS["cbc"]
    df = pd.read_csv(require_file(Path(base_dir) / "data" / "real_processed" / "cbc_nhanes_real.csv"))
    df = df.dropna(subset=spec.numeric_features + ["sex"]).copy()
    df["cbc_risk"] = df.apply(label_cbc, axis=1)
    return df


def load_pathology_real(base_dir: str | Path = ".") -> pd.DataFrame:
    spec = MODEL_SPECS["pathology"]
    df = pd.read_csv(require_file(Path(base_dir) / "data" / "real_processed" / "nhanes_chronic_labs_real.csv"))
    df = df.dropna(subset=spec.numeric_features + ["sex"]).copy()
    df["pathology_risk"] = df.apply(label_pathology, axis=1)
    return df


def load_radiology_real(base_dir: str | Path = ".") -> pd.DataFrame:
    path = require_file(Path(base_dir) / "data" / "real_processed" / "iu_xray_reports_real.csv")
    df = pd.read_csv(path).dropna(subset=["report_text"]).copy()
    df["urgency"] = df["report_text"].map(label_radiology)
    return df[["report_text", "urgency"]]


def load_heart_real(base_dir: str | Path = ".") -> pd.DataFrame:
    df = pd.read_csv(require_file(Path(base_dir) / "data" / "real_processed" / "heart_disease_uci_real.csv")).copy()
    for col in MODEL_SPECS["heart_disease"].categorical_features:
        df[col] = df[col].astype(str)
    df["heart_risk"] = np.where(df["heart_disease_label"] == 1, "high", "low")
    return df


def load_diabetes_real(base_dir: str | Path = ".") -> pd.DataFrame:
    spec = MODEL_SPECS["diabetes"]
    df = pd.read_csv(require_file(Path(base_dir) / "data" / "real_processed" / "nhanes_chronic_labs_real.csv"))
    df = df.dropna(subset=["fasting_glucose", "hba1c", "sex"]).copy()
    df["diabetes_risk"] = np.where(df["diabetes_label"] == 1, "high", "low")
    return df[spec.numeric_features + spec.categorical_features + ["diabetes_risk"]]


def load_hypertension_real(base_dir: str | Path = ".") -> pd.DataFrame:
    spec = MODEL_SPECS["hypertension"]
    df = pd.read_csv(require_file(Path(base_dir) / "data" / "real_processed" / "nhanes_chronic_labs_real.csv"))
    df = df.dropna(subset=["systolic_bp", "diastolic_bp", "sex"]).copy()
    df["hypertension_risk"] = np.where(df["hypertension_label"] == 1, "high", "low")
    return df[spec.numeric_features + spec.categorical_features + ["hypertension_risk"]]


REAL_DATA_LOADERS = {
    "cbc": load_cbc_real,
    "pathology": load_pathology_real,
    "radiology": load_radiology_real,
    "heart_disease": load_heart_real,
    "diabetes": load_diabetes_real,
    "hypertension": load_hypertension_real,
}


def feature_target_split(name: str, df: pd.DataFrame) -> tuple[Any, pd.Series]:
    spec = MODEL_SPECS[name]
    if spec.text_feature:
        return df[spec.text_feature], df[spec.target]
    return df[spec.numeric_features + spec.categorical_features], df[spec.target]


def preprocessing(spec: ModelSpec) -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", numeric, spec.numeric_features), ("cat", categorical, spec.categorical_features)], sparse_threshold=0)


def candidate_pipelines(spec: ModelSpec) -> dict[str, Pipeline]:
    if spec.text_feature:
        return {
            "tfidf_logistic_regression": Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000)), ("model", LogisticRegression(max_iter=1500, class_weight="balanced"))]),
            "tfidf_linear_svm": Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000)), ("model", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=RANDOM_STATE))]),
            "tfidf_random_forest": Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000)), ("model", RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=RANDOM_STATE))]),
        }
    return {
        "logistic_regression": Pipeline([("preprocessor", preprocessing(spec)), ("model", LogisticRegression(max_iter=1500, class_weight="balanced"))]),
        "random_forest": Pipeline([("preprocessor", preprocessing(spec)), ("model", RandomForestClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE))]),
        "gradient_boosting": Pipeline([("preprocessor", preprocessing(spec)), ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))]),
        "hist_gradient_boosting": Pipeline([("preprocessor", preprocessing(spec)), ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE))]),
        "rbf_svm": Pipeline([("preprocessor", preprocessing(spec)), ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE))]),
    }


def positive_recall(y_true: pd.Series, y_pred: np.ndarray, positive_label: str) -> float:
    if positive_label not in (set(y_true) | set(y_pred)):
        return 0.0
    return float(recall_score(y_true, y_pred, labels=[positive_label], average="macro", zero_division=0))


def evaluate_pipeline(pipeline: Pipeline, x_val: Any, y_val: pd.Series, spec: ModelSpec) -> dict[str, Any]:
    pred = pipeline.predict(x_val)
    return {
        "accuracy": float(accuracy_score(y_val, pred)),
        "macro_f1": float(f1_score(y_val, pred, average="macro", zero_division=0)),
        "positive_label": spec.positive_label,
        "positive_recall": positive_recall(y_val, pred, spec.positive_label),
    }


def train_model(name: str, output_dir: str | Path = ".") -> dict[str, Any]:
    spec = MODEL_SPECS[name]
    df = REAL_DATA_LOADERS[name](output_dir)
    x, y = feature_target_split(name, df)
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp)

    candidate_metrics: dict[str, Any] = {}
    fitted: dict[str, Pipeline] = {}
    for candidate_name, pipeline in candidate_pipelines(spec).items():
        pipeline.fit(x_train, y_train)
        fitted[candidate_name] = pipeline
        candidate_metrics[candidate_name] = evaluate_pipeline(pipeline, x_val, y_val, spec)

    best_name = sorted(
        candidate_metrics,
        key=lambda item: (candidate_metrics[item]["positive_recall"], candidate_metrics[item]["macro_f1"], candidate_metrics[item]["accuracy"]),
        reverse=True,
    )[0]
    best_pipeline = fitted[best_name]
    test_pred = best_pipeline.predict(x_test)

    metrics = {
        "model": name,
        "task": spec.task,
        "dataset": "real",
        "label_source": "original dataset target" if name == "heart_disease" else "derived label from real clinical measurements/report text",
        "selection_metric": f"highest validation recall for {spec.positive_label}",
        "best_candidate": best_name,
        "candidate_validation_metrics": candidate_metrics,
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_macro_f1": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
        "test_positive_label": spec.positive_label,
        "test_positive_recall": positive_recall(y_test, test_pred, spec.positive_label),
        "test_confusion_matrix_labels": list(best_pipeline.classes_),
        "test_confusion_matrix": confusion_matrix(y_test, test_pred, labels=best_pipeline.classes_).tolist(),
        "test_classification_report": classification_report(y_test, test_pred, output_dict=True, zero_division=0),
        "class_distribution": y.value_counts().to_dict(),
    }

    base = Path(output_dir)
    (base / "models").mkdir(parents=True, exist_ok=True)
    (base / "metrics").mkdir(parents=True, exist_ok=True)
    (base / "schemas").mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, base / "models" / f"{name}_pipeline.joblib")
    (base / "metrics" / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (base / "schemas" / f"{name}_io_schema.json").write_text(json.dumps(build_io_schema(spec, best_pipeline, df), indent=2), encoding="utf-8")
    return metrics


def train_all(output_dir: str | Path = ".") -> pd.DataFrame:
    # NOTE: only heart_disease and radiology are trained through this legacy path.
    # cbc/pathology are rule-based now (src/lab_triage_rules.py, no training needed).
    # diabetes/hypertension are trained via scripts/retrain_risk_screening.py
    # (non-circular pre-lab risk screening) — do NOT retrain them here, that
    # would silently regenerate the old leaky models.
    legacy_ml_models = ["heart_disease", "radiology"]
    rows = [train_model(name, output_dir) for name in legacy_ml_models]
    return pd.DataFrame(
        [
            {
                "model": item["model"],
                "best_candidate": item["best_candidate"],
                "test_positive_label": item["test_positive_label"],
                "test_positive_recall": item["test_positive_recall"],
                "test_macro_f1": item["test_macro_f1"],
                "test_accuracy": item["test_accuracy"],
            }
            for item in rows
        ]
    )


def build_io_schema(spec: ModelSpec, pipeline: Pipeline | None = None, df: pd.DataFrame | None = None) -> dict[str, Any]:
    labels = list(pipeline.classes_) if pipeline is not None and hasattr(pipeline, "classes_") else []
    properties: dict[str, Any] = {}
    for col in spec.numeric_features:
        properties[col] = {"type": "number", "nullable": True}
    for col in spec.categorical_features:
        values = sorted(df[col].dropna().astype(str).unique().tolist()) if df is not None else []
        properties[col] = {"type": "string", "enum": values, "nullable": True}
    if spec.text_feature:
        properties[spec.text_feature] = {"type": "string", "minLength": 10}
    return {
        "model_name": spec.name,
        "task": spec.task,
        "input": {"type": "object", "required": list(properties), "properties": properties},
        "output": {
            "type": "object",
            "properties": {
                "prediction": {"type": "string", "enum": labels},
                "probabilities": {"type": "object", "additionalProperties": {"type": "number"}},
                "triage_action": {"type": "string", "enum": ["reassure_and_explain", "ask_follow_up_questions", "recommend_doctor_booking"]},
                "explanation": {"type": "string"},
                "safety_disclaimer": {"type": "string"},
            },
        },
    }


SCREENING_FEATURES = {
    "diabetes_risk_screening": ["age", "bmi", "systolic_bp", "diastolic_bp", "total_cholesterol", "sex"],
    "hypertension_risk_screening": ["age", "bmi", "total_cholesterol", "creatinine_mg_dl", "sex"],
}


def predict_for_backend(model_name: str, payload: dict[str, Any], base_dir: str | Path = ".") -> dict[str, Any]:
    # --- Rule-based models: no ML, no leakage risk, fully auditable ---
    if model_name in ("cbc", "pathology"):
        from src.lab_triage_rules import CBCTriage, PathologyTriage

        engine = CBCTriage() if model_name == "cbc" else PathologyTriage()
        result = engine.evaluate(payload)
        return {
            "model_name": model_name,
            "prediction": result.level.value,
            "probabilities": None,  # deterministic rules, not a probabilistic model
            "triggered_rules": result.triggered_rules,
            "triage_action": result.triage_action,
            "explanation": build_patient_friendly_explanation(model_name, result.level.value),
            "safety_disclaimer": "Educational decision support only, not a diagnosis. Urgent symptoms need urgent care.",
            "engine": "rule_based",
        }

    # --- Pre-lab risk screening models (diabetes, hypertension): non-circular ---
    if model_name in ("diabetes_risk_screening", "hypertension_risk_screening"):
        pipeline = joblib.load(Path(base_dir) / "models" / f"{model_name}.joblib")
        features = SCREENING_FEATURES[model_name]
        x = pd.DataFrame([payload], columns=features)
        risk_score = float(pipeline.predict_proba(x)[0, 1])
        prediction = "elevated_risk" if risk_score >= 0.5 else "low_risk"
        return {
            "model_name": model_name,
            "prediction": prediction,
            "risk_score": round(risk_score, 3),
            "triage_action": "ask_follow_up_questions" if risk_score >= 0.5 else "reassure_and_explain",
            "explanation": (
                f"Based on reported risk factors (not lab results), estimated likelihood of an "
                f"abnormal {model_name.split('_')[0]} lab result is {risk_score:.0%}. "
                "Recommend the actual lab test rather than relying on this estimate."
            ),
            "safety_disclaimer": "Educational pre-lab risk screening only, not a diagnosis. "
                                  "An actual lab test is needed to confirm.",
            "engine": "ml_non_circular",
        }

    spec = MODEL_SPECS[model_name]

    if model_name == "radiology":
        from src.radiology_extractor import extract_radiology_triage
        report_text = payload.get("report_text", "")
        extracted = extract_radiology_triage(report_text)
        prediction = extracted["urgency"]
        probabilities = {
            "urgent_review": 1.0 if prediction == "urgent_review" else 0.0,
            "follow_up": 1.0 if prediction == "follow_up" else 0.0,
            "routine": 1.0 if prediction == "routine" else 0.0,
        }
        return {
            "model_name": model_name,
            "prediction": prediction,
            "probabilities": probabilities,
            "triage_action": map_prediction_to_action(prediction),
            "explanation": extracted["explanation_eg"],
            "findings": extracted["findings"],
            "justification": extracted["justification"],
            "safety_disclaimer": "Educational decision support only, not a diagnosis. Urgent symptoms need urgent care.",
            "engine": "rule_based_negation_aware",
        }

    pipeline = joblib.load(Path(base_dir) / "models" / f"{model_name}_pipeline.joblib")
    if spec.text_feature:
        x = pd.Series([payload[spec.text_feature]])
    else:
        x = pd.DataFrame([payload], columns=spec.numeric_features + spec.categorical_features)
    prediction = str(pipeline.predict(x)[0])
    probabilities = {str(label): float(prob) for label, prob in zip(pipeline.classes_, pipeline.predict_proba(x)[0], strict=True)}
    return {
        "model_name": model_name,
        "prediction": prediction,
        "probabilities": probabilities,
        "triage_action": map_prediction_to_action(prediction),
        "explanation": build_patient_friendly_explanation(model_name, prediction),
        "safety_disclaimer": "Educational decision support only, not a diagnosis. Urgent symptoms need urgent care.",
    }



def map_prediction_to_action(prediction: str) -> str:
    if prediction in {"urgent_review", "high"}:
        return "recommend_doctor_booking"
    if prediction == "follow_up":
        return "ask_follow_up_questions"
    return "reassure_and_explain"


def build_patient_friendly_explanation(model_name: str, prediction: str) -> str:
    readable = model_name.replace("_", " ")
    if prediction in {"urgent_review", "high"}:
        return f"The {readable} model found a higher-risk pattern and the chatbot should recommend clinician review."
    if prediction == "follow_up":
        return f"The {readable} model found a non-urgent abnormal pattern and the chatbot should ask follow-up questions."
    return f"The {readable} model did not find a high-risk pattern in the submitted information."


def save_sample_payloads(output_dir: str | Path = ".") -> None:
    samples = {
        "cbc": {"age": 34, "sex": "female", "hemoglobin_g_dl": 10.2, "wbc_10e3_ul": 13.4, "platelets_10e3_ul": 210, "rbc_10e6_ul": 4.1, "mcv_fl": 74, "mch_pg": 25, "neutrophils_pct": 82, "lymphocytes_pct": 12},
        "pathology": {"age": 52, "sex": "male", "bmi": 30.1, "fasting_glucose": 146, "hba1c": 7.2, "total_cholesterol": 252, "creatinine_mg_dl": 1.1, "alt_u_l": 72, "ast_u_l": 48},
        "radiology": {"report_text": "There is a right apical pneumothorax. No pleural effusion."},
        "heart_disease": {"age": 61, "sex": "1.0", "cp": "4.0", "trestbps": 152, "chol": 248, "fbs": "0.0", "restecg": "2.0", "thalach": 108, "exang": "1.0", "oldpeak": 2.1, "slope": "2.0", "ca": 1, "thal": "7.0"},
        "diabetes": {"age": 57, "sex": "female", "bmi": 33.0, "fasting_glucose": 142, "hba1c": 6.9, "systolic_bp": 146, "total_cholesterol": 220},
        "hypertension": {"age": 58, "sex": "male", "bmi": 31.2, "systolic_bp": 154, "diastolic_bp": 94, "creatinine_mg_dl": 1.0},
    }
    base = Path(output_dir) / "schemas"
    base.mkdir(parents=True, exist_ok=True)
    (base / "sample_backend_payloads.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(train_all(root).to_string(index=False))
    save_sample_payloads(root)

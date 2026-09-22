from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(filename: str, cells: list[dict]) -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    (NOTEBOOKS / filename).write_text(json.dumps(notebook(cells), indent=2), encoding="utf-8")


def get_imports_cell() -> str:
    return (
        "# Imports and project path setup\n"
        "from pathlib import Path\n"
        "import sys, json\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import joblib\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.metrics import classification_report, confusion_matrix\n\n"
        "# Set ROOT path relative to project structure\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "sys.path.insert(0, str(ROOT))\n\n"
        "from src.diagnostic_models import MODEL_SPECS, build_io_schema, candidate_pipelines, evaluate_pipeline, predict_for_backend, train_model\n"
    )


def build_cbc_notebook() -> None:
    write_notebook(
        "01_cbc_model.ipynb",
        [
            md("# CBC Model\n\nThis notebook trains and evaluates the `cbc` model. It reads the local processed data file, performs cleaning, applies clinical rules, splits the data, and runs model selection."),
            code(
                "# Imports and project path setup\n"
                "from pathlib import Path\n"
                "import sys, json\n"
                "import pandas as pd\n"
                "import numpy as np\n"
                "import joblib\n"
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "from sklearn.model_selection import train_test_split\n"
                "from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay\n\n"
                "# Set ROOT path relative to project structure\n"
                "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
                "sys.path.insert(0, str(ROOT))\n\n"
                "from src.diagnostic_models import MODEL_SPECS, build_io_schema, candidate_pipelines, evaluate_pipeline, predict_for_backend, train_model\n"
            ),
            code(
                "# 1. Load Dataset\n"
                "csv_path = ROOT / 'data' / 'real_processed' / 'cbc_nhanes_real.csv'\n"
                "df = pd.read_csv(csv_path)\n"
                "display(df.head())\n"
                "print('Shape:', df.shape)\n"
            ),
            code(
                "# 2. Cleaning and Labeling (Best Practice Dropna & Imputation)\n"
                "spec = MODEL_SPECS['cbc']\n"
                "# Drop only rows that miss critical variables needed to compute the target or sex demographics\n"
                "critical_features = ['hemoglobin_g_dl', 'wbc_10e3_ul', 'platelets_10e3_ul', 'sex']\n"
                "df = df.dropna(subset=critical_features).copy()\n\n"
                "def label_cbc(row):\n"
                "    urgent = (\n"
                "        row['hemoglobin_g_dl'] < 8\n"
                "        or row['wbc_10e3_ul'] < 2\n"
                "        or row['wbc_10e3_ul'] > 20\n"
                "        or row['platelets_10e3_ul'] < 50\n"
                "        or row['platelets_10e3_ul'] > 800\n"
                "    )\n"
                "    follow_up = (\n"
                "        row['hemoglobin_g_dl'] < 11\n"
                "        or row['wbc_10e3_ul'] < 3.5\n"
                "        or row['wbc_10e3_ul'] > 12\n"
                "        or row['platelets_10e3_ul'] < 150\n"
                "        or row['platelets_10e3_ul'] > 450\n"
                "        or row['mcv_fl'] < 78\n"
                "        or row['neutrophils_pct'] > 80\n"
                "    )\n"
                "    return 'urgent_review' if urgent else 'follow_up' if follow_up else 'within_reference'\n\n"
                "df['cbc_risk'] = df.apply(label_cbc, axis=1)\n"
                "print(df['cbc_risk'].value_counts(normalize=True).round(3))\n"
            ),
            md("## 3. Exploratory Data Analysis & Visualizations\n\nLet's plot class distributions and analyze feature correlations before splitting and training."),
            code(
                "# Class Distribution Plot\n"
                "plt.figure(figsize=(7, 4))\n"
                "sns.countplot(data=df, x='cbc_risk', order=['within_reference', 'follow_up', 'urgent_review'], palette='viridis')\n"
                "plt.title('CBC Triage Class Distribution (Imbalanced Medical Data)')\n"
                "plt.xlabel('Triage Category')\n"
                "plt.ylabel('Count')\n"
                "plt.show()\n\n"
                "# Feature Correlation Heatmap\n"
                "plt.figure(figsize=(9, 7))\n"
                "corr = df[spec.numeric_features].corr()\n"
                "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)\n"
                "plt.title('Correlation Matrix of Blood Count Features')\n"
                "plt.tight_layout()\n"
                "plt.show()\n"
            ),
            code(
                "# 4. Train / Validation / Test Split\n"
                "X = df[spec.numeric_features + spec.categorical_features]\n"
                "y = df[spec.target]\n"
                "X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)\n"
                "X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)\n"
                "print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')\n"
            ),
            code(
                "# 5. Preprocessing + Model Selection\n"
                "candidate_results = {}\n"
                "fitted = {}\n"
                "for candidate_name, pipeline in candidate_pipelines(spec).items():\n"
                "    pipeline.fit(X_train, y_train)\n"
                "    fitted[candidate_name] = pipeline\n"
                "    candidate_results[candidate_name] = evaluate_pipeline(pipeline, X_val, y_val, spec)\n\n"
                "display(pd.DataFrame(candidate_results).T.sort_values(['positive_recall', 'macro_f1'], ascending=False))\n"
                "best_name = sorted(candidate_results, key=lambda k: (candidate_results[k]['positive_recall'], candidate_results[k]['macro_f1'], candidate_results[k]['accuracy']), reverse=True)[0]\n"
                "pipeline = fitted[best_name]\n"
                "print('Best model:', best_name)\n"
            ),
            md("## 6. Feature Importance\n\nLet's see which features contribute the most to the selected model's predictions."),
            code(
                "if hasattr(pipeline.named_steps['model'], 'feature_importances_'):\n"
                "    importances = pipeline.named_steps['model'].feature_importances_\n"
                "    cat_cols = list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(spec.categorical_features))\n"
                "    feature_names = spec.numeric_features + cat_cols\n"
                "    feat_importances = pd.Series(importances, index=feature_names).sort_values(ascending=True)\n"
                "    plt.figure(figsize=(9, 5))\n"
                "    feat_importances.plot(kind='barh', color='teal')\n"
                "    plt.title(f'Feature Importance ({best_name})')\n"
                "    plt.xlabel('Relative Importance')\n"
                "    plt.tight_layout()\n"
                "    plt.show()\n"
                "else:\n"
                "    print(f'Feature importance is not supported directly for {best_name}.')\n"
            ),
            code(
                "# 7. Evaluation Metrics & Confusion Matrices\n"
                "for split_name, X_split, y_split in [('validation', X_val, y_val), ('test', X_test, y_test)]:\n"
                "    preds = pipeline.predict(X_split)\n"
                "    print('\\n' + split_name.upper())\n"
                "    print(classification_report(y_split, preds, zero_division=0))\n\n"
                "# Visual Confusion Matrix Display\n"
                "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n"
                "for ax, (split_name, X_split, y_split) in zip(axes, [('validation', X_val, y_val), ('test', X_test, y_test)]):\n"
                "    preds = pipeline.predict(X_split)\n"
                "    cm = confusion_matrix(y_split, preds, labels=pipeline.classes_)\n"
                "    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)\n"
                "    disp.plot(cmap='Blues', ax=ax, values_format='d')\n"
                "    ax.set_title(f'Confusion Matrix - {split_name.upper()}')\n"
                "plt.tight_layout()\n"
                "plt.show()\n"
            ),
            code(
                "# 8. Save Model\n"
                "joblib.dump(pipeline, ROOT / 'models' / 'cbc_pipeline.joblib')\n"
                "print('Model saved successfully to models/cbc_pipeline.joblib.')\n"
            ),
            md("## 9. Backend Contract Prediction Simulation\n\nSimulating how the backend would invoke this pipeline for real-world inference:"),
            code(
                "payload = {'age': 34, 'sex': 'female', 'hemoglobin_g_dl': 10.2, 'wbc_10e3_ul': 13.4, 'platelets_10e3_ul': 210, 'rbc_10e6_ul': 4.1, 'mcv_fl': 74, 'mch_pg': 25, 'neutrophils_pct': 82, 'lymphocytes_pct': 12}\n"
                "result = predict_for_backend('cbc', payload, ROOT)\n"
                "print(json.dumps(result, indent=2, ensure_ascii=False))\n"
            )
        ],
    )



def build_pathology_notebook() -> None:
    write_notebook(
        "02_pathology_model.ipynb",
        [
            md("# Pathology Model\n\nThis notebook trains and evaluates the `pathology` model. It reads the local biochemistry profile data, performs cleaning, applies clinical rules, splits the data, and runs model selection."),
            code(get_imports_cell()),
            code(
                "# 1. Load Dataset\n"
                "csv_path = ROOT / 'data' / 'real_processed' / 'nhanes_chronic_labs_real.csv'\n"
                "df = pd.read_csv(csv_path)\n"
                "display(df.head())\n"
                "print('Shape:', df.shape)\n"
            ),
            code(
                "# 2. Cleaning and Labeling\n"
                "spec = MODEL_SPECS['pathology']\n"
                "df = df.dropna(subset=spec.numeric_features + ['sex']).copy()\n\n"
                "def label_pathology(row):\n"
                "    urgent = (\n"
                "        row['fasting_glucose'] >= 250\n"
                "        or row['hba1c'] >= 10\n"
                "        or row['creatinine_mg_dl'] >= 2\n"
                "        or row['alt_u_l'] >= 200\n"
                "        or row['ast_u_l'] >= 200\n"
                "        or row['total_cholesterol'] >= 320\n"
                "    )\n"
                "    follow_up = (\n"
                "        row['fasting_glucose'] >= 126\n"
                "        or row['hba1c'] >= 6.5\n"
                "        or row['creatinine_mg_dl'] >= 1.3\n"
                "        or row['alt_u_l'] >= 45\n"
                "        or row['ast_u_l'] >= 45\n"
                "        or row['total_cholesterol'] >= 240\n"
                "    )\n"
                "    return 'urgent_review' if urgent else 'follow_up' if follow_up else 'within_reference'\n\n"
                "df['pathology_risk'] = df.apply(label_pathology, axis=1)\n"
                "print(df['pathology_risk'].value_counts(normalize=True).round(3))\n"
            ),
            code(
                "# 3. Train / Validation / Test Split\n"
                "X = df[spec.numeric_features + spec.categorical_features]\n"
                "y = df[spec.target]\n"
                "X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)\n"
                "X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)\n"
                "print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')\n"
            ),
            code(
                "# 4. Preprocessing + Model Selection\n"
                "candidate_results = {}\n"
                "fitted = {}\n"
                "for candidate_name, pipeline in candidate_pipelines(spec).items():\n"
                "    pipeline.fit(X_train, y_train)\n"
                "    fitted[candidate_name] = pipeline\n"
                "    candidate_results[candidate_name] = evaluate_pipeline(pipeline, X_val, y_val, spec)\n\n"
                "display(pd.DataFrame(candidate_results).T.sort_values(['positive_recall', 'macro_f1'], ascending=False))\n"
                "best_name = sorted(candidate_results, key=lambda k: (candidate_results[k]['positive_recall'], candidate_results[k]['macro_f1'], candidate_results[k]['accuracy']), reverse=True)[0]\n"
                "pipeline = fitted[best_name]\n"
                "print('Best model:', best_name)\n"
            ),
            code(
                "# 5. Evaluation Metrics\n"
                "for split_name, X_split, y_split in [('validation', X_val, y_val), ('test', X_test, y_test)]:\n"
                "    preds = pipeline.predict(X_split)\n"
                "    print('\\n' + split_name.upper())\n"
                "    print(classification_report(y_split, preds, zero_division=0))\n"
                "    print(confusion_matrix(y_split, preds, labels=pipeline.classes_))\n"
            ),
            code(
                "# 6. Save model\n"
                "joblib.dump(pipeline, ROOT / 'models' / 'pathology_pipeline.joblib')\n"
                "print('Model saved successfully.')\n"
            ),
        ],
    )


def build_radiology_notebook() -> None:
    write_notebook(
        "03_radiology_model.ipynb",
        [
            md("# Radiology Model\n\nThis notebook trains and evaluates the `radiology` text classification model. It reads reports, applies keyword labeling rules, runs classical ML benchmarks, and demonstrates the LLM structured radiology extractor option."),
            code(get_imports_cell()),
            code(
                "# 1. Load Dataset\n"
                "csv_path = ROOT / 'data' / 'real_processed' / 'iu_xray_reports_real.csv'\n"
                "df = pd.read_csv(csv_path)\n"
                "display(df.head())\n"
                "print('Shape:', df.shape)\n"
            ),
            code(
                "# 2. Cleaning and Labeling\n"
                "spec = MODEL_SPECS['radiology']\n"
                "df = df.dropna(subset=['report_text']).copy()\n\n"
                "def label_radiology(text):\n"
                "    t = str(text).lower()\n"
                "    urgent_terms = ['pneumothorax', 'pulmonary edema', 'acute', 'collapse', 'mediastinal shift', 'large pleural effusion', 'consolidation']\n"
                "    follow_terms = ['opacity', 'nodule', 'mass', 'effusion', 'atelectasis', 'cardiomegaly', 'emphysema', 'fibrosis', 'granuloma', 'hernia', 'infiltrate']\n"
                "    negated_urgent = ['no pneumothorax', 'without pneumothorax', 'no acute']\n"
                "    if any(term in t for term in urgent_terms) and not any(term in t for term in negated_urgent):\n"
                "        return 'urgent_review'\n"
                "    if any(term in t for term in follow_terms):\n"
                "        return 'follow_up'\n"
                "    return 'routine'\n\n"
                "df['urgency'] = df['report_text'].map(label_radiology)\n"
                "print(df['urgency'].value_counts(normalize=True).round(3))\n"
            ),
            code(
                "# 3. Train / Validation / Test Split\n"
                "X = df['report_text']\n"
                "y = df['urgency']\n"
                "X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)\n"
                "X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)\n"
                "print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')\n"
            ),
            code(
                "# 4. Preprocessing + Model Selection (Classical TF-IDF baseline)\n"
                "candidate_results = {}\n"
                "fitted = {}\n"
                "for candidate_name, pipeline in candidate_pipelines(spec).items():\n"
                "    pipeline.fit(X_train, y_train)\n"
                "    fitted[candidate_name] = pipeline\n"
                "    candidate_results[candidate_name] = evaluate_pipeline(pipeline, X_val, y_val, spec)\n\n"
                "display(pd.DataFrame(candidate_results).T.sort_values(['positive_recall', 'macro_f1'], ascending=False))\n"
                "best_name = sorted(candidate_results, key=lambda k: (candidate_results[k]['positive_recall'], candidate_results[k]['macro_f1'], candidate_results[k]['accuracy']), reverse=True)[0]\n"
                "pipeline = fitted[best_name]\n"
                "print('Best classical model:', best_name)\n"
            ),
            code(
                "# 5. Classical Evaluation Metrics\n"
                "for split_name, X_split, y_split in [('validation', X_val, y_val), ('test', X_test, y_test)]:\n"
                "    preds = pipeline.predict(X_split)\n"
                "    print('\\n' + split_name.upper())\n"
                "    print(classification_report(y_split, preds, zero_division=0))\n"
                "    print(confusion_matrix(y_split, preds, labels=pipeline.classes_))\n"
            ),
            code(
                "# 6. Save Classical Model (Optional)\n"
                "joblib.dump(pipeline, ROOT / 'models' / 'radiology_pipeline.joblib')\n"
                "print('Classical baseline model saved successfully.')\n"
            ),
            md("## 7. Advanced: LLM Radiology Extractor\n\nFor production, we use the LLM Radiology Extractor which provides clinical structured data and patient-friendly explanations in Egyptian Arabic. Let's test it:"),
            code(
                "from src.radiology_extractor import extract_radiology_triage\n\n"
                "sample_report = \"Chest radiograph: Heart size is normal. No focal air space consolidation or pleural effusion. There is a right-sided apical pneumothorax.\"\n"
                "extracted_data = extract_radiology_triage(sample_report)\n"
                "print(json.dumps(extracted_data, indent=2, ensure_ascii=False))\n"
            )
        ],
    )


def build_heart_disease_notebook() -> None:
    write_notebook(
        "04_heart_disease_model.ipynb",
        [
            md("# Heart Disease Model\n\nThis notebook trains and evaluates the `heart_disease` model. It reads the local Cleveland dataset, splits features, and runs model selection."),
            code(get_imports_cell()),
            code(
                "# 1. Load Dataset\n"
                "csv_path = ROOT / 'data' / 'real_processed' / 'heart_disease_uci_real.csv'\n"
                "df = pd.read_csv(csv_path)\n"
                "display(df.head())\n"
                "print('Shape:', df.shape)\n"
            ),
            code(
                "# 2. Cleaning and Labeling\n"
                "spec = MODEL_SPECS['heart_disease']\n"
                "for col in spec.categorical_features:\n"
                "    df[col] = df[col].astype(str)\n"
                "df['heart_risk'] = np.where(df['heart_disease_label'] == 1, 'high', 'low')\n"
                "print(df['heart_risk'].value_counts(normalize=True).round(3))\n"
            ),
            code(
                "# 3. Train / Validation / Test Split\n"
                "X = df[spec.numeric_features + spec.categorical_features]\n"
                "y = df[spec.target]\n"
                "X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)\n"
                "X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)\n"
                "print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')\n"
            ),
            code(
                "# 4. Preprocessing + Model Selection\n"
                "candidate_results = {}\n"
                "fitted = {}\n"
                "for candidate_name, pipeline in candidate_pipelines(spec).items():\n"
                "    pipeline.fit(X_train, y_train)\n"
                "    fitted[candidate_name] = pipeline\n"
                "    candidate_results[candidate_name] = evaluate_pipeline(pipeline, X_val, y_val, spec)\n\n"
                "display(pd.DataFrame(candidate_results).T.sort_values(['positive_recall', 'macro_f1'], ascending=False))\n"
                "best_name = sorted(candidate_results, key=lambda k: (candidate_results[k]['positive_recall'], candidate_results[k]['macro_f1'], candidate_results[k]['accuracy']), reverse=True)[0]\n"
                "pipeline = fitted[best_name]\n"
                "print('Best model:', best_name)\n"
            ),
            code(
                "# 5. Evaluation Metrics\n"
                "for split_name, X_split, y_split in [('validation', X_val, y_val), ('test', X_test, y_test)]:\n"
                "    preds = pipeline.predict(X_split)\n"
                "    print('\\n' + split_name.upper())\n"
                "    print(classification_report(y_split, preds, zero_division=0))\n"
                "    print(confusion_matrix(y_split, preds, labels=pipeline.classes_))\n"
            ),
            code(
                "# 6. Save model\n"
                "joblib.dump(pipeline, ROOT / 'models' / 'heart_disease_pipeline.joblib')\n"
                "print('Model saved successfully.')\n"
            ),
        ],
    )


def build_diabetes_notebook() -> None:
    write_notebook(
        "05_diabetes_model.ipynb",
        [
            md("# Diabetes Model\n\nThis notebook trains and evaluates the `diabetes` model. It reads the biochemistry profile, extracts metabolic risk factors, splits the data, and runs model selection."),
            code(get_imports_cell()),
            code(
                "# 1. Load Dataset\n"
                "csv_path = ROOT / 'data' / 'real_processed' / 'nhanes_chronic_labs_real.csv'\n"
                "df = pd.read_csv(csv_path)\n"
                "display(df.head())\n"
                "print('Shape:', df.shape)\n"
            ),
            code(
                "# 2. Cleaning and Labeling\n"
                "spec = MODEL_SPECS['diabetes']\n"
                "df = df.dropna(subset=['fasting_glucose', 'hba1c', 'sex']).copy()\n"
                "df['diabetes_risk'] = np.where(df['diabetes_label'] == 1, 'high', 'low')\n"
                "print(df['diabetes_risk'].value_counts(normalize=True).round(3))\n"
            ),
            code(
                "# 3. Train / Validation / Test Split\n"
                "X = df[spec.numeric_features + spec.categorical_features]\n"
                "y = df[spec.target]\n"
                "X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)\n"
                "X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)\n"
                "print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')\n"
            ),
            code(
                "# 4. Preprocessing + Model Selection\n"
                "candidate_results = {}\n"
                "fitted = {}\n"
                "for candidate_name, pipeline in candidate_pipelines(spec).items():\n"
                "    pipeline.fit(X_train, y_train)\n"
                "    fitted[candidate_name] = pipeline\n"
                "    candidate_results[candidate_name] = evaluate_pipeline(pipeline, X_val, y_val, spec)\n\n"
                "display(pd.DataFrame(candidate_results).T.sort_values(['positive_recall', 'macro_f1'], ascending=False))\n"
                "best_name = sorted(candidate_results, key=lambda k: (candidate_results[k]['positive_recall'], candidate_results[k]['macro_f1'], candidate_results[k]['accuracy']), reverse=True)[0]\n"
                "pipeline = fitted[best_name]\n"
                "print('Best model:', best_name)\n"
            ),
            code(
                "# 5. Evaluation Metrics\n"
                "for split_name, X_split, y_split in [('validation', X_val, y_val), ('test', X_test, y_test)]:\n"
                "    preds = pipeline.predict(X_split)\n"
                "    print('\\n' + split_name.upper())\n"
                "    print(classification_report(y_split, preds, zero_division=0))\n"
                "    print(confusion_matrix(y_split, preds, labels=pipeline.classes_))\n"
            ),
            code(
                "# 6. Save model\n"
                "joblib.dump(pipeline, ROOT / 'models' / 'diabetes_pipeline.joblib')\n"
                "print('Model saved successfully.')\n"
            ),
        ],
    )


def build_hypertension_notebook() -> None:
    write_notebook(
        "06_hypertension_model.ipynb",
        [
            md("# Hypertension Model\n\nThis notebook trains and evaluates the `hypertension` model. It reads local blood pressure features, splits the data, and runs model selection."),
            code(get_imports_cell()),
            code(
                "# 1. Load Dataset\n"
                "csv_path = ROOT / 'data' / 'real_processed' / 'nhanes_chronic_labs_real.csv'\n"
                "df = pd.read_csv(csv_path)\n"
                "display(df.head())\n"
                "print('Shape:', df.shape)\n"
            ),
            code(
                "# 2. Cleaning and Labeling\n"
                "spec = MODEL_SPECS['hypertension']\n"
                "df = df.dropna(subset=['systolic_bp', 'diastolic_bp', 'sex']).copy()\n"
                "df['hypertension_risk'] = np.where(df['hypertension_label'] == 1, 'high', 'low')\n"
                "print(df['hypertension_risk'].value_counts(normalize=True).round(3))\n"
            ),
            code(
                "# 3. Train / Validation / Test Split\n"
                "X = df[spec.numeric_features + spec.categorical_features]\n"
                "y = df[spec.target]\n"
                "X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)\n"
                "X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)\n"
                "print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')\n"
            ),
            code(
                "# 4. Preprocessing + Model Selection\n"
                "candidate_results = {}\n"
                "fitted = {}\n"
                "for candidate_name, pipeline in candidate_pipelines(spec).items():\n"
                "    pipeline.fit(X_train, y_train)\n"
                "    fitted[candidate_name] = pipeline\n"
                "    candidate_results[candidate_name] = evaluate_pipeline(pipeline, X_val, y_val, spec)\n\n"
                "display(pd.DataFrame(candidate_results).T.sort_values(['positive_recall', 'macro_f1'], ascending=False))\n"
                "best_name = sorted(candidate_results, key=lambda k: (candidate_results[k]['positive_recall'], candidate_results[k]['macro_f1'], candidate_results[k]['accuracy']), reverse=True)[0]\n"
                "pipeline = fitted[best_name]\n"
                "print('Best model:', best_name)\n"
            ),
            code(
                "# 5. Evaluation Metrics\n"
                "for split_name, X_split, y_split in [('validation', X_val, y_val), ('test', X_test, y_test)]:\n"
                "    preds = pipeline.predict(X_split)\n"
                "    print('\\n' + split_name.upper())\n"
                "    print(classification_report(y_split, preds, zero_division=0))\n"
                "    print(confusion_matrix(y_split, preds, labels=pipeline.classes_))\n"
            ),
            code(
                "# 6. Save model\n"
                "joblib.dump(pipeline, ROOT / 'models' / 'hypertension_pipeline.joblib')\n"
                "print('Model saved successfully.')\n"
            ),
        ],
    )


def build_orchestration_notebook() -> None:
    write_notebook(
        "00_model_router_and_llm_integration.ipynb",
        [
            md("# Model Router and LLM Integration\n\nThis notebook defines how the chatbot decides which diagnostic helper model to call and how to wrap the result for an LLM response."),
            code(
                "from pathlib import Path\n"
                "import sys, json\n"
                "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
                "sys.path.insert(0, str(ROOT))\n"
                "from src.diagnostic_models import predict_for_backend\n"
                "from src.router import route_request_llm, route_request_fallback\n"
            ),
            code(
                "# Recommended LLM setup\n"
                "llm_recommendations = {\n"
                "    'best_cloud_quality': 'GPT-4o or Gemini 2.5 Pro for Egyptian Arabic medical explanations and safety-aware follow-up',\n"
                "    'open_source_arabic_capable': 'Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct via Hugging Face/Transformers',\n"
                "    'medical_rag': 'Add retrieval from approved Arabic medical content before final answers',\n"
                "    'guardrails': ['no final diagnosis', 'urgent symptom escalation', 'ask for missing context']\n"
                "}\n"
                "print(json.dumps(llm_recommendations, indent=2))\n"
            ),
            code(
                "# Prompt template that the backend sends to the LLM after model prediction.\n"
                "def build_llm_prompt(user_message: str, model_output: dict) -> str:\n"
                "    return f'''\n"
                "أنت طبيب مصري ودود وخبير. اشرح النتيجة الطبية التالية للمريض بالعامية المصرية المبسطة جداً (اللهجة المصرية).\n"
                "لا تشخص المرض كتشخيص نهائي، بل وضح له المعنى ببساطة.\n"
                "وضح أي علامات خطر حمراء (Red Flags) بشكل صريح جداً إذا وجدتها.\n"
                "اسأله أسئلة متابعة فقط إذا كانت ستغير من درجة خطورة الحالة.\n"
                "الرسالة من المستخدم: {user_message}\n"
                "مخرجات النموذج الطبي (JSON): {json.dumps(model_output, ensure_ascii=False)}\n"
                "المطلوب إرجاعه بالتفصيل:\n"
                "1. ملخص بسيط للحالة بالعامية المصرية.\n"
                "2. معنى المصطلحات غير الطبيعية (مثل فقر الدم، وظائف الكلى، إلخ) بأسلوب مصري مبسط.\n"
                "3. الخطوة التالية اللي المفروض يعملها (زي حجز ميعاد مع دكتور أو الذهاب للطوارئ).\n"
                "4. أسئلة المتابعة لو فيه حاجة ناقصة.\n"
                "'''.strip()\n"
            ),
            code(
                "# End-to-end example\n"
                "message = 'CBC result: hemoglobin low and WBC high, should I worry?'\n"
                "payload = {\n"
                "    'age': 34, 'sex': 'female', 'hemoglobin_g_dl': 10.2, 'wbc_10e3_ul': 13.4,\n"
                "    'platelets_10e3_ul': 210, 'rbc_10e6_ul': 4.1, 'mcv_fl': 74, 'mch_pg': 25,\n"
                "    'neutrophils_pct': 82, 'lymphocytes_pct': 12\n"
                "}\n"
                "model_name = route_request_llm(message, payload)\n"
                "model_output = predict_for_backend(model_name, payload, ROOT)\n"
                "print('Routed model:', model_name)\n"
                "print(json.dumps(model_output, indent=2, ensure_ascii=False))\n"
                "print('\\nLLM PROMPT:\\n', build_llm_prompt(message, model_output))\n"
            ),
        ],
    )


if __name__ == "__main__":
    build_orchestration_notebook()
    build_cbc_notebook()
    build_pathology_notebook()
    build_radiology_notebook()
    build_heart_disease_notebook()
    build_diabetes_notebook()
    build_hypertension_notebook()
    print("Notebooks written successfully with sequential file loading.")

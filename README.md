# Diagnostic Agent — Medical Conversational Assistant

A patient-facing medical AI assistant designed to provide understandable health information, discuss symptoms, explain medical terminology, and support the interpretation/triage of laboratory and radiology information.

> **Important:** This project is an educational/decision-support prototype. It is not a substitute for a licensed clinician and should not be used as a standalone diagnostic or treatment system.

---

## 1. Project Overview

The system combines:

- Natural-language understanding (NLU) for Arabic, English, and mixed-language messages.
- A conversational medical assistant layer for patient-friendly answers.
- A curated medical Knowledge Base (RAG-style retrieval).
- MedlinePlus live-search fallback when curated retrieval does not contain enough context.
- Safety/triage logic for potentially urgent symptoms.
- Specialized risk-screening models for heart disease, diabetes, and hypertension.
- Rule-based triage for CBC/pathology and radiology instead of relying on circular/label-leaking ML models.
- FastAPI backend and a browser chat interface.

### Main conversation flow

```text
User Message
     |
     v
Natural Language Understanding
     |
     +------------------+-------------------+
     |                  |                   |
     v                  v                   v
General Medical     Lab / CBC /        Risk Screening
Conversation        Radiology              |
     |                  |                   v
     +------------------+------------> Safety Guard
                                      |
                         +------------+------------+
                         |                         |
                       Normal                   Red Flag
                         |                         |
                         v                         v
                      Answer              Urgent-care guidance
```

The safety layer is intended to act as a **guardrail**, not as a mandatory questionnaire for every conversation.

---

## 2. Model Quality / Evaluation

The following metrics are the **held-out test-set metrics stored in the project's `metrics/` directory**. They should not be interpreted as clinical validation or real-world diagnostic performance.

| Model | Test Accuracy | Test Recall | Macro F1 | ROC-AUC | Test Set |
|---|---:|---:|---:|---:|---:|
| Heart Disease Risk | **82.61%** | **80.95%** | 82.48% | — | 46 |
| Diabetes Risk Screening | **68.27%** | **71.05%** | 56.70% | 75.78% | — |
| Hypertension Risk Screening | **60.62%** | **81.25%** | 55.85% | 74.28% | — |

### Metric interpretation

- **Accuracy:** percentage of all test examples classified correctly.
- **Recall:** percentage of positive/high-risk cases detected by the model. Recall is especially important for screening because missing a potentially high-risk case can be more consequential than generating an additional follow-up case.
- **Macro F1:** average F1-score across classes, giving each class equal weight.
- **ROC-AUC:** ability of the model to distinguish the positive and negative classes across thresholds.

### Important model-quality note

Earlier versions of the project contained models affected by **label leakage**, where labels were derived directly from the same laboratory values used as input features. Those models could report artificially near-perfect performance and were therefore deprecated.

The current design avoids using those leaked CBC/pathology models as ML models and uses:

- **CBC:** deterministic rule-based triage.
- **Pathology:** deterministic rule-based triage.
- **Radiology:** negation-aware rule-based triage.
- **Diabetes/Hypertension:** retrained risk-screening models that exclude the defining lab measurements from the model inputs.
- **Heart disease:** trained using the UCI Cleveland heart-disease dataset.

Therefore, the deprecated near-100% results should **not** be reported as the performance of the current system.

---

## 3. Inference Latency

The following is a **local single-sample model-inference benchmark**, not end-to-end chatbot latency.

Benchmark method:

- Models loaded once (warm inference).
- One sample per prediction.
- 200 prediction iterations after warm-up.
- Reported values are median and P95 inference time.
- Includes local preprocessing inside the scikit-learn pipeline.
- **Does not include LLM/API/network latency, MedlinePlus search latency, FastAPI request overhead, or browser rendering.**

| Model | Median Inference | P95 Inference |
|---|---:|---:|
| Heart Disease Risk | **2.32 ms** | **3.13 ms** |
| Diabetes Risk Screening | **2.02 ms** | **2.71 ms** |
| Hypertension Risk Screening | **2.06 ms** | **2.69 ms** |

These values are hardware/environment dependent. They should be treated as a benchmark snapshot rather than a guaranteed production SLA.

For the complete conversational assistant, response latency can be higher because an interaction may also involve retrieval, an external medical search, and/or an LLM API call.

---

## 4. Knowledge Base

The curated Knowledge Base is located at:

```text
data/knowledge_base/
```

Current documents:

| File | Main Content |
|---|---|
| `cbc_interpretation.md` | CBC values and basic interpretation guidance |
| `diabetes.md` | Diabetes criteria, prediabetes, risk factors, and symptoms that warrant testing |
| `general_emergency_signs.md` | General emergency warning signs |
| `headache.md` | Headache information and red-flag symptoms |
| `heart_attack_warning_signs.md` | Heart-attack warning signs and major risk factors |
| `hypertension.md` | Blood-pressure categories, risk factors, and urgent situations |
| `stress_fatigue_general.md` | General information and practical guidance for fatigue/stress |

The Knowledge Base is intentionally used as a controlled source for patient-facing medical explanations. The system can also use MedlinePlus as a live fallback when the local curated documents do not provide enough context.

### Current Knowledge Base scope

The current KB focuses mainly on:

- Common symptoms and safety signs.
- Headache.
- Diabetes.
- Hypertension.
- Heart-attack warning signs.
- CBC interpretation.
- General fatigue/stress information.

It is **not** an exhaustive medical encyclopedia. More medical terminology, diseases, medications, imaging concepts, and laboratory tests can be added as the project grows.

---

## 5. Datasets

The project contains processed real-data files under:

```text
data/real_processed/
```

The project uses data sources including:

- **UCI Cleveland Heart Disease** data for heart-disease risk modeling.
- **NHANES** data for laboratory/blood-pressure related risk screening.
- **IU X-Ray / Open-i report text** for radiology-related experimentation and triage.

Raw/processed dataset files are kept under `data/real_raw/` and `data/real_processed/`.

---

## 6. Main Components

```text
app.py
    FastAPI application entry point

chat.html
    Browser-based chat interface

src/
    conversational_agent.py   Conversation orchestration
    nlu.py                    Intent/domain understanding
    general_wellness.py       General medical conversation + retrieval
    knowledge_base.py         Knowledge Base retrieval/answering
    safety_layer.py           Safety and emergency guardrails
    orchestrator.py           Main domain orchestration
    router.py                 Domain routing
    adaptive_questioner.py    Context-aware follow-up questions
    diagnostic_models.py      ML model training/loading utilities
    lab_triage_rules.py       CBC/pathology rule-based triage
    radiology_triage.py       Radiology rule-based triage
    radiology_extractor.py    Radiology information extraction
    live_medical_search.py    MedlinePlus fallback search
    vision_extraction.py      Image/report extraction utilities
```

---

## 7. Project Structure

```text
Diagnostic agent - fixed/
│
├── app.py
├── chat.html
├── requirements.txt
├── .env.example
│
├── data/
│   ├── knowledge_base/
│   ├── real_processed/
│   └── real_raw/
│
├── models/
│   ├── heart_disease_pipeline.joblib
│   ├── diabetes_risk_screening.joblib
│   └── hypertension_risk_screening.joblib
│
├── metrics/
│   ├── heart_disease_metrics.json
│   ├── diabetes_risk_screening_metrics.json
│   └── hypertension_risk_screening_metrics.json
│
├── schemas/
├── notebooks/
├── scripts/
├── src/
└── tests/
```

---

## 8. Installation

### 8.1 Create the virtual environment

From PowerShell inside the project directory:

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```

Verify that Python is coming from the project environment:

```powershell
python -c "import sys; print(sys.executable); print(sys.version)"
```

The executable should point to:

```text
...\Diagnostic agent - fixed\.venv\Scripts\python.exe
```

### 8.2 Install dependencies

```powershell
python -m pip install -r requirements.txt
```

Do not rely on `py -3.11 -m pip` after activating the environment; use:

```powershell
python -m pip ...
```

so the packages are installed into the active `.venv`.

---

## 9. Environment Variables

Copy:

```text
.env.example
```

to:

```text
.env
```

Then configure the provider/API settings required by the conversational layer.

Typical settings include the selected LLM provider and its API key/model.

**Never commit real API keys to GitHub.**

---

## 10. Run the Application

Make sure the virtual environment is activated:

```powershell
.venv\Scripts\Activate.ps1
```

Then start FastAPI:

```powershell
python -m uvicorn app:app --reload --port 8000
```

Open the application in the browser using:

```text
http://127.0.0.1:8000
```

If the project serves the chat page separately, open `chat.html` through the application as configured by `app.py`.

### Important

Use:

```powershell
python -m uvicorn app:app --reload --port 8000
```

rather than:

```powershell
py -3.11 -m uvicorn app:app --reload --port 8000
```

when you are working inside the activated `.venv`.

---

## 11. Run Tests

From the project root:

```powershell
pytest -q
```

Current project test status:

```text
58 passed
```

The tests cover areas including:

- Conversational-agent behavior.
- Orchestration and routing.
- Safety-layer behavior.
- Radiology triage and negation handling.

Warnings from serialized scikit-learn models can appear if the model is loaded with a different scikit-learn version from the version used during training. For reproducible deployment, keep the training and inference scikit-learn versions aligned.

---

## 12. Training / Retraining Models

The project contains training utilities under:

```text
scripts/
```

Main training entry point:

```powershell
python scripts/train_all.py
```

Risk-screening retraining utilities are also available in:

```text
scripts/retrain_risk_screening.py
```

The project deliberately keeps the deprecated leakage-prone models under:

```text
models/deprecated/
```

They are retained for audit/history and should not be presented as current model performance.

---

## 13. Medical Safety Design

The system is designed as a **patient-facing educational assistant**, not as an autonomous doctor.

It should:

- Explain medical concepts in accessible language.
- Ask relevant follow-up questions when more context is needed.
- Clearly distinguish education from diagnosis.
- Surface potentially urgent warning signs.
- Encourage professional medical evaluation when appropriate.
- Avoid inventing test results or diagnoses.

The system should not:

- Replace a physician's diagnosis.
- Independently prescribe or change medication.
- Treat model predictions as definitive diagnoses.
- Use historical/deprecated leakage-based metrics as evidence of performance.

---

## 14. Future Improvements

Planned improvements include:

- Expanding the curated medical Knowledge Base.
- Adding a dedicated medical terminology layer with Arabic/English explanations.
- Adding more laboratory reference explanations.
- Expanding radiology terminology and report interpretation support.
- Better Arabic/Egyptian Arabic intent recognition.
- Formal end-to-end latency benchmarking.
- External validation on unseen datasets.
- Calibration and threshold analysis for screening models.
- More extensive safety evaluation with clinically reviewed test cases.

---

## 15. Disclaimer

This project is a research/educational prototype for healthcare AI. Model metrics are dataset-specific and do not establish clinical safety, diagnostic accuracy, or regulatory approval. Any real-world clinical deployment requires appropriate clinical validation, monitoring, privacy/security controls, and qualified medical oversight.

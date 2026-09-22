# Diagnostic Models Workspace

> **⚠️ See `CHANGELOG.md` first.** The CBC, pathology, diabetes, and
> hypertension models were found to have label leakage (target computed
> directly from model input features, producing meaningless ~100% accuracy).
> They have been fixed — CBC/pathology are now rule-based, diabetes/
> hypertension are retrained as non-circular pre-lab risk screening models.
> The old models are preserved in `models/deprecated/` for audit purposes
> only and must not be used.

This project contains Jupyter notebooks and reusable Python code for diagnostic decision-support models:

- CBC triage — **rule-based** (`src/lab_triage_rules.py`)
- Pathology/lab panel triage — **rule-based** (`src/lab_triage_rules.py`)
- Radiology report model — **negation-aware rule engine** (`src/radiology_triage.py`); fixed a bug that misclassified ~42% of reports as urgent
- Heart disease risk model (ML, unaffected by the leakage issue)
- Diabetes risk **screening** model — pre-lab, non-circular ML (`models/diabetes_risk_screening.joblib`)
- Hypertension risk **screening** model — pre-lab, non-circular ML (`models/hypertension_risk_screening.joblib`)
- Chatbot router + LLM integration notebook

The notebooks and saved models now use real public datasets. Some targets are original dataset labels, while CBC, pathology, diabetes, hypertension, and radiology triage labels are derived from real measurements/report text using transparent clinical rules that still need clinician review before production.

## Run the FastAPI backend + chat UI

```bash
pip install -r requirements.txt
pip install fastapi uvicorn

# اختياري لكن مهم جداً: بدون ده، الفهم اللغوي هيبقى محدود جداً (fallback بسيط)
# PowerShell:
#   $env:GEMINI_API_KEY="your-key-here"      # أو
#   $env:OPENAI_API_KEY="your-key-here"

# شغّلي السيرفر
uvicorn app:app --reload --port 8000

# افتحي في المتصفح:
#   http://127.0.0.1:8000/docs        <- Swagger UI (تجربة كل endpoint تفاعلياً)
#   chat.html (افتحيه بـ Live Server، بيتكلم مع localhost:8000)
```

### محادثة حرة (النسخة الجديدة — زي شات عادي)

```bash
curl -X POST http://127.0.0.1:8000/chat/start -H "Content-Type: application/json" -d '{}'
# استخدمي session_id من الرد فى:
curl -X POST http://127.0.0.1:8000/chat/<SESSION_ID>/message \
     -H "Content-Type: application/json" -d '{"text": "عمري 45 وعندي ضغط 140 على 90"}'
```

⚠️ **من غير API key**، `src/nlu.py` بيستخدم fallback بسيط جداً (بيفهم أه/لأ
ورقم واحد بس، مش جمل معقدة). للفهم اللغوي الحقيقي، لازم `GEMINI_API_KEY` أو
`OPENAI_API_KEY` في environment variables قبل ما تشغّلي السيرفر.

### محادثة منظمة (النسخة القديمة — بتتحكمي فيها يدوياً حقل حقل)

```bash
# ابدئي محادثة
curl -X POST http://127.0.0.1:8000/session/start -H "Content-Type: application/json" \
     -d '{"domain": "cbc"}'

# جاوبي على السؤال (استخدمي session_id من الرد اللي فات)
curl -X POST http://127.0.0.1:8000/session/<SESSION_ID>/message \
     -H "Content-Type: application/json" \
     -d '{"answers": {"chest_pain": false, "severe_shortness_of_breath": false, "fainting_or_loss_of_consciousness": false, "uncontrolled_bleeding": false, "sudden_weakness_face_arm_speech": false}}'
```

⚠️ **مهم:** `chat.html` بيتكلم مع `localhost:8000` بس — يعني السيرفر لازم يكون شغال
على نفس الجهاز. لنشر حقيقي (مش لوكال)، محتاجة تستضيفي الـ FastAPI على سيرفر
سحابي (Render, Railway, AWS...) وتغيّري `API_BASE` في `chat.html` لعنوانه.

## Train / test the underlying models

```bash
pip install -r requirements.txt
pip install pytest  # for the test suite

# Trains heart_disease (and the currently-unused radiology pipeline) only.
# cbc/pathology need no training (rule-based). diabetes/hypertension are
# trained separately below to avoid accidentally regenerating the old
# leaky models.
python scripts/train_all.py

# Trains the non-circular diabetes/hypertension risk-screening models
python scripts/retrain_risk_screening.py

# Run the full test suite (58 tests: safety layer + orchestrator + radiology negation + conversational agent + hybrid RAG (curated + live))
python -m pytest tests/ -v

jupyter notebook notebooks
```

### Quick test without training anything

```python
import sys; sys.path.insert(0, ".")
from src.orchestrator import UnifiedDiagnosticAgent, AgentAction

agent = UnifiedDiagnosticAgent("cbc")  # or heart_disease, pathology,
                                        # diabetes_risk_screening, hypertension_risk_screening
response = agent.handle_turn({
    "chest_pain": False, "severe_shortness_of_breath": False,
    "fainting_or_loss_of_consciousness": False, "uncontrolled_bleeding": False,
    "sudden_weakness_face_arm_speech": False,
})
print(response.action, response.message)
# keep calling agent.handle_turn({feature: value}) with response.data["feature"]
# until response.action != AgentAction.ASK_QUESTION
```

## Outputs

Training writes:

- `data/real_raw/*`: downloaded real source tables
- `data/real_processed/*`: prepared real training tables
- `models/*_pipeline.joblib`: trained sklearn pipelines
- `metrics/*_metrics.json`: validation/test metrics
- `schemas/*_io_schema.json`: backend input/output contracts
- `schemas/sample_backend_payloads.json`: example request bodies

## Backend Contract

Each endpoint should accept the model-specific JSON schema and return:

```json
{
  "model_name": "cbc",
  "prediction": "follow_up",
  "probabilities": {"within_reference": 0.72, "follow_up": 0.21, "urgent_review": 0.07},
  "triage_action": "ask_follow_up_questions",
  "explanation": "Patient-friendly explanation for the chatbot layer",
  "safety_disclaimer": "This is educational decision support, not a diagnosis."
}
```

Model selection:

- Each notebook trains multiple candidate algorithms.
- The selected model is the candidate with the highest validation recall for the high-risk/urgent class.
- Tie-breakers are macro F1, then accuracy.

Recommended endpoint shape:

- `POST /predict/cbc`
- `POST /predict/pathology`
- `POST /predict/radiology`
- `POST /predict/heart-disease`
- `POST /predict/diabetes`
- `POST /predict/hypertension`

## LLM Integration

Use the ML model for structured risk triage, then send the result to an LLM for Arabic explanation, medical term definitions, and follow-up questions. For best quality use a strong hosted model; for Hugging Face/on-prem tests, start with `Qwen2.5-7B-Instruct` or `Llama-3.1-8B-Instruct`, then add RAG from approved medical content.

Important guardrails:

- Do not present model output as a final diagnosis.
- Escalate urgent symptoms such as severe chest pain, severe shortness of breath, fainting, stroke signs, major bleeding, or altered consciousness.
- Ask for missing context: age, sex, symptoms, duration, pregnancy status when relevant, medications, known diseases, and abnormal test reference ranges.

"""
Question Banks — كل موديل وطبيعته
=====================================
فيه نوعين مختلفين من الأسئلة هنا، ومهم نفرّق بينهم:

1. RISK_FACTOR_BANKS (heart_disease, diabetes_risk_screening, hypertension_risk_screening)
   → أسئلة عن *عوامل خطر* المريض يقدر يجاوب عليها من غير ما يكون عامل أي تحليل.
   → الترتيب بيتحدد ديناميكياً حسب أهمية الـ feature في الموديل نفسه.

2. LAB_PANEL_BANKS (cbc, pathology)
   → أسئلة عن *قيم تحليل فعلي* المريض لازم يكون عامله. مش "هل عندك خطر"،
     لكن "التحليل بتاعك طلع بكام". الترتيب هنا ثابت (مش ديناميكي حسب أهمية
     إحصائية) لأن القاعدة rule-based محتاجة كل القيم عشان تحكم صح، وبنبدأ
     بالقيم اللي بتحدد "urgent" الأول.

⚠️ ملاحظة أمانة: `hypertension_risk_screening` فيها `creatinine_mg_dl` وهو *برضه*
قيمة تحليل (مش عامل خطر بحت) — يعني مش كل أسئلته "قبل التحليل" فعلاً زي ما
كنا بنوصفها. اتعامل معاها هنا كسؤال **اختياري** (optional) بالظبط عشان كده —
لو المريض معندوش نتيجة حديثة، الـ agent يكمل من غيرها.
"""

# ---------------------------------------------------------------------------
# 1) Risk-factor question banks (dynamic ordering by model feature importance)
# ---------------------------------------------------------------------------

RISK_FACTOR_BANKS: dict[str, dict] = {
    "heart_disease": {
        "age": {"prompt": "كام عمرك؟", "type": "number"},
        "sex": {"prompt": "حضرتك راجل ولا ست؟", "type": "choice",
                "options": {"ست": "0.0", "راجل": "1.0"}},
        "cp": {"prompt": "لو باين عندك ألم في الصدر، هتوصفه إزاي؟", "type": "choice",
               "options": {"ألم نموذجي (ضغط/ثقل واضح)": "1.0",
                           "ألم غير نموذجي": "2.0",
                           "مش ألم صدر حقيقي (زي حرقان بسيط)": "3.0",
                           "مفيش ألم صدر خالص": "4.0"}},
        "trestbps": {"prompt": "لو عندك قياس حديث لضغط الدم وقت الراحة، كام (mmHg)؟", "type": "number"},
        "chol": {"prompt": "لو عندك تحليل كوليسترول حديث، كام (mg/dl)؟", "type": "number"},
        "fbs": {"prompt": "سكر الدم صايم عندك بيبقى أعلى من 120 عادةً؟", "type": "choice",
                "options": {"أه": "1.0", "لأ": "0.0"}},
        "restecg": {"prompt": "لو عندك تخطيط قلب حديث، النتيجة كانت طبيعية؟", "type": "choice",
                    "options": {"طبيعية": "0.0", "فيها ملاحظة بسيطة": "1.0", "فيها تضخم واضح": "2.0"}},
        "thalach": {"prompt": "وانتي بتعملي مجهود، بتحسي بضيق نفس أو خفقان بسرعة؟", "type": "choice",
                    "options": {"أه بسرعة (نبض بيوصل لأقصاه بسرعة)": "120",
                                "لأ، بتحمل مجهود عادي": "160"}},
        "exang": {"prompt": "بيجيلك ألم في الصدر لما تعملي مجهود؟", "type": "choice",
                  "options": {"أه": "1.0", "لأ": "0.0"}},
        "oldpeak": {"prompt": "(من تقرير ECG لو موجود) قيمة الـ ST depression كانت كام؟",
                    "type": "number"},
        "slope": {"prompt": "(من تقرير ECG لو موجود) ميل الـ ST segment وقت المجهود كان إزاي؟",
                  "type": "choice", "options": {"صاعد": "1.0", "مستوي": "2.0", "هابط": "3.0"}},
        "ca": {"prompt": "(فحص فلوروسكوبي لو اتعمل) عدد الأوعية الملوّنة؟", "type": "number"},
        "thal": {"prompt": "(فحص Thallium لو اتعمل) النتيجة كانت إيه؟", "type": "choice",
                 "options": {"طبيعي": "3.0", "عيب ثابت": "6.0", "عيب قابل للعكس": "7.0"}},
    },
    "diabetes_risk_screening": {
        "age": {"prompt": "كام عمرك؟", "type": "number"},
        "sex": {"prompt": "حضرتك راجل ولا ست؟", "type": "choice",
                "options": {"ست": "female", "راجل": "male"}},
        "bmi": {"prompt": "طولك ووزنك كام؟ (هنحسب الـ BMI منهم)", "type": "bmi_pair"},
        "systolic_bp": {"prompt": "لو عندك قياس ضغط حديث، الرقم الكبير كان كام؟", "type": "number"},
        "diastolic_bp": {"prompt": "والرقم الصغير كان كام؟", "type": "number"},
        "total_cholesterol": {"prompt": "لو عندك تحليل كوليسترول حديث، كان كام؟", "type": "number"},
    },
    "hypertension_risk_screening": {
        "age": {"prompt": "كام عمرك؟", "type": "number"},
        "sex": {"prompt": "حضرتك راجل ولا ست؟", "type": "choice",
                "options": {"ست": "female", "راجل": "male"}},
        "bmi": {"prompt": "طولك ووزنك كام؟ (هنحسب الـ BMI منهم)", "type": "bmi_pair"},
        "total_cholesterol": {"prompt": "لو عندك تحليل كوليسترول حديث، كان كام؟", "type": "number"},
        "creatinine_mg_dl": {
            "prompt": "لو عندك تحليل كرياتينين حديث (وظائف كلى)، كان كام؟ لو معندكش، قوليلي 'معرفش' ونكمل من غيره.",
            "type": "number", "optional": True,
        },
    },
}

# BMI مركّبة من طول ووزن — بتتفكّ لسؤالين فعليين
BMI_PAIR_QUESTIONS = {
    "height_cm": {"prompt": "طولك كام بالسنتيمتر؟", "type": "number"},
    "weight_kg": {"prompt": "وزنك كام بالكيلو؟", "type": "number"},
}


# ---------------------------------------------------------------------------
# 2) Lab-panel question banks (fixed order: urgent-determining fields first)
# ---------------------------------------------------------------------------

LAB_PANEL_BANKS: dict[str, list[dict]] = {
    "cbc": [
        {"field": "hemoglobin_g_dl", "prompt": "قيمة الهيموجلوبين (Hemoglobin) في تحليل CBC كام؟"},
        {"field": "wbc_10e3_ul", "prompt": "قيمة كرات الدم البيضاء (WBC) كام؟"},
        {"field": "platelets_10e3_ul", "prompt": "قيمة الصفائح الدموية (Platelets) كام؟"},
        {"field": "mcv_fl", "prompt": "قيمة متوسط حجم الكرة الحمراء (MCV) كام؟", "optional": True},
        {"field": "neutrophils_pct", "prompt": "نسبة العدلات (Neutrophils %) كام؟", "optional": True},
    ],
    "pathology": [
        {"field": "fasting_glucose", "prompt": "قيمة السكر الصايم (Fasting Glucose) كام؟"},
        {"field": "hba1c", "prompt": "قيمة السكر التراكمي (HbA1c) كام؟", "optional": True},
        {"field": "creatinine_mg_dl", "prompt": "قيمة الكرياتينين (وظائف كلى) كام؟"},
        {"field": "alt_u_l", "prompt": "قيمة إنزيم ALT (وظائف كبد) كام؟"},
        {"field": "ast_u_l", "prompt": "قيمة إنزيم AST (وظائف كبد) كام؟"},
        {"field": "total_cholesterol", "prompt": "قيمة الكوليسترول الكلي كام؟", "optional": True},
    ],
}

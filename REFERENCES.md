# References — مصادر القواعد والعتبات الطبية

> **ملاحظة أمانة:** الملف ده كتب بعد ما اتلاحظ إن المشروع مفهوش أي توثيق لمصادر
> علمية للعتبات (thresholds) المستخدمة. اللي هنا نتيجة بحث سريع، **مش مراجعة
> طبية كاملة** — لازم طبيب متخصص (قلب، باثولوجي، أشعة) يراجعها قبل أي استخدام
> حقيقي، بالظبط زي التحذيرات في `CHANGELOG.md`.

## `src/safety_layer.py` — أعراض الطوارئ

| القاعدة | المصدر | ملاحظة |
|---|---|---|
| ألم صدر + تعرّق/امتداد للذراع | American Heart Association — Heart Attack Warning Signs | من أشهر المعايير المعروفة عالمياً لأعراض الأزمة القلبية |
| ضيق نفس حاد | معرفة طبية عامة (علامة طوارئ قياسية) | يحتاج توثيق أدق (مثلاً NEWS2 score للـ vital signs) |
| علامات سكتة دماغية (وجه/ذراع/كلام) | FAST (Face-Arm-Speech-Time) — American Stroke Association | معيار قياسي معروف عالمياً |
| نزيف غير متحكم فيه | معرفة طبية عامة | يحتاج threshold أدق (كمية الدم، مدة الاستمرار) |

## `src/lab_triage_rules.py` — CBC & Pathology

| القيمة | العتبة المستخدمة | مصادر مرجعية حقيقية | ملاحظة |
|---|---|---|---|
| Hemoglobin (urgent) | < 8 g/dL | Labcorp critical values: ≤7.0؛ OHSU: <6.6 | **إحنا أكثر تحفظاً عمداً** — الهدف "روحي دكتور" مش "اتصال طوارئ فوري للمعمل"، فناخد هامش أمان أوسع |
| WBC (urgent) | <2 أو >20 K/uL | OHSU: <2.0 أو >40.0 | حد أعلى أكثر تحفظاً من المعيار المعملي |
| Platelets (urgent) | <50 أو >800 K/uL | Labcorp/OHSU: ≤10-30 (critical) | هامش أمان أوسع بكتير — starter threshold محتاج ضبط من طبيب |
| Fasting glucose (follow-up) | ≥126 mg/dL | ADA Standards of Care — Diagnostic criteria for diabetes | معيار عالمي معتمد وموثّق كويس |
| HbA1c (follow-up) | ≥6.5% | ADA Standards of Care | نفس المصدر |

**مصادر:** [Labcorp Critical Values](https://www.labcorp.com/test-menu/resources/critical-values),
[OHSU CBC with Differential](https://www.ohsu.edu/lab-services/cbc-differential)

## `src/radiology_triage.py` — تقارير الأشعة

قائمة المصطلحات (`URGENT_TERMS`, `FOLLOW_UP_TERMS`) مبنية على **معرفة طبية عامة
لمصطلحات الأشعة الشائعة**، مش من مصدر أكاديمي واحد موثّق. ده أضعف جزء من ناحية
التوثيق العلمي في المشروع كله — يستاهل مراجعة كاملة من طبيب أشعة يحدد:
- هل قائمة المصطلحات شاملة كفاية؟ (فيه احتمال مصطلحات مهمة ناقصة)
- هل التصنيف (urgent/follow_up) صح طبياً لكل مصطلح؟

## `scripts/retrain_risk_screening.py` — Diabetes & Hypertension

| الحالة | العتبة المستخدمة في الـ label | المعيار الأحدث | ⚠️ |
|---|---|---|---|
| Diabetes | Fasting glucose ≥126 أو HbA1c ≥6.5 | مطابق لـ **ADA Standards of Care** (لسه معتمد) | ✅ محدّث |
| Hypertension | Systolic ≥140 أو Diastolic ≥90 | **ACC/AHA 2017: ≥130/80** | ❌ **قديم — يحتاج تحديث** |

**اكتشاف مهم أثناء المراجعة:** الـ dataset الأصلي (NHANES) بنى `hypertension_label`
على معيار **JNC7/JNC8 القديم (140/90)**. المعيار الحالي المعتمد من **ACC/AHA
2017 هو 130/80 mmHg** — فرق كبير جداً (بيغيّر نسبة انتشار المرض في العينة من
~32% لـ ~46% من السكان حسب أي معيار بتستخدمي). **الموديل الحالي محتاج
إعادة تدريب بمعيار 130/80** في جلسة قادمة — مش اتصلح في الجلسة دي.

**مصادر:**
[2017 ACC/AHA Hypertension Guideline (Cleveland Clinic Journal of Medicine)](https://www.ccjm.org/content/85/10/771),
[ADA Standards of Medical Care in Diabetes](https://diabetes.org/about-diabetes/diagnosis)

## خلاصة: إيه اللي محتاج مراجعة طبية فورية قبل أي استخدام حقيقي

1. **كل عتبات `safety_layer.py`** — احتمال ناقصة أعراض حرجة تانية
2. **قائمة مصطلحات radiology** — أضعف نقطة توثيقاً في المشروع
3. **hypertension threshold** — لازم يتحدّث لـ 130/80 (ACC/AHA 2017)
4. **CBC/pathology thresholds** — الاتجاه صح (متحفظ عمداً) لكن الأرقام الدقيقة محتاجة ضبط من معملي/باثولوجي

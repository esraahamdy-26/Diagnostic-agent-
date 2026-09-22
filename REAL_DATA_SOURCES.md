# Real Dataset Sources

These real public datasets are used by the current notebooks and training scripts.

## Best Practical Mapping

| Model | Recommended real source | Why |
|---|---|---|
| CBC | CDC NHANES CBC files | Public population CBC values with codebooks and lab QA notes |
| Pathology | CDC NHANES Standard Biochemistry Profile, CRP, ESR where available | Public lab chemistry values for liver/kidney/metabolic screening |
| Radiology | Indiana University Open-i chest X-ray reports for report text; NIH ChestX-ray14 for image labels; MIMIC-CXR for larger credentialed research | Real radiology text/images, but licensing and access differ |
| Heart disease | UCI Heart Disease Cleveland dataset | Classic labeled tabular benchmark with clinical variables |
| Diabetes | NHANES glucose/HbA1c/diabetes questionnaire or Kaggle Pima for a small benchmark | NHANES is broader; Pima is simple but demographically narrow |
| Hypertension | NHANES blood pressure + demographics + body measures | Public measured BP data, suitable for hypertension labels |

## Public Links Checked

- NHANES 2021-2023 overview: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023
- NHANES CBC 2021-2023: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/CBC_L.htm
- NHANES Laboratory data page: https://wwwn.cdc.gov/nchs/nhanes/search/DataPage.aspx?Component=Laboratory&Cycle=2021-2023
- NHANES Demographics 2021-2023: https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics&Cycle=2021-2023
- NHANES Blood Pressure examination data: https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=
- NHANES Fasting Glucose 2021-2023: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GLU_L.htm
- NHANES Glycohemoglobin 2021-2023: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.htm
- NHANES Total Cholesterol 2021-2023: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TCHOL_L.htm
- NHANES Standard Biochemistry Profile 2021-2023: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BIOPRO_L.htm
- UCI Heart Disease: https://archive.ics.uci.edu/dataset/45/heart+disease
- Kaggle Pima Indians Diabetes: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- Open-i Indiana University chest X-rays/reports FAQ: https://openi-vip.nlm.nih.gov/faq
- NIH ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC
- MIMIC-CXR: https://physionet.org/content/mimic-cxr/

## Access Notes

- NHANES files are XPT/SAS transport files and can be loaded with `pandas.read_sas(url, format="xport")`.
- NHANES uses `SEQN` as the participant key, so CBC, demographics, glucose, HbA1c, cholesterol, body measures, and blood pressure can be merged.
- UCI Heart Disease is small but labeled and good for a first real model.
- Kaggle datasets need a Kaggle account/API token for command-line download.
- Open-i radiology reports are usable for research, but check the license before redistribution.
- MIMIC-CXR requires PhysioNet credentialing/training and a data use agreement.

## Production Warning

Real clinical data still needs clinical label design. For example, NHANES does not directly say "urgent CBC case"; you must define medically reviewed labels from reference ranges, disease history, symptoms, or follow-up outcomes. Do not deploy these models for diagnosis without clinician validation.

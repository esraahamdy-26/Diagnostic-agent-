from __future__ import annotations

from pathlib import Path
import json
import urllib.request

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "real_raw"
PROCESSED = ROOT / "data" / "real_processed"

NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles"

NHANES_FILES = {
    "demographics": f"{NHANES_BASE}/DEMO_L.xpt",
    "cbc": f"{NHANES_BASE}/CBC_L.xpt",
    "body_measures": f"{NHANES_BASE}/BMX_L.xpt",
    "blood_pressure": f"{NHANES_BASE}/BPXO_L.xpt",
    "glucose": f"{NHANES_BASE}/GLU_L.xpt",
    "hba1c": f"{NHANES_BASE}/GHB_L.xpt",
    "cholesterol": f"{NHANES_BASE}/TCHOL_L.xpt",
    "biochemistry": f"{NHANES_BASE}/BIOPRO_L.xpt",
}

UCI_HEART_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
IU_XRAY_URLS = {
    "train": "https://huggingface.co/datasets/dz-osamu/IU-Xray/resolve/main/train.jsonl",
    "val": "https://huggingface.co/datasets/dz-osamu/IU-Xray/resolve/main/val.jsonl",
    "test": "https://huggingface.co/datasets/dz-osamu/IU-Xray/resolve/main/test.jsonl",
}
UCI_HEART_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]


def read_xpt(url: str) -> pd.DataFrame:
    return pd.read_sas(url, format="xport")


def download_nhanes() -> dict[str, pd.DataFrame]:
    RAW.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, url in NHANES_FILES.items():
        print(f"Downloading {name}: {url}")
        df = read_xpt(url)
        tables[name] = df
        df.to_csv(RAW / f"nhanes_{name}.csv", index=False)
    return tables


def build_nhanes_model_tables(tables: dict[str, pd.DataFrame]) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    demo = tables["demographics"][["SEQN", "RIDAGEYR", "RIAGENDR"]].rename(
        columns={"RIDAGEYR": "age", "RIAGENDR": "sex_code"}
    )
    demo["sex"] = demo["sex_code"].map({1: "male", 2: "female"})

    cbc = tables["cbc"].merge(demo, on="SEQN", how="left")
    cbc = cbc.rename(
        columns={
            "LBXHGB": "hemoglobin_g_dl",
            "LBXWBCSI": "wbc_10e3_ul",
            "LBXPLTSI": "platelets_10e3_ul",
            "LBXRBCSI": "rbc_10e6_ul",
            "LBXMCVSI": "mcv_fl",
            "LBXMCHSI": "mch_pg",
            "LBXNEPCT": "neutrophils_pct",
            "LBXLYPCT": "lymphocytes_pct",
        }
    )
    cbc_cols = [
        "age",
        "sex",
        "hemoglobin_g_dl",
        "wbc_10e3_ul",
        "platelets_10e3_ul",
        "rbc_10e6_ul",
        "mcv_fl",
        "mch_pg",
        "neutrophils_pct",
        "lymphocytes_pct",
    ]
    cbc[cbc_cols].to_csv(PROCESSED / "cbc_nhanes_real.csv", index=False)

    merged = demo
    for key in ["body_measures", "blood_pressure", "glucose", "hba1c", "cholesterol", "biochemistry"]:
        merged = merged.merge(tables[key], on="SEQN", how="left")

    # Derived labels for research experiments only. Get clinician review before use.
    merged["diabetes_label"] = ((merged["LBXGLU"] >= 126) | (merged["LBXGH"] >= 6.5)).astype(int)
    merged["hypertension_label"] = ((merged["BPXOSY1"] >= 140) | (merged["BPXODI1"] >= 90)).astype(int)

    chronic_cols = [
        "age",
        "sex",
        "BMXBMI",
        "LBXGLU",
        "LBXGH",
        "LBXTC",
        "BPXOSY1",
        "BPXODI1",
        "LBXSCR",
        "LBXSATSI",
        "LBXSASSI",
        "diabetes_label",
        "hypertension_label",
    ]
    available_cols = [col for col in chronic_cols if col in merged.columns]
    merged[available_cols].rename(
        columns={
            "BMXBMI": "bmi",
            "LBXGLU": "fasting_glucose",
            "LBXGH": "hba1c",
            "LBXTC": "total_cholesterol",
            "BPXOSY1": "systolic_bp",
            "BPXODI1": "diastolic_bp",
            "LBXSCR": "creatinine_mg_dl",
            "LBXSATSI": "alt_u_l",
            "LBXSASSI": "ast_u_l",
        }
    ).to_csv(PROCESSED / "nhanes_chronic_labs_real.csv", index=False)


def download_uci_heart() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    heart = pd.read_csv(UCI_HEART_URL, header=None, names=UCI_HEART_COLUMNS, na_values="?")
    heart["heart_disease_label"] = (heart["num"] > 0).astype(int)
    heart.to_csv(RAW / "uci_heart_processed_cleveland.csv", index=False)
    heart.to_csv(PROCESSED / "heart_disease_uci_real.csv", index=False)
    print("Saved UCI Heart Disease dataset")


def download_iu_xray_reports() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, url in IU_XRAY_URLS.items():
        target = RAW / f"iu_xray_{split}.jsonl"
        print(f"Downloading IU-Xray {split}: {url}")
        urllib.request.urlretrieve(url, target)
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                rows.append(
                    {
                        "split": split,
                        "report_text": record.get("response", ""),
                        "source_id": ",".join(record.get("images", [])),
                    }
                )
    pd.DataFrame(rows).to_csv(PROCESSED / "iu_xray_reports_real.csv", index=False)
    print("Saved IU-Xray radiology reports")


if __name__ == "__main__":
    nhanes_tables = download_nhanes()
    build_nhanes_model_tables(nhanes_tables)
    download_uci_heart()
    download_iu_xray_reports()
    print(f"Done. Raw files: {RAW}")
    print(f"Processed files: {PROCESSED}")

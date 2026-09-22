from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.diagnostic_models import save_sample_payloads, train_all


if __name__ == "__main__":
    summary = train_all(ROOT)
    save_sample_payloads(ROOT)
    print(summary.to_string(index=False))

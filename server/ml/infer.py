from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "ml_artifacts" / "model_manifest.json"
MODEL_PATH = ROOT / "ml_artifacts" / "models" / "flight_price_model.joblib"
FEATURES = ["airline", "flight", "source_city", "departure_time", "stops", "arrival_time", "destination_city", "class", "duration", "days_left"]


def main() -> None:
    payload = json.loads(sys.stdin.read())
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    missing = [feature for feature in FEATURES if feature not in payload]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")
    frame = pd.DataFrame([{feature: payload[feature] for feature in FEATURES}])
    model = joblib.load(MODEL_PATH)
    price = float(model.predict(frame)[0])
    print(json.dumps({"predictedPrice": price, "model": manifest}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        raise SystemExit(1)

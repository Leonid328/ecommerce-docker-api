#!/usr/bin/env python3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

RNG = np.random.default_rng(7)
N = 600
df = pd.DataFrame(
    {
        "recency": RNG.uniform(1, 90, N),
        "frequency": RNG.integers(1, 40, N),
        "monetary": RNG.uniform(10, 800, N),
        "tenure_days": RNG.uniform(10, 730, N),
    }
)
y = ((df["frequency"] > 8) & (df["monetary"] > 120) & (df["recency"] < 30)).astype(int)
model = RandomForestClassifier(n_estimators=40, random_state=7)
model.fit(df, y)
out = Path("models/recommendation_model.pkl")
out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, out)
print(f"Wrote {out}")

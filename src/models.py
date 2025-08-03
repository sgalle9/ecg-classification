from typing import Any

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_model(name: str, params: dict[str, Any], random_state: int):
    if name == "XGBoost":
        return XGBClassifier(random_state=random_state, **params)
    elif name == "RandomForest":
        return RandomForestClassifier(random_state=random_state, **params)
    else:
        raise ValueError(f"Model '{name}' not recognized.")

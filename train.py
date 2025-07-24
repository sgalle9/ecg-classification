import json
from pathlib import Path

import joblib
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.dataset import get_dataset
from src.models import get_model
from src.transforms import get_preprocessor
from src.utils.config_utils import handle_reproducibility, load_config


def train():
    config = load_config()
    rs_numpy = handle_reproducibility(config["seed"])

    # ==================================================
    # Data Preparation
    # ==================================================
    X, y = get_dataset(**config["dataset"])

    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=config["seed"], stratify=y)

    # Preprocessor
    preprocessor = get_preprocessor(config["preprocessor"])

    # ==================================================
    # Model initialization
    # ==================================================
    model = get_model(random_state=config["seed"], **config["model"])

    # Complete pipeline with model
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # ==================================================
    # Training Preparation
    # ==================================================
    # Label encoding
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Compute sample weights for handling class imbalance
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # ==================================================
    # Training Process
    # ==================================================
    pipeline.fit(X_train, y_train, model__sample_weight=sample_weights, **config["fit_params"])

    # Save the entire pipeline object and the label encoder
    path = Path(f"{config['output_dir']}/{config['experiment']}/{config['run_id']}")
    with open(path / "pipeline.joblib", "wb") as f:
        joblib.dump(pipeline, f, protocol=5)

    with open(path / "label_encoder.joblib", "wb") as f:
        joblib.dump(le, f, protocol=5)

    # ==================================================
    # Testing Process
    # ==================================================
    y_train_pred = pipeline.predict(X_train)
    score_train = f1_score(y_train, y_train_pred, average="macro")
    print(f"F1 train set: {score_train}")

    y_test_pred = pipeline.predict(X_test)
    score_test = f1_score(y_test, y_test_pred, average="macro")
    print(f"F1 test set: {score_test}")

    # Log metrics
    with open(path / "metrics.json", "w") as f:
        json.dump({"f1_macro_train": score_train, "f1_macro_test": score_test}, f, indent=4)


if __name__ == "__main__":
    train()

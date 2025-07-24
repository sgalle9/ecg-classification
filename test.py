from pathlib import Path

import joblib
import pandas as pd

from src.dataset import get_dataset
from src.utils.config_utils import load_config


def test():
    config = load_config()

    # ==================================================
    # Data Preparation
    # ==================================================
    X = get_dataset(**config["dataset"])

    # Store record names from the DataFrame index before processing
    record_names = X.index

    # ==================================================
    # Model initialization
    # ==================================================
    model_dir = Path(config["model_dir"])
    with open(model_dir / "pipeline.joblib", "rb") as f:
        pipeline = joblib.load(f)

    with open(model_dir / "label_encoder.joblib", "rb") as f:
        le = joblib.load(f)

    # ==================================================
    # Prediction Process
    # ==================================================
    # Get numeric predictions from the pipeline
    predictions = pipeline.predict(X)

    # Convert numeric predictions back to original class labels
    predictions = le.inverse_transform(predictions)

    # Save the predictions to a CSV file
    predictions_df = pd.DataFrame({"record_name": record_names, "prediction": predictions})
    output_path = Path(config["output_dir"]) / "predictions.csv"
    predictions_df.to_csv(output_path, **config["to_csv_params"])


if __name__ == "__main__":
    test()

# Atrial Fibrillation Classification from ECGs

This project focuses on the classification of single-lead electrocardiogram (ECG) recordings. The recordings are categorized into one of four classes: normal sinus rhythm (`N`), atrial fibrillation (`A`), other rhythm (`O`), or too noisy to be classified (`~`).

An XGBoost model was employed for this classification task. The model utilizes features that were extracted from the raw ECG signals.

Training and evaluation were performed using the [2017 PhysioNet/CinC Challenge](https://physionet.org/content/challenge-2017/1.0.0/) dataset.


## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/sgalle9/ecg-classification.git
    cd ecg-classification
    ```

2. **Build the Docker image:**
    ```bash
    docker build -t ecg .
    ```

3.  **Download and prepare the train data:**

    ```bash
    wget -O training2017.zip "https://physionet.org/files/challenge-2017/1.0.0/training2017.zip?download"
    unzip -j training2017.zip -d data/train/
    rm training2017.zip
    ```


## Usage

The `data`, `configs`, and `outputs` directories on your local machine are mounted as volumes into the Docker container. This allows you to edit configuration files and access model outputs directly from your host machine.

### Training

1.  **Configure:** Modify the training parameters in `configs/train.yaml`.

2.  **Run training:**

    ```bash
    docker run --rm \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/outputs:/app/outputs \
        -v $(pwd)/configs:/app/configs \
        ecg \
        python train.py -c configs/train.yaml
    ```

    The following outputs will be saved to `<output_dir>/<experiment>/<run_id>`:
    - `config.yaml`: A copy of the configuration used for the run.
    - `pipeline.joblib`: The trained model pipeline.
    - `label_encoder.joblib`: The fitted label encoder.
    - `metrics.json`: A JSON file with performance metrics.

### Testing

1.  **Add test data:** Place the signal files (`.hea` and `.mat`) you want to classify into the `./data/test/` directory.

2.  **Configure:** Modify `configs/test.yaml`. You must update `model_dir` to point to the training output directory you want to use for testing.
    ```yaml
    # In configs/test.yaml
    model_dir: "./outputs/my_experiment/xgboost"
    ```

2.  **Run testing:**

    ```bash
    docker run --rm \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/outputs:/app/outputs \
        -v $(pwd)/configs:/app/configs \
        ecg \
        python test.py -c configs/test.yaml
    ```

    Predictions will be saved as `predictions.csv` in the location specified by `output_dir` in your `test.yaml`.

# PMLDL Assignment 1: Deployment

An automated MLOps pipeline that cleans data, trains a model and deploys it as
an API with a web application. Orchestrated by Airflow, runs every 5 minutes.

**Dataset:** California Housing (original StatLib version — 20,640 districts,
207 missing values, one categorical feature).
**Task:** regression — predict the median house value of a district.
**Model:** Random Forest, R² ≈ 0.78 on the test set.

## Pipeline

```
Airflow DAG "pmldl_pipeline", schedule */5 * * * *

  data_engineering        model_engineering             deployment
  ────────────────        ─────────────────             ──────────
  load raw CSV            build 3 derived features      build API image
  drop duplicates         one-hot encode category       build app image
  impute missing          train Random Forest           docker compose up
  drop censored target    evaluate on test set
  trim outliers (IQR)     log metrics to MLflow
  split 80 / 20           save model.pkl

  data/processed/*.csv    models/model.pkl              two containers
```

## Repository structure

```
code/
  config.py                  paths and cleaning constants
  features.py                column names and feature engineering
  datasets/
    download_data.py         fetch raw data (run once)
    prepare_data.py          Stage 1
  models/
    train_model.py           Stage 2
  deployment/
    api/                     FastAPI service + Dockerfile
    app/                     Streamlit app + Dockerfile
    docker-compose.yml       Stage 3
data/
  raw/housing.csv            pipeline input (committed)
  processed/                 train.csv, test.csv (generated)
models/                      model.pkl, metrics.json (generated)
services/airflow/dags/       the DAG
requirements.txt
```

`features.py` is imported by both the training script and the API, so the
feature formulas cannot drift between training and serving.

## Requirements

- Python 3.13
- Docker Desktop (running)

## Setup

```bash
git clone https://github.com/aniksel/pmldl-a1-deployment.git
cd pmldl-a1-deployment

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.3.1/constraints-3.13.txt"
```

The constraints file is required: Airflow pins the versions of its ~600
dependencies, and `requirements.txt` follows those pins.

The raw data is already in `data/raw/housing.csv`, so nothing else is needed.
The script that fetched it is kept for reference and overwrites the file:

```bash
python code/datasets/download_data.py
```

## Running the pipeline manually

`models/model.pkl` is **not** committed to the repository: it is an output of
Stage 2 and is regenerated on every pipeline run. Run the stages in order — the
Docker build of the API copies that file into the image, so Stage 3 fails if
Stage 2 has not run yet.

```bash
python code/datasets/prepare_data.py     # Stage 1
python code/models/train_model.py        # Stage 2 -> creates models/model.pkl

cd code/deployment
docker compose up --build                # Stage 3
```

- Web application: http://localhost:8501
- API docs: http://localhost:8000/docs

## Running the pipeline automatically

From the project root, with the virtual environment active:

```bash
source .venv/bin/activate

export AIRFLOW_HOME=$(pwd)/services/airflow
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export NO_PROXY="*" 

airflow standalone
```

Airflow UI: http://localhost:8080. The generated admin password is printed in
the startup log and stored in
`services/airflow/simple_auth_manager_passwords.json.generated`.

Enable the `pmldl_pipeline` DAG in the UI — it then runs every 5 minutes on its
own, ending each run with the two containers up to date. The DAG calls the
scripts through `.venv/bin/python`, so keep the virtual environment in `.venv`.

## Viewing the experiment tracking

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

MLflow UI: http://localhost:5001 (port 5000 is taken by AirPlay on macOS).

## Ports

| Service | Port |
|---|---|
| Streamlit app | 8501 |
| Model API | 8000 |
| Airflow UI | 8080 |
| MLflow UI | 5001 |

## Notes on the data

- `median_house_value` is censored: every district above $500,000 was recorded
  as 500001. Those 992 rows are dropped — they carry no price information.
- `total_bedrooms` has 207 missing values, imputed with the column median.
- Count-like columns have extreme tails (one district reports 39,320 rooms).
  They are trimmed with the IQR rule.
- 16,959 of 20,640 rows survive cleaning (82%).

## Version pinning

`scikit-learn`, `numpy` and `joblib` are pinned to the same exact versions in
`requirements.txt` and in `code/deployment/api/requirements.txt`. The model is
trained in the virtual environment and unpickled inside the container, so a
version mismatch there would break predictions silently.

# стадия 2: признаки, обучение, оценка, сохранение модели
# data/processed/*.csv -> models/model.pkl + метрики в MLflow

import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    METRICS_FILE,
    MLFLOW_ARTIFACTS,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_FILE,
    RANDOM_STATE,
    TEST_FILE,
    TRAIN_FILE,
)
from features import CATEGORICAL, MODEL_NUMERIC, TARGET, add_features

N_ESTIMATORS = 50
MAX_DEPTH = 15

train = add_features(pd.read_csv(TRAIN_FILE))
test = add_features(pd.read_csv(TEST_FILE))

columns = MODEL_NUMERIC + CATEGORICAL
X_train, y_train = train[columns], train[TARGET]
X_test, y_test = test[columns], test[TARGET]

# деревьям масштабирование не нужно, кодируем только категорию
preprocessor = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
    remainder="passthrough",
)

model = Pipeline(
    [
        ("prep", preprocessor),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
    mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACTS)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

with mlflow.start_run():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "rmse": root_mean_squared_error(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "r2": r2_score(y_test, pred),
    }

    mlflow.log_params({
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    })
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, name="model")

MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
# сжатие: без него файл весит 100 МБ вместо 6
joblib.dump(model, MODEL_FILE, compress=3)
METRICS_FILE.write_text(json.dumps(metrics, indent=2))

print("RMSE:", round(metrics["rmse"], 2))
print("MAE:", round(metrics["mae"], 2))
print("R2:", round(metrics["r2"], 4))
print("Model saved to", MODEL_FILE)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "housing.csv"
TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "model.pkl"
METRICS_FILE = PROJECT_ROOT / "models" / "metrics.json"

DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

MLFLOW_TRACKING_URI = "sqlite:///" + str(PROJECT_ROOT / "mlflow.db")
MLFLOW_ARTIFACTS = (PROJECT_ROOT / "mlartifacts").as_uri()
MLFLOW_EXPERIMENT = "california-housing"

# все дома дороже 500000 записаны одним и тем же числом, поэтому их убираем
TARGET_CAP = 500000

# колонки с длинным правым хвостом, режем их по IQR
OUTLIER_COLUMNS = ["total_rooms", "total_bedrooms", "population", "households", "median_income"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

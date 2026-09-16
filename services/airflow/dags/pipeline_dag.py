# весь пайплайн: данные -> модель -> деплой, запуск каждые 5 минут

from datetime import datetime
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

# dags/ -> airflow/ -> services/ -> корень проекта
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
DEPLOYMENT_DIR = PROJECT_ROOT / "code" / "deployment"

with DAG(
    dag_id="pmldl_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="*/5 * * * *",
    catchup=False,
    # не запускать новый прогон, пока не закончился предыдущий
    max_active_runs=1,
    tags=["pmldl"],
) as dag:
    data_engineering = BashOperator(
        task_id="data_engineering",
        bash_command=f"{PYTHON} {PROJECT_ROOT}/code/datasets/prepare_data.py",
    )

    model_engineering = BashOperator(
        task_id="model_engineering",
        bash_command=f"{PYTHON} {PROJECT_ROOT}/code/models/train_model.py",
    )

    deployment = BashOperator(
        task_id="deployment",
        bash_command=f"cd {DEPLOYMENT_DIR} && docker compose up -d --build",
    )

    data_engineering >> model_engineering >> deployment

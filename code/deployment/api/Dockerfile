FROM python:3.13-slim

WORKDIR /app

COPY code/deployment/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code/features.py .
COPY code/deployment/api/main.py .
# модель кладётся в образ, поэтому образ пересобирается после каждого обучения
COPY models/model.pkl .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

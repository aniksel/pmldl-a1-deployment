FROM python:3.13-slim

WORKDIR /app

COPY code/deployment/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code/features.py .
COPY code/deployment/api/main.py .
# the model goes inside the image, so the image is rebuilt after every training
COPY models/model.pkl .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

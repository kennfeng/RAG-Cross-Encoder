FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/data/hf-cache
WORKDIR /app
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.1 && pip install -r requirements.txt
COPY . .
RUN chmod +x scripts/entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["/bin/sh", "scripts/entrypoint.sh"]

FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# git: needed to pip-install LightGlue from GitHub.
# libgl1/libglib2.0-0: common runtime deps for opencv even in headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements-docker.txt ./
COPY web/requirements.txt ./web/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements-docker.txt && pip install git+https://github.com/cvg/LightGlue.git
COPY . .
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "7860"]

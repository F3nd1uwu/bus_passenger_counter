FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# CPU is the default deployment. A GPU override needs compatible CUDA wheels.
RUN pip install --no-cache-dir --no-deps --index-url https://download.pytorch.org/whl/cpu \
    torch==2.15.0 torchvision==0.29.2
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 10001 counter \
    && mkdir -p /app/state /app/data /app/models \
    && chown -R counter:counter /app
COPY --chown=counter:counter . .
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 YOLO_AUTOINSTALL=false
USER counter
CMD ["python", "main.py", "--profile", "test2"]

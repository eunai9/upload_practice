# Base image
FROM python:3.11-slim

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
#RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "src/m6_practice/train.py"]
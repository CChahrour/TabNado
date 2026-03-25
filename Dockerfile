FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System packages required by scientific Python wheels and common build steps.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies before copying source so this layer is cached
# independently of code changes.
RUN pip install --upgrade pip
COPY pyproject.toml README.md ./
RUN mkdir -p src/tabnado && touch src/tabnado/__init__.py
RUN pip install . && pip install pytorch-tabular==1.2.0 --no-deps

COPY . /app
RUN pip install --no-deps .

# Default command runs the full pipeline; pass --help or custom args at runtime.
ENTRYPOINT ["tabnado-run"]

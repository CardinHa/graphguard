FROM python:3.11-slim

WORKDIR /app

# System deps for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Default: launch CLI help
ENTRYPOINT ["graphguard"]
CMD ["--help"]

# To launch API:  docker run -p 8000:8000 graphguard api
# To launch dashboard: docker run -p 8501:8501 graphguard dashboard

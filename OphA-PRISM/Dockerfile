# Python reference environment; R is executed separately on the institutional server.
FROM python:3.10-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/opha-prism
COPY requirements-core.txt requirements-models.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY ophaprism/ ophaprism/
COPY server_audit/ server_audit/
COPY config/ config/
COPY run_pipeline.py ./
ENTRYPOINT ["python", "run_pipeline.py"]
CMD ["--help"]

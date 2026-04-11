FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ARG ATST_SED_REF=3cc3ccfd57b5808d34ad9ef2e89d562f9c220ff9

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ATST_SED_REPO_DIR=/opt/atst-sed \
    PYTHONPATH=/app:/opt/atst-sed

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/Audio-WestlakeU/ATST-SED.git "${ATST_SED_REPO_DIR}" \
    && git -C "${ATST_SED_REPO_DIR}" checkout "${ATST_SED_REF}"

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY scripts /app/scripts

RUN chmod +x /app/scripts/start.sh

CMD ["/app/scripts/start.sh"]

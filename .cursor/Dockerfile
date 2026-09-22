FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Stable system dependencies. Python dependency installation itself happens in
# the environment `install` step so it tracks the checked-out lockfile/source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-venv \
        python3-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# The ubuntu:24.04 image ships a non-root `ubuntu` user (uid 1000).
USER ubuntu

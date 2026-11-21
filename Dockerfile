# syntax=docker/dockerfile:1

#
# Segment3D - AI-Powered 3D Model Segmentation Tool
# Multi-stage Dockerfile following 2024 best practices
#

# ==============================================================================
# Build Stage: Install dependencies and prepare environment
# ==============================================================================
FROM python:3.10-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxrender1 \
    libxext6 \
    libxft2 \
    libxss1 \
    libxi6 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libasound2t64 \
    libpangocairo-1.0-0 \
    libatk1.0-0t64 \
    libcairo-gobject2 \
    libgtk-3-0t64 \
    libgdk-pixbuf-2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Runtime Stage: Create minimal runtime environment
# ==============================================================================
FROM python:3.10-slim as runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    QT_X11_NO_MITSHM=1 \
    DISPLAY=:0

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential GUI libraries
    libgl1 \
    libglu1-mesa \
    libxrender1 \
    libxext6 \
    libxft2 \
    libxss1 \
    libxi6 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libasound2t64 \
    libpangocairo-1.0-0 \
    libatk1.0-0t64 \
    libcairo-gobject2 \
    libgtk-3-0t64 \
    libgdk-pixbuf-2.0-0 \
    # X11 utilities for display forwarding
    x11-apps \
    xauth \
    # Networking tools
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd -r segment3d && \
    useradd -r -g segment3d -d /app -s /bin/bash -c "Segment3D User" segment3d && \
    mkdir -p /app /app/data /app/output /app/temp && \
    chown -R segment3d:segment3d /app

# Set working directory
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=segment3d:segment3d src/ ./src/
COPY --chown=segment3d:segment3d data/ ./data/
COPY --chown=segment3d:segment3d scripts/run-gui-test.py ./scripts/

# Create necessary directories with proper permissions
RUN mkdir -p /app/output /app/temp && \
    chmod 755 /app/output /app/temp && \
    chown -R segment3d:segment3d /app/output /app/temp

# Switch to non-root user
USER segment3d

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; from src.modal_client import ModalSAMClient; sys.exit(0)" || exit 1

# Set default command
CMD ["python", "src/main.py"]

# ==============================================================================
# Development Stage: Additional development tools
# ==============================================================================
FROM runtime as development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    git \
    curl \
    htop \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install development Python packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    ipython \
    pytest \
    black \
    flake8 \
    mypy

USER segment3d

# Override CMD for development (keeps container running)
CMD ["tail", "-f", "/dev/null"] 
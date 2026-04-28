# syntax=docker/dockerfile:1.4

# ==============================================================================
# Dockerfile: Inference Runtime
# ==============================================================================
#
# Purpose:
#   Universal inference image:
#   - vLLM OpenAI server (serve mode)
#   - LoRA merge deps (peft/accelerate/etc) for merge jobs
#   - SSH server (for control-plane access + SSH tunnel)
#
# Used for both single_node (merge + serve containers) and RunPod Pods.
# The pipeline pulls this image from Docker Hub at deploy time.
#
# Build:
#   docker build -f docker/inference/Dockerfile -t ryotenkai/inference-vllm:v1.0 docker/inference
#
# ==============================================================================

# Build-time descriptors — passed in by ``build_and_push.sh`` so the
# tag stays a clean semver while the dependency profile (vLLM base
# version, PEFT, etc.) is introspectable via ``docker inspect`` and
# ``/opt/ryotenkai/version.txt``.
ARG IMAGE_VERSION="dev"
ARG VLLM_BASE_VERSION="v0.6.3.post1"

# Base: Official vLLM image (GPU-ready)
FROM vllm/vllm-openai:v0.6.3.post1

# Re-declare ARGs after FROM so they're available in the build-stage.
ARG IMAGE_VERSION
ARG VLLM_BASE_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps:
# - openssh-server: SSH access (RunPod)
# - curl/ca-certificates: health checks + TLS
# - git: optional (some HF flows rely on git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    curl \
    ca-certificates \
    git \
    tar \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# SSH setup: allow root login via injected keys (PUBLIC_KEY → authorized_keys in entrypoint)
RUN mkdir -p /var/run/sshd /root/.ssh && chmod 700 /root/.ssh \
    && sed -i 's/^#\\?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/^#\\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && sed -i 's/^#\\?PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config \
    && echo "ClientAliveInterval 60" >> /etc/ssh/sshd_config \
    && echo "ClientAliveCountMax 3" >> /etc/ssh/sshd_config

# Merge deps:
# Keep minimal to reduce risk of breaking vLLM dependency set.
# vLLM image already ships torch/transformers; we add PEFT tooling for merge script.
RUN python3 -m pip install --upgrade pip "setuptools<70.0.0" wheel \
    && python3 -m pip install --no-cache-dir \
      "peft>=0.10.0" \
      "accelerate>=0.29.0" \
      "safetensors>=0.4.0" \
      "huggingface-hub>=0.20.0"

# Copy merge script (optional convenience; control plane may still upload its own copy per run)
RUN mkdir -p /opt/helix
COPY scripts/merge_lora.py /opt/helix/merge_lora.py
RUN chmod +x /opt/helix/merge_lora.py

# Workspace contract
RUN mkdir -p /workspace
WORKDIR /workspace

# HF cache defaults (overridable at runtime)
ENV HF_HOME=/workspace/hf_cache \
    HUGGINGFACE_HUB_CACHE=/workspace/hf_cache \
    TRANSFORMERS_CACHE=/workspace/hf_cache

# Entrypoint: start sshd, then exec passed command
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Self-describing version manifest (mirrors training image).
RUN mkdir -p /opt/ryotenkai && \
    printf 'ryotenkai-inference-vllm %s\nvllm_base: %s\n' \
    "${IMAGE_VERSION}" "${VLLM_BASE_VERSION}" \
    > /opt/ryotenkai/version.txt

EXPOSE 22

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash", "-lc", "tail -f /dev/null"]

# OCI image-spec labels (https://github.com/opencontainers/image-spec).
# Tools like ``docker inspect``, GitHub Container Registry UI, and
# Trivy all read these — keep the dependency profile discoverable
# without unpacking the image.
LABEL org.opencontainers.image.title="ryotenkai-inference-vllm" \
      org.opencontainers.image.description="RyotenkAI inference runtime (vLLM + merge deps + SSH)" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.vendor="ryotenkai" \
      org.opencontainers.image.licenses="proprietary" \
      ryotenkai.vllm_base="${VLLM_BASE_VERSION}"

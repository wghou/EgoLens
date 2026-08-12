# syntax=docker/dockerfile:1.7
#
# Self-contained EgoLens training / inference image.
#
# Matches the stack validated in README.md: Python 3.11, PyTorch 2.5.1, CUDA 12.4.
# The source tree is cloned from GitHub during the build into /workspace/EgoLens
# and the SAM2 CUDA extension is compiled into the image, so nothing is copied
# from or mounted off the host. Datasets and checkpoints are out of scope here;
# put them under /workspace/EgoLens/{data,pretrained} at runtime however you like.
#
# Build (the build context is unused, any directory works):
#   docker build -f docker/Dockerfile -t egolens:latest .
#
# Useful build args:
#   --build-arg EGOLENS_REF=main                  # branch, tag or commit sha
#   --build-arg TORCH_CUDA_ARCH_LIST="8.9"        # RTX 4090
#   --build-arg TORCH_CUDA_ARCH_LIST="9.0+PTX"    # H100 / H200
#
# Layer caching will reuse an existing clone when the args are unchanged. To pull
# newer commits from the same ref, bump --build-arg EGOLENS_CACHEBUST=2 or pass
# --no-cache.
#
# Optional sshd, off unless ENABLE_SSH=1 is set at runtime. It refuses to start
# without a key or password, see the entrypoint below.
#
# The GUI dataset visualizer (visualizer/, needs PyQt5) is intentionally not
# installed; run it outside the container.

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG PYTHON_VERSION=3.11
ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG TORCH_CUDA_ARCH_LIST="8.9;9.0+PTX"
ARG FLASH_ATTN_VERSION=2.7.0.post2
# Prebuilt wheel for cu12 / torch2.5 / cp311. Falls back to a source build if the
# download fails or the URL is set to an empty string.
ARG FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
ARG MAX_JOBS=8

# Where the repository lives inside the image.
ENV EGOLENS_ROOT=/workspace/EgoLens

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# ---------------------------------------------------------------------------
# System packages + Python 3.11
# ---------------------------------------------------------------------------
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg software-properties-common \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-dev \
      python${PYTHON_VERSION}-venv \
      build-essential \
      ninja-build \
      git \
      libaio-dev \
      libgomp1 \
      openssh-server \
      procps \
      htop \
      less \
      rsync \
      tmux \
      vim \
 && curl -sSL https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION} 1 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel ninja packaging

# ---------------------------------------------------------------------------
# PyTorch (installed first: deepspeed / flash-attn compile against it)
# ---------------------------------------------------------------------------
RUN python -m pip install \
      torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
      --index-url https://download.pytorch.org/whl/cu124

# ---------------------------------------------------------------------------
# Source tree, cloned from GitHub
# ---------------------------------------------------------------------------
ARG EGOLENS_REPO=https://github.com/wghou/EgoLens.git
ARG EGOLENS_REF=main
ARG EGOLENS_CACHEBUST=1

RUN git clone "${EGOLENS_REPO}" "${EGOLENS_ROOT}" \
 && cd "${EGOLENS_ROOT}" \
 && git checkout --detach "${EGOLENS_REF}" \
 && git rev-parse HEAD | tee "${EGOLENS_ROOT}/.build-commit"

# ---------------------------------------------------------------------------
# Project dependencies (requirements-optional.txt includes requirements.txt)
# ---------------------------------------------------------------------------
RUN python -m pip install -r "${EGOLENS_ROOT}/requirements-optional.txt"

# ---------------------------------------------------------------------------
# FlashAttention 2
# ---------------------------------------------------------------------------
RUN set -eux; \
    if [ -n "${FLASH_ATTN_WHEEL_URL}" ] && python -m pip install "${FLASH_ATTN_WHEEL_URL}"; then \
      echo "flash-attn: installed prebuilt wheel"; \
    else \
      echo "flash-attn: building from source (this takes a while)"; \
      MAX_JOBS=${MAX_JOBS} python -m pip install \
        flash-attn==${FLASH_ATTN_VERSION} --no-build-isolation; \
    fi

# ---------------------------------------------------------------------------
# SAM2 CUDA extension (sam2/_C.so)
# ---------------------------------------------------------------------------
# Only get_connected_components() needs it and the training / evaluation paths in
# this repo never call it, so a failure here must not break the image.
RUN cd "${EGOLENS_ROOT}/src/segment_anything_2" \
 && ( MAX_JOBS=${MAX_JOBS} python setup.py build_ext --inplace \
      && echo "sam2 extension: built" ) \
    || echo "sam2 extension: build FAILED, continuing without it" >&2

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
# PYTHONPATH mirrors what the train / eval shell scripts export.
ENV PYTHONPATH=/workspace/EgoLens:/workspace/EgoLens/src/segment_anything_2:/workspace/EgoLens/eval \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    WANDB_MODE=offline

# Expected layout for data and checkpoints, left empty on purpose.
RUN mkdir -p /cache "${EGOLENS_ROOT}/data" "${EGOLENS_ROOT}/pretrained" "${EGOLENS_ROOT}/outputs" \
 && chmod 777 /cache

# An ssh session is not a child of the container's PID 1, so it does not inherit
# the ENV above. Export it through /etc/environment (read by PAM, covers
# `ssh host cmd`) and a profile snippet (covers interactive login shells).
RUN set -eux; \
    for kv in \
      "EGOLENS_ROOT=${EGOLENS_ROOT}" \
      "CUDA_HOME=${CUDA_HOME}" \
      "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
      "PYTHONPATH=${PYTHONPATH}" \
      "HF_HOME=${HF_HOME}" \
      "TORCH_HOME=${TORCH_HOME}" \
      "TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}" \
      "TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM}" \
      "OMP_NUM_THREADS=${OMP_NUM_THREADS}" \
      "WANDB_MODE=${WANDB_MODE}" \
    ; do \
      echo "${kv}" >> /etc/environment; \
      echo "export ${kv}" >> /etc/profile.d/egolens.sh; \
    done; \
    echo "cd ${EGOLENS_ROOT}" >> /etc/profile.d/egolens.sh; \
    chmod 644 /etc/profile.d/egolens.sh

# Key-based root login only. The entrypoint switches this on when a password is
# explicitly supplied.
RUN sed -i \
      -e 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' \
      -e 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' \
      -e 's/^#\?X11Forwarding.*/X11Forwarding no/' \
      /etc/ssh/sshd_config \
 && mkdir -p /run/sshd /root/.ssh \
 && chmod 700 /root/.ssh \
 && rm -f /etc/ssh/ssh_host_*

# Generated here rather than COPYed so the build stays independent of the host.
RUN cat > /usr/local/bin/entrypoint.sh <<'EOS' && chmod +x /usr/local/bin/entrypoint.sh
#!/usr/bin/env bash
# Starts sshd only when explicitly asked, and only with real credentials.
#
#   ENABLE_SSH=1        opt in
#   SSH_PUBKEY="ssh-ed25519 AAAA... user@host"   preferred
#   ROOT_PASSWORD=...   fallback, enables password auth (weaker)
set -euo pipefail

if [[ "${ENABLE_SSH:-0}" == "1" ]]; then
    mkdir -p /run/sshd /root/.ssh
    chmod 700 /root/.ssh

    if [[ -n "${SSH_PUBKEY:-}" ]]; then
        printf '%s\n' "${SSH_PUBKEY}" >> /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
    fi

    if [[ -s /root/.ssh/authorized_keys ]] || [[ -n "${ROOT_PASSWORD:-}" ]]; then
        if [[ -n "${ROOT_PASSWORD:-}" ]]; then
            echo "root:${ROOT_PASSWORD}" | chpasswd
            sed -i -e 's/^#\?PasswordAuthentication.*/PasswordAuthentication yes/' \
                   -e 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' \
                   /etc/ssh/sshd_config
            echo "[entrypoint] sshd: password auth enabled for root" >&2
        fi
        ssh-keygen -A >/dev/null
        /usr/sbin/sshd
        echo "[entrypoint] sshd listening on 22" >&2
    else
        echo "[entrypoint] ENABLE_SSH=1 but neither SSH_PUBKEY nor ROOT_PASSWORD was" >&2
        echo "[entrypoint] provided; refusing to start an sshd nobody can log into." >&2
    fi
fi

if [[ $# -eq 0 ]]; then
    exec bash
fi
exec "$@"
EOS

# Fail the build if a transitive dependency replaced torch, or if the project
# modules do not import.
RUN cd "${EGOLENS_ROOT}" && python - <<'PY'
import torch, transformers, accelerate, deepspeed, trl, flash_attn, numpy
assert torch.__version__.startswith("2.5.1"), torch.__version__
assert torch.version.cuda.startswith("12.4"), torch.version.cuda
assert numpy.__version__.startswith("1.26"), numpy.__version__
from src.open_r1.trainer.samr1 import SAMR1ForConditionalGeneration_qwen2p5
from src.open_r1.trainer.trainer_sft import Qwen2VLSFTTrainer
from src.open_r1.dataset import EgoAffordTrainDataset
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__, "| trl", trl.__version__)
print("deepspeed", deepspeed.__version__, "| flash_attn", flash_attn.__version__)
print("EgoLens modules import OK")
PY

EXPOSE 22
WORKDIR /workspace/EgoLens
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

#!/usr/bin/env bash
#
# Full-task EgoLens training, self-contained and unattended.
#
# Designed to be the ONLY command run in a freshly created container. It
# bootstraps everything, launches training, and records enough diagnostics in its
# log that nothing else has to be inspected interactively:
#
#   1. reports the environment (GPUs, driver, CPU, RAM, shm, disk, versions)
#   2. wires up the data / weight symlinks the README layout expects
#   3. rebuilds the SAM2 CUDA extension for the GPU actually present
#   4. derives GPU count, compute capability, grad accumulation, dataloader workers
#   5. samples GPU memory in the background and reports the peak
#   6. retries with a smaller per-device batch if it hits CUDA OOM
#   7. prints a final summary with wall time and artifact locations
#
# Usage:
#   bash scripts/train_qwen2p5_3b_full_auto.sh                # full 40-epoch run
#   SMOKE=1 bash scripts/train_qwen2p5_3b_full_auto.sh        # short end-to-end check
#   SETUP_ONLY=1 bash scripts/train_qwen2p5_3b_full_auto.sh   # bootstrap + report, no training
#
#   RES_LOSS_MODE=mean bash scripts/train_qwen2p5_3b_full_auto.sh
#       Mask loss aggregation. legacy (default) reproduces upstream, where the
#       empty-mask BCE outweighs the non-empty BCE+Dice by about 3*batch_size and
#       the non-empty terms sit at 1/3 of lm_loss. norm fixes the first, mean fixes
#       both. Non-legacy modes append _res<mode> to RUN_NAME.
#
# Everything is overridable by environment variable:
#   GPUS EFFECTIVE_BATCH PER_DEVICE_BS GRAD_ACCUM EPOCHS LR WARMUP_STEPS
#   DATALOADER_WORKERS SAVE_TOTAL_LIMIT DATASET_SIZE RUN_NAME CKPT_ROOT DISK
#   MAX_OOM_RETRIES MIN_DISK_GB SMOKE SETUP_ONLY
#
# Batch sizing note: grad accumulation is derived so the effective batch stays at
# the paper's 512 regardless of GPU count, which keeps lr / schedule / epoch count
# comparable to the published numbers.
#     8 GPUs -> 16 x 4 accum     4 GPUs -> 16 x 8 accum     2 GPUs -> 16 x 16 accum

set -uo pipefail

START_TS=$(date +%s)

REPO_ROOT="${REPO_ROOT:-/workspace/EgoLens}"
cd "${REPO_ROOT}" || { echo "[error] repo not found: ${REPO_ROOT}" >&2; exit 1; }
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# Real mount point of the shared disk. Everything is derived from it, so a missing
# ${REPO_ROOT}/wghou-disk convenience link is not a problem.
DISK="${DISK:-/mnt/robot/robot2/system-architecture-data/wghou-disk}"
SRC_DATA="${DISK}/datasets/Pantheonmonilaum/EgoAfford"
SRC_QWEN="${DISK}/models/Qwen/Qwen2.5-VL-3B-Instruct"
SRC_EGOLENS="${DISK}/models/Pantheonmonilaum/EgoLens"
SRC_SAM2="${DISK}/models/facebook/sam2-hiera-large/sam2_hiera_large.pt"

DATA_DIR="${DATA_DIR:-data/EgoAfford}"
MODEL_PATH="${MODEL_PATH:-./pretrained/Qwen/Qwen2.5-VL-3B-Instruct}"
SAM_CKPT="./pretrained/sam2_hiera_large.pt"   # path is hardcoded in trainer_sft.py

RUN_NAME="${RUN_NAME:-qwen2p5_sft_full}"

# How the three mask loss terms are combined; see compute_loss in trainer_sft.py.
# legacy reproduces upstream, norm removes the empty-term over-weighting, mean
# removes that and the extra 1/3 on the non-empty terms. Anything other than legacy
# gets its own run name so results never overwrite an existing run. Resolved here
# rather than with the other hyperparameters because RUN_NAME feeds OUTPUT_DIR.
RES_LOSS_MODE="${RES_LOSS_MODE:-legacy}"
case "${RES_LOSS_MODE}" in
    legacy)    ;;
    norm|mean) RUN_NAME="${RUN_NAME}_res${RES_LOSS_MODE}" ;;
    *) echo "[error] RES_LOSS_MODE must be legacy, norm or mean" >&2; exit 1 ;;
esac

[[ "${SMOKE:-0}" == "1" ]] && RUN_NAME="${RUN_NAME}_smoke"

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { printf '\n[FATAL] %s\n' "$*" >&2; exit 1; }
rule() { printf -- '--- %s %s\n' "$1" "$(printf '%.0s-' $(seq 1 $((66 - ${#1}))))"; }

# ---------------------------------------------------------------------------
# 0. Logging. Capture everything, including this bootstrap, on the shared disk so
#    the run can be diagnosed later without touching the container.
# ---------------------------------------------------------------------------
if [[ -d "${DISK}" ]]; then
    CKPT_ROOT="${CKPT_ROOT:-${DISK}/ckpts/EgoLens}"
else
    CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/outputs}"
    echo "[warn] shared disk not mounted at ${DISK}; falling back to ${CKPT_ROOT}" >&2
fi
OUTPUT_DIR="${CKPT_ROOT}/${RUN_NAME}"
export LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}" || die "cannot create ${LOG_DIR}"
RUN_LOG="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${RUN_LOG}") 2>&1

MEM_SAMPLES="${LOG_DIR}/.gpu_mem_samples"
cleanup() {
    [[ -n "${MEM_PID:-}" ]] && kill "${MEM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "==============================================================="
echo " EgoLens full-task training"
echo " run       ${RUN_NAME}"
echo " started   $(date -Iseconds)"
echo " log       ${RUN_LOG}"
echo "==============================================================="

# ---------------------------------------------------------------------------
# 1. Environment report
# ---------------------------------------------------------------------------
rule "environment"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version \
           --format=csv 2>&1 | sed 's/^/  /'
printf '  host   %s\n' "$(hostname)"
# nproc reports the scheduling affinity, which in a container can be far below
# both the node's core count and the cgroup CPU quota. Dataloader throughput
# depends on this, so show all three rather than just nproc.
cpu_detail() {
    printf 'affinity %s' "$(nproc)"
    [[ -r /proc/cpuinfo ]] && printf ' | node %s' "$(grep -c ^processor /proc/cpuinfo)"
    if [[ -r /sys/fs/cgroup/cpu.max ]]; then                      # cgroup v2
        read -r q p </sys/fs/cgroup/cpu.max
        [[ "${q}" == "max" ]] && printf ' | quota none' \
                              || printf ' | quota %s' "$((q / p))"
    elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then       # cgroup v1
        q=$(</sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        p=$(</sys/fs/cgroup/cpu/cpu.cfs_period_us)
        ((q > 0)) && printf ' | quota %s' "$((q / p))" || printf ' | quota none'
    fi
}
printf '  cpu    %s | ram %s\n' "$(cpu_detail)" "$(free -g | awk 'NR==2{print $2" GB"}')"
printf '  shm    %s\n' "$(df -h /dev/shm 2>/dev/null | awk 'NR==2{print $2" ("$4" free)"}')"
printf '  disk   overlay %s | shared %s\n' \
    "$(df -h / | awk 'NR==2{print $4" free"}')" \
    "$(df -h "${DISK}" 2>/dev/null | awk 'NR==2{print $4" free"}')"
printf '  commit %s\n' "$(cat "${REPO_ROOT}/.build-commit" 2>/dev/null \
                          || git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null \
                          || echo unknown)"
python - <<'PY' 2>/dev/null | sed 's/^/  /'
import torch, transformers, trl, deepspeed, accelerate, flash_attn, numpy
print(f"torch {torch.__version__} (cuda {torch.version.cuda}) | numpy {numpy.__version__}")
print(f"transformers {transformers.__version__} | trl {trl.__version__} | "
      f"accelerate {accelerate.__version__}")
print(f"deepspeed {deepspeed.__version__} | flash_attn {flash_attn.__version__}")
PY

# ---------------------------------------------------------------------------
# 2. GPU topology and target architecture
# ---------------------------------------------------------------------------
DETECTED_GPUS="$(nvidia-smi -L | wc -l)"
GPUS="${GPUS:-${DETECTED_GPUS}}"
((GPUS >= 1)) || die "no GPU detected"
((GPUS <= DETECTED_GPUS)) || die "GPUS=${GPUS} exceeds the ${DETECTED_GPUS} GPUs present"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
# H200, H20 and H100 are all compute capability 9.0. The container image bakes in
# 8.9 (RTX 4090); leaving that in place makes every runtime CUDA JIT emit kernels
# these GPUs cannot execute.
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | tr -d ' ')"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST_OVERRIDE:-${COMPUTE_CAP:-9.0}}"
SM="sm_${COMPUTE_CAP//./}"
log "using ${GPUS}/${DETECTED_GPUS} x ${GPU_NAME} (${SM}), TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

# ---------------------------------------------------------------------------
# 3. Data and weight symlinks (idempotent)
# ---------------------------------------------------------------------------
rule "bootstrap"
[[ -d "${DISK}" ]] || die "shared disk not mounted: ${DISK}"

link() {  # link <source> <destination>
    local src="$1" dst="$2"
    [[ -e "${src}" ]] || die "source missing: ${src}"
    mkdir -p "$(dirname "${dst}")"
    ln -sfn "${src}" "${dst}"   # -n replaces an existing symlink-to-dir
}

link "${DISK}"                    "${REPO_ROOT}/wghou-disk"
link "${SRC_DATA}/EgoAfford"      "${REPO_ROOT}/data/EgoAfford"
link "${SRC_DATA}/EgoAfford_real" "${REPO_ROOT}/data/EgoAfford_real"
link "${SRC_QWEN}"                "${REPO_ROOT}/pretrained/Qwen/Qwen2.5-VL-3B-Instruct"
link "${SRC_EGOLENS}"             "${REPO_ROOT}/pretrained/EgoLens"

if [[ -e "${SRC_SAM2}" ]]; then
    link "${SRC_SAM2}" "${REPO_ROOT}/${SAM_CKPT#./}"
else
    die "SAM2 weights missing: ${SRC_SAM2}
       This host has no route to the public internet. Fetch on a machine that does:
         curl -L -A 'Mozilla/5.0' -o sam2_hiera_large.pt \\
           https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
         # 897952466 bytes, md5 08083462423be3260cd6a5eef94dc01c
       then place it at ${SRC_SAM2}"
fi
log "symlinks wired to ${DISK}"

# ---------------------------------------------------------------------------
# 4. SAM2 CUDA extension
# ---------------------------------------------------------------------------
# The image ships sam2/_C.so built for whatever TORCH_CUDA_ARCH_LIST was set at
# build time. On a different GPU generation it imports fine but every kernel
# launch fails with "no kernel image is available for execution on the device", so
# probe it functionally rather than checking for the file. Training and evaluation
# never call into it (only get_connected_components does): best-effort, not fatal.
ext_works() {
    python - <<'PY' >/dev/null 2>&1
import torch
from src.segment_anything_2.sam2.utils.misc import get_connected_components
m = torch.zeros(1, 1, 32, 32, dtype=torch.uint8, device="cuda")
m[0, 0, 4:12, 4:12] = 1
get_connected_components(m)
PY
}

if ext_works; then
    log "sam2 extension already valid for ${SM}"
else
    log "sam2 extension rebuilding for ${SM} ..."
    if ( cd src/segment_anything_2 && rm -rf build sam2/_C*.so \
         && python setup.py build_ext --inplace ) >"${LOG_DIR}/sam2_ext_build.log" 2>&1; then
        ext_works && log "sam2 extension rebuilt and verified" \
                  || log "WARN sam2 extension built but still unusable (unused by training)"
    else
        log "WARN sam2 extension rebuild failed, see ${LOG_DIR}/sam2_ext_build.log"
        log "     (unused by training and evaluation, continuing)"
    fi
fi

# ---------------------------------------------------------------------------
# 5. Hyperparameters
# ---------------------------------------------------------------------------
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-512}"
PER_DEVICE_BS="${PER_DEVICE_BS:-16}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-3e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
DATASET_SIZE="${DATASET_SIZE:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
MAX_OOM_RETRIES="${MAX_OOM_RETRIES:-2}"
MIN_DISK_GB="${MIN_DISK_GB:-120}"

# Image decoding and bit-packed mask unpacking are CPU bound; give each rank a
# share of the cores without oversubscribing.
if [[ -z "${DATALOADER_WORKERS:-}" ]]; then
    DATALOADER_WORKERS=$(( $(nproc) / GPUS / 2 ))
    ((DATALOADER_WORKERS < 2)) && DATALOADER_WORKERS=2
    ((DATALOADER_WORKERS > 16)) && DATALOADER_WORKERS=16
fi

derive_accum() {  # keep the effective batch fixed for the current PER_DEVICE_BS
    local per_step=$((PER_DEVICE_BS * GPUS))
    GRAD_ACCUM=$((EFFECTIVE_BATCH / per_step))
    ((GRAD_ACCUM >= 1)) || GRAD_ACCUM=1
    if ((per_step * GRAD_ACCUM != EFFECTIVE_BATCH)); then
        log "WARN ${PER_DEVICE_BS}x${GPUS} does not divide ${EFFECTIVE_BATCH};" \
            "accum=${GRAD_ACCUM} gives $((per_step * GRAD_ACCUM))"
    fi
}
[[ -n "${GRAD_ACCUM:-}" ]] || derive_accum

# Short run over ~30 scenes at the real per-device batch size, so the memory
# footprint stays representative of the full run.
if [[ "${SMOKE:-0}" == "1" ]]; then
    EPOCHS=1
    GRAD_ACCUM=1
    DATASET_SIZE=130
    SAVE_TOTAL_LIMIT=1
    log "SMOKE mode: scenes 101-${DATASET_SIZE}, 1 epoch, accum=1"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-EgoLens}"
export WANDB_TAGS="${RUN_NAME}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# 6. Preflight. dataset.py only warns and returns an empty sample list when
#    image_dir is wrong, which surfaces later as an unrelated error.
# ---------------------------------------------------------------------------
rule "preflight"
[[ -d "${DATA_DIR}" ]]              || die "missing dataset: ${DATA_DIR}"
[[ -f "${DATA_DIR}/invalid.json" ]] || die "missing ${DATA_DIR}/invalid.json"
[[ -d "${MODEL_PATH}" ]]            || die "missing backbone: ${MODEL_PATH}"
[[ -f "${SAM_CKPT}" ]]              || die "missing SAM2 weights: ${SAM_CKPT}"
[[ -f ./configs/zero3.json ]]       || die "missing ./configs/zero3.json"
[[ -f ./src/open_r1/sft_vllm_sam.py ]] || die "missing training entrypoint"

# -L: DATA_DIR is a symlink, and find will not descend into one otherwise.
SCENES=$(find -L "${DATA_DIR}" -maxdepth 1 -name 'scene_*' -type d | wc -l)
((SCENES > 100)) || die "only ${SCENES} scenes under ${DATA_DIR}, expected 2000"

# Each checkpoint is ~9 GB; save_total_limit keeps SAVE_TOTAL_LIMIT plus the final
# model, so require a little more than that up front.
FREE_GB=$(df -BG --output=avail "${CKPT_ROOT}" 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -n "${FREE_GB}" ]] && ((FREE_GB < MIN_DISK_GB)); then
    die "only ${FREE_GB} GB free at ${CKPT_ROOT}, need >= ${MIN_DISK_GB} GB"
fi

printf '  %-14s %s\n' \
    "data"      "${DATA_DIR} -> $(readlink -f "${DATA_DIR}") (${SCENES} scenes)" \
    "backbone"  "${MODEL_PATH}" \
    "sam2"      "$(readlink -f "${SAM_CKPT}")" \
    "output"    "${OUTPUT_DIR}" \
    "free space" "${FREE_GB:-?} GB" \
    "batch"     "${PER_DEVICE_BS} x ${GRAD_ACCUM} accum x ${GPUS} gpu = $((PER_DEVICE_BS * GRAD_ACCUM * GPUS)) effective" \
    "epochs"    "${EPOCHS} | lr ${LR} | warmup ${WARMUP_STEPS}" \
    "mask loss"  "${RES_LOSS_MODE}" \
    "workers"   "${DATALOADER_WORKERS} per rank" \
    "keep ckpts" "${SAVE_TOTAL_LIMIT}"

if [[ "${SETUP_ONLY:-0}" == "1" ]]; then
    rule "done"
    log "SETUP_ONLY=1, bootstrap complete, nothing launched"
    exit 0
fi

# ---------------------------------------------------------------------------
# 7. Background GPU memory sampler, so the peak is in the log without needing a
#    second shell.
# ---------------------------------------------------------------------------
: >"${MEM_SAMPLES}"
( while :; do
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
          >>"${MEM_SAMPLES}" 2>/dev/null
      sleep 20
  done ) &
MEM_PID=$!

report_peak_memory() {
    [[ -s "${MEM_SAMPLES}" ]] || return 0
    local total
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
    echo "  peak GPU memory (of ${total} MiB per card):"
    awk -F', *' '{ if ($2+0 > m[$1]) m[$1]=$2+0 }
                 END { for (i in m) printf "    gpu %s  %d MiB  (%.0f%%)\n", i, m[i], 100*m[i]/T }' \
        T="${total}" "${MEM_SAMPLES}" | sort
}

# ---------------------------------------------------------------------------
# 8. Train, retrying at a smaller per-device batch if CUDA OOM shows up. There is
#    nobody around to intervene, so degrade automatically instead of dying.
# ---------------------------------------------------------------------------
run_training() {
    local train_log="$1"
    torchrun \
        --nproc_per_node="${GPUS}" \
        --nnodes="${NNODES:-1}" \
        --node_rank="${NODE_RANK:-0}" \
        --master_addr="${MASTER_ADDR:-127.0.0.1}" \
        --master_port="${MASTER_PORT:-12345}" \
        ./src/open_r1/sft_vllm_sam.py \
        --deepspeed ./configs/zero3.json \
        --image_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --model_name_or_path "${MODEL_PATH}" \
        --dataset_size "${DATASET_SIZE}" \
        --per_device_train_batch_size "${PER_DEVICE_BS}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --num_train_epochs "${EPOCHS}" \
        --learning_rate "${LR}" \
        --lr_scheduler_type "cosine" \
        --warmup_steps "${WARMUP_STEPS}" \
        --logging_steps 1 \
        --bf16 \
        --gradient_checkpointing true \
        --attn_implementation flash_attention_2 \
        --max_pixels 1000000 \
        --num_of_query 64 \
        --if_use_qwen_connector true \
        --prompt_difficulty "hard" \
        --res_loss_mode "${RES_LOSS_MODE}" \
        --dataloader_num_workers "${DATALOADER_WORKERS}" \
        --save_only_model true \
        --save_total_limit "${SAVE_TOTAL_LIMIT}" \
        --report_to wandb \
        --run_name "${RUN_NAME}" \
        >"${train_log}" 2>&1
}

mkdir -p "${OUTPUT_DIR}"
cp -rT ./src "${OUTPUT_DIR}/src"   # source snapshot, -T avoids src/src

STATUS=1
for ((attempt = 0; attempt <= MAX_OOM_RETRIES; attempt++)); do
    rule "training (attempt $((attempt + 1)))"
    log "batch ${PER_DEVICE_BS} x ${GRAD_ACCUM} accum x ${GPUS} gpu" \
        "= $((PER_DEVICE_BS * GRAD_ACCUM * GPUS)) effective"

    TRAIN_LOG="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
    log "training log ${TRAIN_LOG}"
    # Mirror progress into the run log without duplicating the multi-GB raw output.
    tail -F "${TRAIN_LOG}" 2>/dev/null | grep --line-buffered -E \
        "'loss'|train_runtime|Error|error:|Traceback|out of memory" &
    TAIL_PID=$!

    run_training "${TRAIN_LOG}"
    STATUS=$?
    kill "${TAIL_PID}" 2>/dev/null || true

    if ((STATUS == 0)); then
        log "training finished cleanly"
        break
    fi

    if grep -qiE "CUDA out of memory|torch.OutOfMemoryError" "${TRAIN_LOG}" \
       && ((PER_DEVICE_BS > 1)) && ((attempt < MAX_OOM_RETRIES)); then
        PER_DEVICE_BS=$((PER_DEVICE_BS / 2))
        derive_accum
        log "WARN CUDA OOM detected; halving per-device batch to ${PER_DEVICE_BS}" \
            "and retrying (effective batch unchanged)"
        sleep 30
        continue
    fi

    log "training FAILED with status ${STATUS}; last lines of ${TRAIN_LOG}:"
    tail -n 25 "${TRAIN_LOG}" | sed 's/^/    /'
    break
done

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
cleanup
ELAPSED=$(( $(date +%s) - START_TS ))
rule "summary"
printf '  status        %s\n' "$( ((STATUS == 0)) && echo SUCCESS || echo "FAILED (${STATUS})" )"
printf '  wall time     %02dh%02dm%02ds\n' $((ELAPSED / 3600)) $((ELAPSED % 3600 / 60)) $((ELAPSED % 60))
printf '  final batch   %s x %s accum x %s gpu = %s effective\n' \
    "${PER_DEVICE_BS}" "${GRAD_ACCUM}" "${GPUS}" "$((PER_DEVICE_BS * GRAD_ACCUM * GPUS))"
report_peak_memory
printf '  output        %s\n' "${OUTPUT_DIR}"
printf '  size          %s\n' "$(du -sh "${OUTPUT_DIR}" 2>/dev/null | cut -f1)"
if ((STATUS == 0)); then
    printf '  final model   %s\n' \
        "$(ls "${OUTPUT_DIR}"/model*.safetensors 2>/dev/null | wc -l) shard(s)"
    ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sed 's/^/  checkpoint    /'
    grep -o "{'train_runtime'.*}" "${TRAIN_LOG}" 2>/dev/null | tail -1 | sed 's/^/  metrics       /'
fi
printf '  run log       %s\n' "${RUN_LOG}"
printf '  train log     %s\n' "${TRAIN_LOG:-n/a}"
echo "==============================================================="
exit "${STATUS}"

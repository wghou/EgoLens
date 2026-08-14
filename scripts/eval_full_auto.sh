#!/usr/bin/env bash
#
# Full-task EgoLens evaluation, self-bootstrapping.
#
# Companion to train_qwen2p5_3b_full_auto.sh: safe to run as the only command in a
# freshly created container. Wires up the symlinks, points HuggingFace at the
# offline cache on the shared disk, sizes torchrun to the GPUs present, and runs
# the multi-reference evaluator.
#
#   bash scripts/eval_full_auto.sh                    # generated test set (488 samples)
#   SPLIT=real bash scripts/eval_full_auto.sh         # EgoAfford-Real, zero-shot (102)
#   SPLIT=both bash scripts/eval_full_auto.sh         # both, sequentially
#   SETUP_ONLY=1 bash scripts/eval_full_auto.sh       # bootstrap + report, no evaluation
#
# Overridable: MODEL_PATH GPUS SPLIT VIS VIS_FREQ BATCH_SIZE DISK SETUP_ONLY
#              EGOLENS_CE_MODEL
#
# Results land in ${MODEL_PATH}/evaluations_multi (and evaluations_multi_real),
# i.e. next to the checkpoint on the shared disk, because that is where
# evaluate_egoaff_multi.py writes them.

set -uo pipefail

START_TS=$(date +%s)

REPO_ROOT="${REPO_ROOT:-/workspace/EgoLens}"
cd "${REPO_ROOT}" || { echo "[error] repo not found: ${REPO_ROOT}" >&2; exit 1; }
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src/segment_anything_2:${REPO_ROOT}/eval:${PYTHONPATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

DISK="${DISK:-/mnt/robot/robot2/system-architecture-data/wghou-disk}"
SRC_DATA="${DISK}/datasets/Pantheonmonilaum/EgoAfford"
SRC_QWEN="${DISK}/models/Qwen/Qwen2.5-VL-3B-Instruct"
SRC_EGOLENS="${DISK}/models/Pantheonmonilaum/EgoLens"
SRC_SAM2="${DISK}/models/facebook/sam2-hiera-large/sam2_hiera_large.pt"

# Default to the checkpoint produced by train_qwen2p5_3b_full_auto.sh.
MODEL_PATH="${MODEL_PATH:-${DISK}/ckpts/EgoLens/qwen2p5_sft_full}"
SPLIT="${SPLIT:-generated}"          # generated | real | both
VIS="${VIS:-1}"
VIS_FREQ="${VIS_FREQ:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"

# No route to huggingface.co from these hosts; the text cross-encoder lives on the
# shared disk in standard hub layout. See eval/metrics.py for why the model id
# differs from upstream.
export HF_HOME="${HF_HOME_OVERRIDE:-${DISK}/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export EGOLENS_CE_MODEL="${EGOLENS_CE_MODEL:-cross-encoder/stsb-roberta-base}"

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { printf '\n[FATAL] %s\n' "$*" >&2; exit 1; }
rule() { printf -- '--- %s %s\n' "$1" "$(printf '%.0s-' $(seq 1 $((66 - ${#1}))))"; }

# ---------------------------------------------------------------------------
# Logging, on the shared disk next to the results.
# ---------------------------------------------------------------------------
[[ -d "${DISK}" ]]       || die "shared disk not mounted: ${DISK}"
[[ -d "${MODEL_PATH}" ]] || die "checkpoint not found: ${MODEL_PATH}"
LOG_DIR="${MODEL_PATH}/logs"
mkdir -p "${LOG_DIR}" || die "cannot write to ${LOG_DIR}"
RUN_LOG="${LOG_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "==============================================================="
echo " EgoLens full-task evaluation"
echo " checkpoint  ${MODEL_PATH}"
echo " split       ${SPLIT}"
echo " started     $(date -Iseconds)"
echo " log         ${RUN_LOG}"
echo "==============================================================="

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
rule "environment"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version \
           --format=csv 2>&1 | sed 's/^/  /'
printf '  host   %s | cpu affinity %s of %s | ram %s GB\n' \
    "$(hostname)" "$(nproc)" "$(grep -c ^processor /proc/cpuinfo)" \
    "$(free -g | awk 'NR==2{print $2}')"
printf '  commit %s\n' "$(cat "${REPO_ROOT}/.build-commit" 2>/dev/null \
                          || git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null \
                          || echo unknown)"

DETECTED_GPUS="$(nvidia-smi -L | wc -l)"
GPUS="${GPUS:-${DETECTED_GPUS}}"
((GPUS >= 1)) || die "no GPU detected"
((GPUS <= DETECTED_GPUS)) || die "GPUS=${GPUS} exceeds the ${DETECTED_GPUS} present"
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | tr -d ' ')"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST_OVERRIDE:-${COMPUTE_CAP:-9.0}}"
log "using ${GPUS}/${DETECTED_GPUS} GPU, TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

# ---------------------------------------------------------------------------
# Symlinks (idempotent; identical to the training script)
# ---------------------------------------------------------------------------
rule "bootstrap"
link() {
    local src="$1" dst="$2"
    [[ -e "${src}" ]] || die "source missing: ${src}"
    mkdir -p "$(dirname "${dst}")"
    ln -sfn "${src}" "${dst}"
}
link "${DISK}"                    "${REPO_ROOT}/wghou-disk"
link "${SRC_DATA}/EgoAfford"      "${REPO_ROOT}/data/EgoAfford"
link "${SRC_DATA}/EgoAfford_real" "${REPO_ROOT}/data/EgoAfford_real"
link "${SRC_QWEN}"                "${REPO_ROOT}/pretrained/Qwen/Qwen2.5-VL-3B-Instruct"
link "${SRC_EGOLENS}"             "${REPO_ROOT}/pretrained/EgoLens"
[[ -e "${SRC_SAM2}" ]] && link "${SRC_SAM2}" "${REPO_ROOT}/pretrained/sam2_hiera_large.pt"
log "symlinks wired to ${DISK}"

# The extension is unused by evaluation (only get_connected_components calls it),
# so only rebuild when it is outright missing for this GPU generation.
if ! python - <<'PY' >/dev/null 2>&1
import torch
from src.segment_anything_2.sam2.utils.misc import get_connected_components
m = torch.zeros(1, 1, 32, 32, dtype=torch.uint8, device="cuda")
m[0, 0, 4:12, 4:12] = 1
get_connected_components(m)
PY
then
    log "sam2 extension not usable on sm_${COMPUTE_CAP//./}; rebuilding"
    ( cd src/segment_anything_2 && rm -rf build sam2/_C*.so \
      && python setup.py build_ext --inplace ) >"${LOG_DIR}/sam2_ext_build.log" 2>&1 \
        && log "sam2 extension rebuilt" \
        || log "WARN rebuild failed (unused by evaluation, continuing)"
else
    log "sam2 extension valid for sm_${COMPUTE_CAP//./}"
fi

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
rule "preflight"
[[ -f "${MODEL_PATH}/config.json" ]] || die "no config.json in ${MODEL_PATH}"
ls "${MODEL_PATH}"/model*.safetensors >/dev/null 2>&1 \
    || die "no model shards in ${MODEL_PATH}"
[[ -f ./eval/evaluate_egoaff_multi.py ]] || die "missing eval/evaluate_egoaff_multi.py"

# The text metrics need the cross-encoder locally: these hosts cannot reach
# huggingface.co, and a missing model would only surface after the first sample.
ce_dir="${HF_HOME}/hub/models--${EGOLENS_CE_MODEL//\//--}"
[[ -d "${ce_dir}" ]] || die "cross-encoder not cached: ${ce_dir}
       No route to huggingface.co from this host. Fetch on a machine that has one:
         HF_ENDPOINT=https://hf-mirror.com HF_HOME=/tmp/hf python -c \\
           \"from huggingface_hub import snapshot_download as d; \\
             d('${EGOLENS_CE_MODEL}')\"
       then copy /tmp/hf/hub/models--${EGOLENS_CE_MODEL//\//--} into ${HF_HOME}/hub/"
python - <<PY || die "cross-encoder failed to load offline"
from sentence_transformers import CrossEncoder
m = CrossEncoder("${EGOLENS_CE_MODEL}", device="cpu")
same, diff = m.predict([("a cat sits on the mat", "a cat is sitting on the mat"),
                        ("a cat sits on the mat", "quantum chromodynamics is hard")])
print(f"  cross-encoder  {'${EGOLENS_CE_MODEL}'}")
print(f"    synonymous {float(same):.4f} | unrelated {float(diff):.4f}"
      f"{'' if float(same) > 0.6 > float(diff) else '   <-- WARNING: no discrimination'}")
PY

check_split() {  # check_split <dir> <label>
    local d="$1" label="$2"
    [[ -d "${d}" ]] || die "missing ${label}: ${d}"
    local n_scene n_alt
    # -L: the split directories are symlinks, find will not descend otherwise.
    n_scene=$(find -L "${d}" -maxdepth 1 -name 'scene_*' -type d | wc -l)
    n_alt=$(find -L "${d}" -maxdepth 2 -name 'alternatives.json' | wc -l)
    ((n_scene > 0)) || die "no scenes under ${d}"
    ((n_alt > 0))   || die "no alternatives.json under ${d}; multi-reference \
evaluation needs alternatives.json, masks_multi.npz and masks_multi_index.json"
    printf '  %-16s %s (%s scenes, %s with alternatives.json)\n' \
        "${label}" "${d}" "${n_scene}" "${n_alt}"
}
case "${SPLIT}" in
    generated) SPLITS=("data/EgoAfford") ;;
    real)      SPLITS=("data/EgoAfford_real") ;;
    both)      SPLITS=("data/EgoAfford" "data/EgoAfford_real") ;;
    *)         die "SPLIT must be generated, real or both (got '${SPLIT}')" ;;
esac
for d in "${SPLITS[@]}"; do check_split "${d}" "$(basename "${d}")"; done

printf '  %-16s %s\n' "checkpoint" "${MODEL_PATH}" \
                      "results to" "${MODEL_PATH}/evaluations_multi[_real]" \
                      "gpus" "${GPUS}" \
                      "vis" "$( ((VIS)) && echo "on (every ${VIS_FREQ})" || echo off )"

if [[ "${SETUP_ONLY:-0}" == "1" ]]; then
    rule "done"
    log "SETUP_ONLY=1, bootstrap complete, nothing launched"
    exit 0
fi

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
STATUS=0
for data_dir in "${SPLITS[@]}"; do
    rule "evaluating $(basename "${data_dir}")"
    vis_flag=(); ((VIS)) && vis_flag=(--vis --vis_freq "${VIS_FREQ}")
    EVAL_LOG="${LOG_DIR}/eval_$(basename "${data_dir}")_$(date +%Y%m%d_%H%M%S).log"
    log "detail log ${EVAL_LOG}"

    # Mirror progress without duplicating the multi-GB raw output.
    tail -F "${EVAL_LOG}" 2>/dev/null \
        | grep --line-buffered -iE "giou|ciou|it/s\]|Error|Traceback|out of memory" &
    TAIL_PID=$!

    torchrun --standalone --nproc_per_node "${GPUS}" \
        eval/evaluate_egoaff_multi.py \
        --model_path "${MODEL_PATH}" \
        --image_dir "${data_dir}" \
        --batch_size "${BATCH_SIZE}" \
        "${vis_flag[@]}" >"${EVAL_LOG}" 2>&1
    rc=$?
    kill "${TAIL_PID}" 2>/dev/null || true

    if ((rc == 0)); then
        log "$(basename "${data_dir}") finished"
    else
        STATUS=$rc
        log "$(basename "${data_dir}") FAILED (status ${rc}); last lines:"
        tail -n 25 "${EVAL_LOG}" | sed 's/^/    /'
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
ELAPSED=$(( $(date +%s) - START_TS ))
rule "summary"
printf '  status      %s\n' "$( ((STATUS == 0)) && echo SUCCESS || echo "FAILED (${STATUS})" )"
printf '  wall time   %02dh%02dm%02ds\n' $((ELAPSED / 3600)) $((ELAPSED % 3600 / 60)) $((ELAPSED % 60))
for out in "${MODEL_PATH}/evaluations_multi" "${MODEL_PATH}/evaluations_multi_real"; do
    [[ -d "${out}" ]] || continue
    printf '  results     %s\n' "${out}"
    find "${out}" -maxdepth 1 -name '*.txt' -o -maxdepth 1 -name '*.json' \
        | sort | sed 's/^/                /'
    # The evaluator writes a one-line summary next to the metrics dump.
    for f in "${out}"/*.txt; do
        [[ -f "${f}" ]] && sed 's/^/                /' "${f}"
    done
done
printf '  paper       gIoU 0.700  cIoU 0.486  FirstStepSim 0.666  F1 0.500  CSR 0.624  Cov 0.566  (generated)\n'
printf '              gIoU 0.666  cIoU 0.455  FirstStepSim 0.631  F1 0.426  CSR 0.609  Cov 0.481  (real)\n'
printf '  log         %s\n' "${RUN_LOG}"
echo "==============================================================="
exit "${STATUS}"

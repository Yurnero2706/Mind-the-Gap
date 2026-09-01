#!/bin/bash
#PBS -A UTSUROLB
#PBS -b 1
#PBS -q gpu
#PBS -l elapstim_req=07:00:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=5.0.10/gcc11.4.0-cuda12.6.3
VENV_PREFIX=/work/UTSUROLB/utlb_ngy/work/Mind-the-Gap/math_eval/eval
source ${VENV_PREFIX}/bin/activate

set -ex
# 1 H100 per node -> tensor_parallel_size=1 (eval.py uses len(CUDA_VISIBLE_DEVICES)).
# The SFT'd 1.5B/8B models fit on a single H100. Use "0,1,2,3" only on a real 4-GPU box.
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
# Checkpoints, tokenizers and the benchmark files are all local -> stay offline.
export HF_HOME=/work/UTSUROLB/utlb_ngy/work        # shared HF cache ($HF_HOME/hub)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# NQSV runs the job from a spool dir, so cd to where eval.py and its relative
# data/ and outputs/ paths live.
EVAL_DIR=/work/UTSUROLB/utlb_ngy/work/Mind-the-Gap/math_eval
cd "${EVAL_DIR}"
mkdir -p logs
LIVE_LOG="${EVAL_DIR}/logs/batch_eval_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LIVE_LOG}") 2>&1
echo "Node: $(hostname)"; nvidia-smi -L
echo "Live log: ${LIVE_LOG}"

# --- datasets: evaluated for EVERY model (eval.py loops this comma list,
#     loading each model only once). These are the six Table-1 benchmarks. ---
DATA_NAME="amc23,gaokao2023en,gsm8k,math500,olympiadbench,odyssey"

# --- models: one eval.py run per entry (batches the models) ---
# Repo IDs -> resolved offline from $HF_HOME/hub (models--zjuxhl--...). These are
# the authors' released checkpoints, already cached under /work/.../work/hub.
MODELS=(
  "zjuxhl/Llama3.1-8B-NuminaMath"
  "zjuxhl/Llama3.1-8B-NuminaMath-bridge"
)

for MODEL_NAME in "${MODELS[@]}"; do
    echo "==================================================================="
    echo "Evaluating model : ${MODEL_NAME}"
    echo "On datasets      : ${DATA_NAME}"
    echo "==================================================================="

    python3 -u eval.py \
        --model_name "${MODEL_NAME}" \
        --output_dir "${MODEL_NAME}" \
        --data_name "${DATA_NAME}" \
        --max_tokens 2048 \
        --temperature 0 \
        --n_sampling 4
        # --few_shot          # uncomment for the 4-shot base baseline
done

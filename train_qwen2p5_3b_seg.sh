export PYTHONPATH=$PYTHONPATH:$(pwd)
#!/bin/bash
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1  # ! MODIFY HERE
MASTER_PORT=12345

# MODIFY HERE: please prepare the env related variables
PR1_PATH="./"
CHECKPOINT_PATH="./outputs" # directory to save the checkpoint
RUN_NAME="qwen2p5_sft_seg" # describe what your experiment is about

# Default Setting
OUTPUT_DIR="${CHECKPOINT_PATH}/${RUN_NAME}" # path to save the output
SRC_PATH="${OUTPUT_DIR}/src" # path to backup the source code

export LOG_DIR="${OUTPUT_DIR}/logs" # path to save the log
export WANDB_PROJECT="EgoLens" # project name in wandb
export WANDB_TAGS="qwen2p5_sft_seg" # tags for the experiment in wandb
export WANDB_MODE=offline 

if [ ! -d "${OUTPUT_DIR}"/src ]; then
    mkdir -p ${OUTPUT_DIR}/src
fi

# backup the source code
cp -r ${PR1_PATH}/src ${SRC_PATH}
mkdir -p ${LOG_DIR}

# run the training
torchrun \
    --nproc_per_node="8" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    ${PR1_PATH}/src/open_r1/sft_vllm_sam.py \
    --deepspeed ${PR1_PATH}/configs/zero3.json \
    --image_dir "/Path/to/EgoAfford" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path ./pretrained/Qwen/Qwen2.5-VL-3B-Instruct \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --report_to wandb \
    --max_pixels 1000000 \
    --num_train_epochs 40 \
    --run_name ${RUN_NAME} \
    --save_only_model true \
    --learning_rate 3e-5 \
    --num_of_query 64 \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine" \
    --if_use_qwen_connector true \
    --prompt_difficulty "easy_cot" 

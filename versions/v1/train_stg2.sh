#!/bin/bash

NUM_GPU_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
MASTER_ADDR=$(ping -c 1 ${MASTER_ADDR} |grep PING | grep -E -o '([0-9]{1,3}\.){3}[0-9]{1,3}')

echo "====="
echo "NUM_WORKERS:${WORLD_SIZE}"
echo "NUM_GPU_PER_NODE:${NUM_GPU_PER_NODE}"
echo "MASTER_ADDR:${MASTER_ADDR}"
echo "====="


CLUSTER_ARGS="
    --nnodes $WORLD_SIZE \
    --nproc_per_node $NUM_GPU_PER_NODE \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:29400"


torchrun ${CLUSTER_ARGS} \
    sft/train.py \
    --deepspeed "conf/ds_config_zero2.json" \
    --optim "adamw_torch" \
    --learning_rate 2e-5 \
    --output_dir "models/v1/stg2" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --model_name_or_path "models/v1/stg1/checkpoint-xxxx" \
    --tokenizer_name_or_path "models/qwen2.5-7b-edited" \
    --data_path "versions/v1/stg2.conf" \
    --gradient_checkpointing true \
    --num_train_epochs 2 \
    --max_grad_norm 10 \
    --model_max_length 4096 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
    --save_steps 200 \
    --warmup_steps 50 \
    --evaluation_strategy "no" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --bf16 \
    --report_to none


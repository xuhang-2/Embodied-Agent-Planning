#!/bin/bash

# 设置环境变量
export PROJECT_ROOT="/mnt/public/tangyehui/code/LLaMA-Factory"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# "--model_name_or_path=/home/zy/code/models/llama_31_8b_instruct/"
# "--output_dir=/home/zy/code/LLaMA-Factory/saves/llama3-8b/lora/sft"
# 获取当前时间
current_time=$(date "+%Y%m%d_%H%M%S")

# 更新 output_dir 参数
output_dir="/mnt/tangyehui/code/LLaMA-Factory/saves/llama3-8b/lora/sft_${current_time}"

# 设置配置参数
# "--max_samples=4000"
CONFIG_PARAMS=(
    "--model_name_or_path=/mnt/public/tangyehui/model/llama_31_8b_instruct/"
    "--stage=sft"
    "--do_train=true"
    "--finetuning_type=lora"
    "--lora_target=all"
    "--dataset=alfworld_train_data_single_0913_format"
    "--template=llama3"
    "--cutoff_len=2048"
    "--overwrite_cache=true"
    "--preprocessing_num_workers=16"
    "--output_dir=${output_dir}"
    "--logging_steps=10"
    "--save_steps=1000"
    "--plot_loss=true"
    "--overwrite_output_dir=true"
    "--per_device_train_batch_size=1"
    "--gradient_accumulation_steps=8"
    "--learning_rate=1.0e-4"
    "--num_train_epochs=2.0"
    "--lr_scheduler_type=cosine"
    "--warmup_ratio=0.1"
    "--bf16=true"
    "--ddp_timeout=180000000"
    "--val_size=0.1"
    "--per_device_eval_batch_size=1"
    "--eval_strategy=steps"
    "--eval_steps=1000"
    "--lora_rank=16"
)

# 运行 Python 脚本
# torchrun /mnt/public/tangyehui/code/LLaMA-Factory/src/train.py "${CONFIG_PARAMS[@]}"
torchrun --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 /mnt/public/tangyehui/code/LLaMA-Factory/src/train.py "${CONFIG_PARAMS[@]}"
#!/bin/bash

# 设置环境变量
export PROJECT_ROOT="/mnt/public/tangyehui/code/LLaMA-Factory"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
export ALFWORLD_DATA=/mnt/public/tangyehui/code/LLaMA-Factory/alfworld_data

# "--model_name_or_path=/home/zy/code/models/llama_31_8b_instruct/"
# "--output_dir=/home/zy/code/LLaMA-Factory/saves/llama3-8b/lora/sft"

# --template=llama3,
# --template=gpt4 \
# 查找 Xvfb 进程
xvfb_pid=$(pgrep -f "Xvfb :2")

if [ -n "$xvfb_pid" ]; then
    echo "找到 Xvfb 进程，PID: $xvfb_pid"
    echo "正在终止 Xvfb 进程..."
    kill -9 $xvfb_pid
    echo "Xvfb 进程已终止"
else
    echo "未找到正在运行的 Xvfb 进程"
fi

# 启动 Xvfb
Xvfb :2 -screen 0 1024x768x16 &

# 设置 DISPLAY 环境变量
export DISPLAY=:1

# 运行 Python 脚本
python /mnt/tangyehui/code/LLaMA-Factory/run_scipts/evaluation/alfworld/eval_alfworld.py \
    --model_name_or_path=/mnt/public/tangyehui/model/llama_31_8b_instruct/ \
    --template=llama3 \
    --config_file=/mnt/tangyehui/code/LLaMA-Factory/run_scipts/evaluation/alfworld/base_config.yaml \
    --save_path=/mnt/tangyehui/code/LLaMA-Factory/saves/llama3-8b/lora/eval_0912_20240912_220122 \
    --adapter_name_or_path=/mnt/tangyehui/code/LLaMA-Factory/saves/llama3-8b/lora/sft_20240914_132522/checkpoint-9000/ \
    --finetuning_type=lora \
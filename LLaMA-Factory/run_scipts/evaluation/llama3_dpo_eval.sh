#!/bin/bash

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

# 设置环境变量
export PROJECT_ROOT="/mnt/public/tangyehui/code/LLaMA-Factory"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
export ALFWORLD_DATA=/mnt/public/tangyehui/code/LLaMA-Factory/alfworld_data

# 启动 Xvfb
Xvfb :2 -screen 0 1024x768x16 &

# 设置 DISPLAY 环境变量
export DISPLAY=:2
save_path="/mnt/tangyehui/code/LLaMA-Factory/saves/llama3-8b/lora/dpo_eval"

current_time=$(date "+%Y%m%d_%H%M%S")
# 定义日志文件路径
log_file="${save_path}/logs/alfworld_eval_${current_time}.log"

mkdir -p "${save_path}/logs"

# 运行 Python 脚本
python /mnt/tangyehui/code/LLaMA-Factory/run_scipts/evaluation/alfworld/eval_alfworld_v1.py \
    --model_name_or_path=/mnt/public/tangyehui/model/llama_31_8b_instruct/ \
    --template=llama3 \
    --config_file=/mnt/tangyehui/code/LLaMA-Factory/run_scipts/evaluation/alfworld/base_config_v1.yaml \
    --save_path="${save_path}" \
    --adapter_name_or_path=/mnt/tangyehui/code/LLaMA-Factory/saves/llama3-8b/lora/dpo_train/dpo_20241024_155656/checkpoint-478/ \
    --finetuning_type=lora \
    2>&1 | tee "${log_file}"



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
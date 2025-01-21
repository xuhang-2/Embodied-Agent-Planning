# 设置环境变量
export PROJECT_ROOT="/mnt/public/tangyehui/code/LLaMA-Factory"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
export OMP_NUM_THREADS=8  # 添加这行来优化性能

# 获取当前时间
current_time=$(date "+%Y%m%d_%H%M%S")

# 更新 output_dir 参数
output_dir="/mnt/tangyehui/code/LLaMA-Factory/saves/qwen2_vl-7b/dpo_train/dpo_${current_time}"

mkdir -p "$output_dir"

# 设置配置参数
CONFIG_PARAMS=(
    "--model_name_or_path=/mnt/tangyehui/model/qwen_2_7b_instruct"
    "--stage=dpo"
    "--do_train=true"
    "--finetuning_type=lora"
    "--lora_target=all"
    "--template=qwen2_vl"
    "--cutoff_len=2048"
    "--overwrite_cache=true"
    "--preprocessing_num_workers=16"
    "--output_dir=${output_dir}"
    "--logging_steps=10"
    "--save_steps=100"
    "--plot_loss=true"
    "--overwrite_output_dir=true"
    "--lr_scheduler_type=cosine"
    "--warmup_ratio=0.1"
    "--bf16=true"
    "--ddp_timeout=180000000"
    "--val_size=0.1"
    "--eval_strategy=steps"
    "--eval_steps=500"
    "--lora_rank=16"
    "--ddp_find_unused_parameters=false"     # 添加这行解决警告
    "--ddp_backend=nccl"                     # 显式指定nccl后端
    "--gradient_checkpointing=false"         # H100显存足够，不需要梯度检查点
    "--max_grad_norm=1.0"                    # 添加梯度裁剪
    "--freeze_vision_tower=true"         
    "--print_param_status=true"
    "--dataset=alfworld_mcts_dpo_mllm"

    "--pref_beta=0.5"
    "--pref_loss=sigmoid"
    "--learning_rate=5.0e-6"
    "--per_device_eval_batch_size=1"         # 增加到4，加快评估速度
    "--per_device_train_batch_size=1"        # 增加到8，因为H100显存充足
    "--gradient_accumulation_steps=4"        # 相应减少到4，保持总batch size
    "--num_train_epochs=1.0"
    "--adapter_name_or_path=/mnt/tangyehui/code/LLaMA-Factory/saves/qwen2_vl-7b/sft_train/sft_20241118_192041/checkpoint-4000"
)

# 运行训练
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    $PROJECT_ROOT/src/train.py "${CONFIG_PARAMS[@]}" 2>&1 | tee "$output_dir/train_log.logs"



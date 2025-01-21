---
library_name: peft
license: other
base_model: /mnt/tangyehui/model/qwen_2_7b_instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: dpo_20241205_144915
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# dpo_20241205_144915

This model is a fine-tuned version of [/mnt/tangyehui/model/qwen_2_7b_instruct](https://huggingface.co//mnt/tangyehui/model/qwen_2_7b_instruct) on the alfworld_mcts_dpo_mllm dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8797
- Rewards/chosen: 0.6689
- Rewards/rejected: -0.8016
- Rewards/accuracies: 0.5921
- Rewards/margins: 1.4704
- Logps/rejected: -7.4001
- Logps/chosen: -3.8593
- Logits/rejected: -1.1719
- Logits/chosen: -1.1708

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-06
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 4
- total_train_batch_size: 8
- total_eval_batch_size: 2
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 1.0

### Training results



### Framework versions

- PEFT 0.12.0
- Transformers 4.45.0
- Pytorch 2.5.1+cu124
- Datasets 2.21.0
- Tokenizers 0.20.3
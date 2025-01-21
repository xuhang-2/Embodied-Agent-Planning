---
library_name: peft
license: other
base_model: /mnt/tangyehui/model/qwen_2_7b_instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: sft_20241113_154436
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft_20241113_154436

This model is a fine-tuned version of [/mnt/tangyehui/model/qwen_2_7b_instruct](https://huggingface.co//mnt/tangyehui/model/qwen_2_7b_instruct) on the alfworld_qwen_correct_sft_single dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0197

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- total_eval_batch_size: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 2.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.0342        | 0.8157 | 500  | 0.0343          |
| 0.0223        | 1.6313 | 1000 | 0.0223          |


### Framework versions

- PEFT 0.12.0
- Transformers 4.45.0
- Pytorch 2.5.1+cu124
- Datasets 2.21.0
- Tokenizers 0.20.3
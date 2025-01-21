---
library_name: peft
license: other
base_model: /mnt/tangyehui/model/qwen_2_7b_instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: sft_20241118_192041
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft_20241118_192041

This model is a fine-tuned version of [/mnt/tangyehui/model/qwen_2_7b_instruct](https://huggingface.co//mnt/tangyehui/model/qwen_2_7b_instruct) on the alfworld_qwen_correct_sft_single, the alfworld_qwen_mcts_sft_single and the alfworld_qwen_correct_fake_data_sft_single datasets.
It achieves the following results on the evaluation set:
- Loss: 0.0093

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
- num_epochs: 4.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.0293        | 0.4649 | 500  | 0.0321          |
| 0.0261        | 0.9298 | 1000 | 0.0249          |
| 0.0177        | 1.3947 | 1500 | 0.0189          |
| 0.0132        | 1.8596 | 2000 | 0.0136          |
| 0.0098        | 2.3245 | 2500 | 0.0112          |
| 0.0083        | 2.7894 | 3000 | 0.0099          |
| 0.0058        | 3.2543 | 3500 | 0.0097          |
| 0.0069        | 3.7192 | 4000 | 0.0094          |


### Framework versions

- PEFT 0.12.0
- Transformers 4.45.0
- Pytorch 2.5.1+cu124
- Datasets 2.21.0
- Tokenizers 0.20.3
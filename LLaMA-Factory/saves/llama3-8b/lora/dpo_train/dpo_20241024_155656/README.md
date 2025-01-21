---
base_model: /mnt/public/tangyehui/model/llama_31_8b_instruct/
library_name: peft
license: other
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: dpo_20241024_155656
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# dpo_20241024_155656

This model is a fine-tuned version of [/mnt/public/tangyehui/model/llama_31_8b_instruct/](https://huggingface.co//mnt/public/tangyehui/model/llama_31_8b_instruct/) on the alfworld_mcts_dpo_v2_multiround dataset.
It achieves the following results on the evaluation set:
- Loss: 1.0348
- Rewards/chosen: -3.5334
- Rewards/rejected: -4.4577
- Rewards/accuracies: 0.6385
- Rewards/margins: 0.9243
- Logps/rejected: -20.4199
- Logps/chosen: -16.6997
- Logits/rejected: 0.1368
- Logits/chosen: 0.1357

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
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 2.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Rewards/chosen | Rewards/rejected | Rewards/accuracies | Rewards/margins | Logps/rejected | Logps/chosen | Logits/rejected | Logits/chosen |
|:-------------:|:------:|:----:|:---------------:|:--------------:|:----------------:|:------------------:|:---------------:|:--------------:|:------------:|:---------------:|:-------------:|
| 1.1992        | 0.4180 | 100  | 1.2864          | -1.7637        | -2.5633          | 0.6056             | 0.7996          | -16.6312       | -13.1603     | 0.1298          | 0.1287        |
| 1.1987        | 0.8359 | 200  | 1.1421          | -2.1521        | -3.1465          | 0.6244             | 0.9944          | -17.7976       | -13.9370     | 0.1352          | 0.1341        |
| 1.0782        | 1.2539 | 300  | 1.0485          | -2.8497        | -3.7887          | 0.6244             | 0.9390          | -19.0819       | -15.3322     | 0.1380          | 0.1369        |
| 0.705         | 1.6719 | 400  | 1.0352          | -3.3824        | -4.2968          | 0.6385             | 0.9144          | -20.0982       | -16.3977     | 0.1367          | 0.1357        |


### Framework versions

- PEFT 0.12.0
- Transformers 4.44.2
- Pytorch 2.4.0+cu121
- Datasets 2.21.0
- Tokenizers 0.19.1
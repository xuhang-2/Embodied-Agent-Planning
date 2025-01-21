---
base_model: /mnt/public/tangyehui/model/llama_31_8b_instruct/
library_name: peft
license: other
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: sft_20240914_132522
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft_20240914_132522

This model is a fine-tuned version of [/mnt/public/tangyehui/model/llama_31_8b_instruct/](https://huggingface.co//mnt/public/tangyehui/model/llama_31_8b_instruct/) on the alfworld_train_data_single_0913_format dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0098

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

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.0438        | 0.2059 | 1000 | 0.0601          |
| 0.0446        | 0.4119 | 2000 | 0.0345          |
| 0.0128        | 0.6178 | 3000 | 0.0211          |
| 0.002         | 0.8237 | 4000 | 0.0159          |
| 0.0018        | 1.0297 | 5000 | 0.0139          |
| 0.0151        | 1.2356 | 6000 | 0.0125          |
| 0.0134        | 1.4416 | 7000 | 0.0105          |
| 0.0076        | 1.6475 | 8000 | 0.0102          |
| 0.0046        | 1.8534 | 9000 | 0.0099          |


### Framework versions

- PEFT 0.12.0
- Transformers 4.44.2
- Pytorch 2.4.1+cu121
- Datasets 2.21.0
- Tokenizers 0.19.1
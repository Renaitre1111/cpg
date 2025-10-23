# Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning
Yaxin Hou, Bo Han, Yuheng Jia, Hui Liu, Junhui Hou, Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning, Advances in Neural Information Processing Systems, 2nd-7th December, San Diego, 2025.

This is an official [PyTorch](http://pytorch.org) implementation for **Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning**.

## Introduction
This code is based on the public and widely-used codebase [USB](https://github.com/microsoft/Semi-supervised-learning).

What I've done is just adding our CPG algorithm in `semilearn/imb_algorithms/cpg`.

Also, I've made corresponding modifications to `semilearn/nets/` and several `__init__.py`.

## How to run
For example, on CIFAR-10-LT with long-tailed labeled data ($\gamma_l=100$) and arbitrary unlabeled data($\gamma_u=100$)

```
CUDA_VISIBLE_DEVICES=0 python train.py --c "config/config-1/cpg/203-fixmatch_cpg_cifar10_lb400_100_ulb4600_100_random_0.0_1.yaml"
```

(Note: I know that USB supports multi-GPUs, but I still recommend you to run on single GPU, as some weird problems may occur.)

The model will be automatically evaluated every 1024 iterations during training. After training, the last two lines in `/saved_models/cpg/203-fixmatch_cpg_cifar10_lb400_100_ulb4600_100_random_0.0_1/log.txt` will tell you the best accuracy. 

For example,
```
[2025-04-20 13:35:17,784 INFO] model saved: ./saved_models/cpg/203-fixmatch_cpg_cifar10_lb400_100_ulb4600_100_random_0.0_1/latest_model.pth
[2025-04-20 13:35:17,815 INFO] Model result - eval/best_acc : 0.8233
[2025-04-20 13:35:17,816 INFO] Model result - eval/best_it : 244735
```

## Results

## generate
'''
accelerate launch generate/train_conditional.py \
  --train_data_dir="generate/cifar10/train" \
  --model_config_name_or_path="generate/config.json" \
  --resolution=32 \
  --output_dir="generate/pretrained_weights_cifar10" \
  --train_batch_size=1024 \
  --dataloader_num_workers=20 \
  --eval_batch_size=10 \
  --num_epochs=2000 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=1000 \
  --mixed_precision="bf16" \
  --save_images_epochs=20 \
  --ddpm_beta_schedule="squaredcos_cap_v2" \
  --checkpointing_steps=1000 \
  --resume_from_checkpoint="latest" \
  --num_classes=10 \
  --prediction_type="epsilon" \
  --logger="tensorboard"
'''

'''
python generate/sampling.py 
'''

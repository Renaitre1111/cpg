import argparse
import os
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torchvision import transforms
from tqdm.auto import tqdm
from huggingface_hub import HfFolder, Repository, whoami

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional diffusion model on CIFAR-10.")
    parser.add_argument("--dataset_name", type=str, default="cifar10", help="Dataset name.")
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory for the dataset.")
    parser.add_argument("--output_dir", type=str, default="pretrain_weights/ddpm-conditional-cifar10", help="Output directory for logs and checkpoints.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the output directory if it exists.")
    parser.add_argument("--resolution", type=int, default=32, help="Image resolution.")
    parser.add_argument("--train_batch_size", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=256, help="Batch size for evaluation/sampling.")
    parser.add_argument("--num_epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="Learning rate scheduler type ('linear', 'cosine', 'constant').") 
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps for the learning rate scheduler.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam optimizer beta1.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam optimizer beta2.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6, help="Adam optimizer weight decay.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Adam optimizer epsilon.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps for gradient accumulation.")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Use mixed precision training.")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="Decay rate for EMA weights (optional).")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use Exponential Moving Average of models weights.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for DataLoader.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Logging directory (sub-directory of output_dir).")
    parser.add_argument("--save_interval_steps", type=int, default=1000, help="Save checkpoints every N steps.")
    parser.add_argument("--save_total_limit", type=int, default=5, help="Limit the total number of checkpoints.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model checkpoints to Hugging Face Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Repository ID on Hugging Face Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="Authentication token for Hugging Face Hub.")

    # --- Model Specific Args ---
    parser.add_argument(
        "--block_out_channels",
        type=int,
        nargs="+",
        default=[128, 128, 256, 256], # Common choice for CIFAR-10 32x32
        help="Output channels for each UNet block."
    )
    parser.add_argument(
        "--down_block_types",
        type=str,
        nargs="+",
        default=["DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"],
        help="Types of down blocks in the UNet."
    )
    parser.add_argument(
        "--up_block_types",
        type=str,
        nargs="+",
        default=["UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"],
        help="Types of up blocks in the UNet."
    )
    # --- Conditional Specific Arg ---
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes for conditioning (CIFAR-10 has 10).")


    args = parser.parse_args()

    # Create output dir path
    args.output_dir = os.path.join(args.output_dir, f"lr_{args.learning_rate}_bs_{args.train_batch_size}")
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None,
        project_dir=os.path.join(args.output_dir, args.logging_dir)
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            if args.overwrite_output_dir:
                 pass

        if args.push_to_hub:
            repo = Repository(args.output_dir, clone_from=args.hub_model_id, token=args.hub_token)
            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")


    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Normalize to [-1, 1] range
        ]
    )
    dataset = torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=False, transform=preprocess)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers
    )

    # --- Initialize Model and Noise Scheduler ---
    model = UNet2DModel(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=tuple(args.block_out_channels),
        down_block_types=tuple(args.down_block_types),
        up_block_types=tuple(args.up_block_types),
        num_class_embeds=args.num_classes, # Crucial for class conditioning
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2") # or linear

    # --- Initialize Optimizer and LR Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_decay,
            device=accelerator.device,
            dtype=torch.float32     
        )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("train_example", config=vars(args)) # Adjust tracker name

    # --- Training Loop ---
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}")

        for step, batch in enumerate(train_dataloader):
            clean_images, labels = batch
            clean_images = clean_images.to(accelerator.device)
            labels = labels.to(accelerator.device)

            noise = torch.randn(clean_images.shape).to(clean_images.device)

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # --- Predict the noise residual ---
                # Pass class labels for conditioning
                noise_pred = model(noisy_images, timesteps, class_labels=labels).sample

                # --- Calculate Loss (MSE between predicted and actual noise) ---
                loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0) # Optional: Gradient Clipping

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # --- Update EMA ---
            if args.use_ema:
                ema_model.step(model.parameters())

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

            # --- Save Checkpoints ---
            if accelerator.is_main_process:
                 if global_step % args.save_interval_steps == 0:
                     save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                     accelerator.save_state(save_path)
                     logger.info(f"Saved state to {save_path}")

                     # Optionally save model directly for easier inference pipeline loading
                     # unwrap_model = accelerator.unwrap_model(model)
                     # unwrap_model.save_pretrained(os.path.join(args.output_dir, f"unet-{global_step}"))

                     # Optional: Push to Hub
                     if args.push_to_hub:
                         repo.push_to_hub(commit_message=f"Step {global_step}", blocking=False)


        progress_bar.close()
        # --- End of Epoch ---
        # Optional: Add evaluation/sampling logic here
        # E.g., generate images for each class using the current model/EMA model

    accelerator.wait_for_everyone()

    # --- Save Final Model ---
    if accelerator.is_main_process:
        # Save the final trained model
        unwrap_model = accelerator.unwrap_model(model)
        unwrap_model.save_pretrained(os.path.join(args.output_dir, "unet_final"))
        # Optional: Save EMA weights
        if args.use_ema:
            ema_model.copy_to(unwrap_model.parameters())
            unwrap_model.save_pretrained(os.path.join(args.output_dir, "unet_final_ema"))

        if args.push_to_hub:
             repo.push_to_hub(commit_message="End of training", blocking=True)


    accelerator.end_training()


if __name__ == "__main__":
    main()
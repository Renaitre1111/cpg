import argparse
import os
from datetime import datetime

import torch
from diffusers import UNet2DModel, DDPMScheduler
from torchvision.utils import save_image

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

def parse_args():
    p = argparse.ArgumentParser("Conditional DDPM CIFAR-10 Inference Demo")
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="Path to training output dir (contains unet_final_ema/ or unet_final/).")
    p.add_argument("--num_inference_steps", type=int, default=1000,
                   help="DDPM reverse steps. 1000 matches training.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--outdir", type=str, default="samples", help="Output image directory.")
    p.add_argument("--use_ema_first", action="store_true", default=True,
                   help="Prefer EMA weights if available.")
    p.add_argument("--per_class", type=int, default=1, help="Samples per class.")
    p.add_argument("--resolution", type=int, default=32, help="Image size; must match training.")
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    ema_dir = os.path.join(args.ckpt_dir, "unet_final_ema")
    plain_dir = os.path.join(args.ckpt_dir, "unet_final")
    load_dir = None
    if args.use_ema_first and os.path.isdir(ema_dir):
        load_dir = ema_dir
    elif os.path.isdir(plain_dir):
        load_dir = plain_dir
    elif os.path.isdir(args.ckpt_dir):
        load_dir = args.ckpt_dir
    else:
        raise FileNotFoundError(f"Could not find model in: {ema_dir} or {plain_dir}")
    
    print(f"Loading UNet from: {load_dir}")
    unet = UNet2DModel.from_pretrained(load_dir).to(device)
    unet.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    scheduler.set_timesteps(args.num_inference_steps, device=device)

    classes = list(range(10))
    labels = []
    for c in classes:
        labels += [c] * args.per_class
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    batch_size = labels.shape[0]
    img_size = args.resolution
    generator = torch.Generator(device=device).manual_seed(args.seed)

    x = torch.randn((batch_size, 3, img_size, img_size), device=device, generator=generator)

    for t in scheduler.timesteps:
        noise_pred = unet(x, t, class_labels=labels).sample
        step = scheduler.step(noise_pred, t, x)
        x = step.prev_sample

    imgs = (x.clamp(-1, 1) + 1) / 2.0
    nrow = len(classes)
    imgs_reshaped = imgs.view(len(classes), args.per_class, 3, img_size, img_size)
    imgs_grid_order = imgs_reshaped.permute(1, 0, 2, 3, 4).reshape(-1, 3, img_size, img_size)

    os.makedirs(args.outdir, exist_ok=True)
    imgs_grid_order = imgs_grid_order.detach().cpu()
    for i, img in enumerate(imgs_grid_order):
        save_image(img, os.path.join(args.outdir, f"{i:04d}.png"))

if __name__ == "__main__":
    main()

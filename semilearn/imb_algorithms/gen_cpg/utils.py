import torch
import torch.nn.functinal as F
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from typing import List, Optional, Tuple, Union
import math
from PIL import Image

# Reliable Sample Screening Module
@torch.no_grad()
def get_max_confidence_and_residual_variance(predictions):
    # predictions [N, C]
    # max_confidence [N], max_indices [N]
    # Step 1: Calculate the maximum confidence and corresponding class
    max_confidence, max_indices = torch.max(predictions, dim=1)
    
    # Step 2: Create a mask to exclude the maximum confidence class from each predictionß
    one_hot_max = F.one_hot(max_indices, num_classes=predictions.shape[1]) # [N, C]
    
    # Step 3: Mask out the maximum prediction by multiplying with (1 - one_hot_max)
    remaining_predictions = predictions * (1 - one_hot_max)
    
    # Step 4: Compute the mean of the remaining predictions
    sum_remaining_predictions = torch.sum(remaining_predictions, dim=1)
    num_remaining_classes = (predictions.shape[1] - 1)
    mean_remaining_predictions = sum_remaining_predictions / num_remaining_classes

    # Step 5: Calculate variance for remaining classes
    remaining_predictions_diff = remaining_predictions - mean_remaining_predictions.unsqueeze(1)
    remaining_predictions_squared_diff = remaining_predictions_diff ** 2 # [N, C]

    # Sum the squared differences and divide by the number of remaining classes to get the variance
    sum_squared_diff = torch.sum(remaining_predictions_squared_diff, dim=1)
    residual_variance = sum_squared_diff / num_remaining_classes
    return max_confidence, residual_variance

@torch.no_grad()
def batch_class_stats(max_conf, res_var):
    # max_conf [N], res_var [N]
    features = torch.stack([max_conf, res_var], dim=-1) # [N, 2]
    valid_mask = ~torch.isnan(features).any(dim=-1)
    valid_features = features[valid_mask] # [N', 2]

    if valid_mask.size(0) == 0:
        selected_mean = torch.tensor((1.0, 0.0), device=max_conf.device)
        selected_var = torch.tensor((1.0, 1.0), device=max_conf.device)
        return selected_mean, selected_var
    
    class_assignments = _class_assignment(valid_features, 2)
    class_centers = _compute_class_centers(valid_features, class_assignments, 2)
    max_mean_idx = torch.argmax(class_centers[0][:, 0])
    selected_mean = class_centers[0][max_mean_idx]
    selected_var = class_centers[1][max_mean_idx]
    return selected_mean, selected_var

@torch.no_grad()
def _compute_eigenvectors_with_svd(X, num_classes):
    U, S, Vt = torch.linalg.svd(X.T, full_matrices=False)
    eigvals = S ** 2 
    idx = torch.argsort(-eigvals) 
    eigvecs = Vt.T[:, idx[:num_classes]]  
    return eigvecs

@torch.no_grad()
def _class_assignment(input, num_classes):
    eigenvectors = _compute_eigenvectors_with_svd(input, num_classes)
    class_assignments = torch.argmax(torch.abs(eigenvectors), dim=1)
    return class_assignments

@torch.no_grad()
def _compute_class_centers(features, class_assignments, num_classes):
    means = []
    vars = []
    for class_id in range(num_classes):
        points_in_class = features[class_assignments == class_id]
        num_points = points_in_class.size(0)
        if num_points == 0:
            mean = torch.zeros(features.size(1), device=features.device)
            var = torch.zeros(features.size(1), device=features.device)
        elif num_points == 1:
            mean = points_in_class.squeeze(0)
            var = torch.zeros(features.size(1), device=features.device)
        else:
            mean = points_in_class.mean(dim=0)
            var = points_in_class.var(dim=0, unbiased=True)
        means.append(mean)
        vars.append(var)
    return torch.stack(means), torch.stack(vars)


# Diffusion Sampling Module
class DDPMPipelinenew(DiffusionPipeline):
    def __init__(self, unet, scheduler, num_classes: int):
        super.__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.num_classes = num_classes
        self._device = unet.device  # Ensure the pipeline knows the device
    
    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 64,
        class_labels: Optional[torch.Tensor] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        class_labels = class_labels.to(self._device)
        if class_labels.ndim == 0:
            class_labels = class_labels.unsqueeze(0).expand(batch_size)
        else:
            class_labels = class_labels.expand(batch_size)
        
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        image = randn_tensor(image_shape, generator=generator, device=self._device)

        # Set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # Ensure the class labels are correctly broadcast to match the input tensor shape
            model_output = self.unet(image, t, class_labels).sample

            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
    
def load_diffusion_pipeline(model_path, scheduler_path, num_classes, device):
    unet = UNet2DModel.from_pretrained(model_path).to(device)
    scheduler = DDPMScheduler.from_pretrained(scheduler_path)
    pipeline = DDPMPipelinenew(unet=unet, scheduler=scheduler, num_classes=num_classes)
    return pipeline.to(device)  # Move the entire pipeline to the device

@torch.no_grad()
def sample_diffusion_data(pipeline, classes_to_sample, num_samples_per_class: List[int], batch_size=64, num_inference_steps=1000):
    all_gen_imgs = []
    all_gen_labels = []
    device = pipeline.device

    print(f"sampling for {len(classes_to_sample)} classes")

    for class_label, num_samples in zip(classes_to_sample, num_samples_per_class):
        if num_samples <= 0:
            continue

        num_generated = 0
        num_batches = math.ceil(num_samples / batch_size)
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - num_generated)
            if current_batch_size <= 0:
                continue
            labels = torch.tensor([class_label] * current_batch_size, device=device, dtype=torch.long)
            imgs_output = pipeline(
                batch_siz=current_batch_size,
                class_labels=labels,
                num_inference_steps=num_inference_steps,
                output_type="pil"
            )

            gen_imgs_batch = imgs_output.images

            all_gen_imgs.extend(gen_imgs_batch)
            all_gen_labels.extend([class_label] * current_batch_size)
            num_generated += current_batch_size
        
    return all_gen_imgs, all_gen_labels

        


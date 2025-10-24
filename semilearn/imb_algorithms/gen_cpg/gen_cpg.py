# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import TensorDataset, ConcatDataset
from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.core.utils import get_data_loader
from semilearn.datasets.augmentation import RandAugment
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
from .utils import get_max_confidence_and_residual_variance, batch_class_stats, load_diffusion, sample_guided

def get_weights(pred, valid_mask, alpha, epsilon=1e-8):
    weight = torch.zeros_like(pred[:, 0]) # [B]
    max_confidence, residual_variance = get_max_confidence_and_residual_variance(pred)
    if valid_mask.sum() > 0:
        means, vars = batch_class_stats(
            max_confidence[valid_mask],
            residual_variance[valid_mask]
        )
    else:
        means = torch.tensor([1.0, 0.0], device=pred.device)
        vars = torch.tensor([1.0, 1.0], device=pred.device)

    conf_mean = means[0]
    res_mean = means[1]
    conf_var = vars[0]
    res_var = vars[1]

    conf_z = (max_confidence - conf_mean) / torch.sqrt(conf_var + epsilon) # [N]
    res_z = (residual_variance - res_mean) / torch.sqrt(res_var + epsilon) # [N]

    weight_conf = torch.exp(-(conf_z ** 2) / alpha)
    weight_res = torch.exp(-(res_z ** 2) / alpha)

    weight = weight_conf * weight_res # [N]
    confidence_mask = (conf_z > 0) & (res_z > 0) # [N]
    weight = torch.where(confidence_mask, torch.ones_like(weight), weight)

    final_weight = torch.where(valid_mask, weight, torch.zeros_like(weight))

    return final_weight


@IMB_ALGORITHMS.register('gen_cpg')
class Gen_CPG(ImbAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super(Gen_CPG, self).__init__(args, net_builder, tb_log, logger)

        # warm_up epoch
        self.gen_start_epoch = args.gen_start_epoch
        self.gen_mid_epoch = args.gen_mid_epoch
        self.ulb_start_epoch = args.ulb_start_epoch

        # diffusion hyperparameters
        self.head_threshold = args.head_threshold
        self.mid_threshold = args.mid_threshold
        self.gen_target_count = args.gen_target_count
        self.gen_batch_size = args.gen_batch_size
        self.gen_steps = args.gen_steps
        self.guidance_scale = args.guidance_scale

        self.unet_model, self.scheduler = load_diffusion(model_id=args.model_id, device=args.gpu)
        self.generated_dataset_moderate = None
        self.generated_dataset_extreme = None

        # dataset update step
        if args.dataset == 'cifar10':
            self.update_step = 5
            self.memory_step = 5
        if args.dataset == 'cifar100':
            self.update_step = 5
            self.memory_step = 5
        if args.dataset == 'food101':
            self.update_step = 5
            self.memory_step = 5
        if args.dataset == 'svhn':
            self.update_step = 5
            self.memory_step = 5

        # adaptive labeled (include the pseudo labeled) data and its dataloader
        self.current_x = None # current labeled image (include the pseudo labeled image)
        self.current_y = None # curren label (include the pseudo label)
        self.current_idx = None
        self.current_noise_y = None
        self.current_one_hot_y = None
        self.current_one_hot_noise_y = None

        self.select_ulb_idx = None
        self.select_ulb_label = None
        self.select_ulb_pseudo_label = None
        self.select_ulb_pseudo_label_distribution = None

        self.adaptive_lb_dest = None
        self.adaptive_lb_dest_loader = None

        self.dataset = args.dataset
        self.data = self.dataset_dict['data']
        self.targets = self.dataset_dict['targets']
        self.noised_targets = self.dataset_dict['noised_targets']
        self.lb_idx =  self.dataset_dict['lb_idx']
        self.ulb_idx =  self.dataset_dict['ulb_idx']

        self.mean, self.std = {}, {}

        self.mean['cifar10'] = [0.485, 0.456, 0.406]
        self.mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
        self.mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]
        self.mean['svhn'] = [0.4380, 0.4440, 0.4730]
        self.mean['food101'] = [0.485, 0.456, 0.406]

        self.std['cifar10'] = [0.229, 0.224, 0.225]
        self.std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]
        self.std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]
        self.std['svhn'] = [0.1751, 0.1771, 0.1744]
        self.std['food101'] = [0.229, 0.224, 0.225]

        if self.dataset == 'food101':
            self.transform_weak = transforms.Compose([
                                # transforms.Resize(args.img_size),
                                transforms.RandomCrop((args.img_size, args.img_size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])

            self.transform_strong = transforms.Compose([
                                # transforms.Resize(args.img_size),
                                transforms.RandomCrop((args.img_size, args.img_size)),
                                transforms.RandomHorizontalFlip(),
                                RandAugment(3, 5),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])
        else:
            self.transform_weak = transforms.Compose([
                                transforms.Resize(args.img_size),
                                transforms.RandomCrop(args.img_size, padding=int(args.img_size * (1 - args.crop_ratio)), padding_mode='reflect'),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])

            self.transform_strong = transforms.Compose([
                                transforms.Resize(args.img_size),
                                transforms.RandomCrop(args.img_size, padding=int(args.img_size * (1 - args.crop_ratio)), padding_mode='reflect'),
                                transforms.RandomHorizontalFlip(),
                                RandAugment(3, 5),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])

        # compute lb dist
        lb_class_dist = [0 for _ in range(self.num_classes)]
        if args.noise_ratio > 0:
            for c in self.dataset_dict['train_lb'].noised_targets:
                lb_class_dist[c] += 1
        else:
            for c in self.dataset_dict['train_lb'].targets:
                lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)

        self.lb_dist = torch.from_numpy(lb_class_dist.astype(np.float32)).cuda(args.gpu)

        # compute select_ulb and ulb dist
        ulb_class_dist = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_ulb'].targets:
            ulb_class_dist[c] += 1
        ulb_class_dist = np.array(ulb_class_dist)

        self.ulb_dist = torch.from_numpy(ulb_class_dist.astype(np.float32)).cuda(args.gpu)

        self.select_ulb_dist = torch.zeros(self.num_classes).cuda(args.gpu)

        # compute lb_select_ulb and lb_ulb dist
        lb_ulb_class_dist = lb_class_dist + ulb_class_dist

        self.lb_ulb_dist = torch.from_numpy(lb_ulb_class_dist.astype(np.float32)).cuda(args.gpu)

        self.lb_select_ulb_dist = self.lb_dist + self.select_ulb_dist

    def _generate_data(self, target_min_count, target_max_count, target_total_count):
        """
        generate data function
        """
        lb_counts = self.lb_dist.cpu().numpy()
        all_gen_imgs = []
        all_gen_labels = []

        for class_idx in range(self.num_classes):
            current_count = int(lb_counts[class_idx])

            if target_min_count <= current_count <= target_max_count:
                num_to_gen = max(0, target_total_count - current_count)
                
                if num_to_gen <= 0:
                    continue
                generated_count = 0
                while generated_count < num_to_gen:
                    batch_to_gen = min(self.gen_batch_size, num_to_gen - generated_count)
                    if batch_to_gen <= 0:
                        break

                    pil_images = sample_guided(
                        unet_model=self.unet_model,
                        scheduler=self.scheduler,
                        classifier=self.model,
                        class_label=class_idx,
                        batch_size=batch_to_gen,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.gen_steps,
                        device=self.args.gpu
                    )

                    if not isinstance(pil_images, list):
                        pil_images = [pil_images]

                    for img in pil_images:
                        tensor_img = self.transform_weak(img)
                        all_gen_imgs.append(tensor_img)
                    
                    labels = torch.full((batch_to_gen,), class_idx, dtype=torch.long)
                    all_gen_labels.append(labels)
                    generated_count += batch_to_gen
        
        if all_gen_imgs:
            all_gen_imgs = torch.stack(all_gen_imgs)
            all_gen_labels = torch.stack(all_gen_labels)
            all_gen_labels_one_hot = F.one_hot(all_gen_labels, num_classes=self.num_classes).float()

            dummy_indices = np.arange(len(all_gen_imgs)) * -1 -1

            generated_dataset = BasicDataset(
                dummy_indices,
                all_gen_imgs,
                all_gen_labels,
                all_gen_labels_one_hot,
                self.num_classes,
                False,
                weak_transform=None,
                strong_transform=None,
                one_hot=False
            )

            return generated_dataset
        
        return None
    
    def _update_adaptive_loader(self):
        """
        update adaptive loader function
        """
        if self.epoch >= self.ulb_start_epoch:
            self.current_idx = np.concatenate((self.lb_idx, self.select_ulb_idx), axis=0)
            self.current_x = self.data[self.current_idx]
            self.current_y = np.concatenate((self.targets[self.lb_idx], self.select_ulb_pseudo_label), axis=0)
            self.current_noise_y = np.concatenate((self.noised_targets[self.lb_idx], self.select_ulb_pseudo_label), axis=0)
            current_one_hot_y = np.full((len(self.targets[self.lb_idx]), self.num_classes), self.args.smoothing / (self.num_classes - 1))
            current_one_hot_y[np.arange(len(self.targets[self.lb_idx])), self.targets[self.lb_idx]] = 1.0 - self.args.smoothing
            current_one_hot_noise_y = np.full((len(self.noised_targets[self.lb_idx]), self.num_classes), self.args.smoothing / (self.num_classes - 1))
            current_one_hot_noise_y[np.arange(len(self.noised_targets[self.lb_idx])), self.noised_targets[self.lb_idx]] = 1.0 - self.args.smoothing
            self.current_one_hot_y = np.concatenate((current_one_hot_y, self.select_ulb_pseudo_label_distribution), axis=0)
            self.current_one_hot_noise_y = np.concatenate((current_one_hot_noise_y, self.select_ulb_pseudo_label_distribution), axis=0)
        else:
            self.current_idx = self.lb_idx
            self.current_x = self.data[self.current_idx]
            self.current_y = self.targets[self.current_idx]
            self.current_noise_y = self.noised_targets[self.current_idx]
            self.current_one_hot_y = np.full((len(self.targets[self.lb_idx]), self.num_classes), self.args.smoothing / (self.num_classes - 1))
            self.current_one_hot_y[np.arange(len(self.targets[self.lb_idx])), self.targets[self.lb_idx]] = 1.0 - self.args.smoothing
            self.current_one_hot_noise_y = np.full((len(self.noised_targets[self.lb_idx]), self.num_classes), self.args.smoothing / (self.num_classes - 1))
            self.current_one_hot_noise_y[np.arange(len(self.noised_targets[self.lb_idx])), self.noised_targets[self.lb_idx]] = 1.0 - self.args.smoothing

        base_dataset = BasicDataset(self.current_idx, self.current_x, self.current_one_hot_y, self.current_one_hot_noise_y, self.num_classes, False, weak_transform=self.transform_weak, strong_transform=self.transform_strong, one_hot=False)
        
        datasets_to_concat = [base_dataset]
        if self.generated_dataset_moderate is not None:
            datasets_to_concat.append(self.generated_dataset_moderate)
        if self.generated_dataset_extreme is not None:
            datasets_to_concat.append(self.generated_dataset_extreme)
        
        final_adaptive_dataset = ConcatDataset(datasets_to_concat)
        self.adaptive_lb_dest_loader = get_data_loader(self.args, final_adaptive_dataset, self.args.batch_size,
                                                       data_sampler=self.args.train_sampler, num_iters=self.num_train_iter,
                                                       num_epochs=self.epochs, num_workers=self.args.num_workers,
                                                       distributed=self.distributed)
        
        self.current_x = None
        self.current_y = None
        self.current_idx = None
        self.current_noise_y = None
        self.current_one_hot_y = None
        self.current_one_hot_noise_y = None
    
    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch

            # ~self.gen_start_epoch ce loss only and not select unlabeled data
            if self.epoch < self.gen_start_epoch:
                self.adaptive_lb_dest_loader = self.loader_dict['train_lb']
                self.lb_select_ulb_dist = self.lb_dist
                self.select_ulb_dist = torch.ones(self.num_classes).cuda(self.args.gpu)

            # self.warm_up select unlabeled data but still use labeled data only to compute loss
            elif self.epoch == self.gen_start_epoch:
                self.print_fn("Starting Diffusion Generation Phase 1 (Moderate Tail)")

                self.generated_dataset_moderate = self._generate_data(
                    target_min_count=self.mid_threshold, 
                    target_max_count=self.head_threshold,
                    target_total_count=self.gen_target_count
                )

                self._update_adaptive_loader()

                self.lb_select_ulb_dist = self.lb_dist
                self.select_ulb_dist = torch.ones(self.num_classes).cuda(self.args.gpu)

            elif self.epoch == self.gen_mid_epoch:
                self.print_fn("Starting Diffusion Generation Phase 2 (Extreme Tail)")

                self.generated_dataset_extreme = self._generate_data(
                    target_min_count=0,
                    target_max_count=self.mid_threshold,
                    target_total_count=self.gen_target_count
                )

                self._update_adaptive_loader()

            # self.warm_up+1~ use labeled (include the pseudo labeled) data and continue select unlabeled data
            # update the labeled (include the pseudo labeled) dataset and labeled (include the pseudo labeled) data distribution and selected unlabeled data distribution
            else:
                if self.epoch >= self.ulb_start_epoch:
                    if self.epoch % self.memory_step == 0:
                        self.current_x = None
                        self.current_y = None
                        self.current_idx = None
                        self.current_noise_y = None
                        self.current_one_hot_y = None
                        self.current_one_hot_noise_y = None
                        self.select_ulb_pseudo_label_distribution = None

                        # process selected condident unlabeled data
                        # delete the same idx / same data contribution to gradient once
                        select_ulb_idx_to_label = {}
                        select_ulb_idx_to_pseudo_label = {}

                        for ulb_idx, ulb_pseudo_label, ulb_label in zip(self.select_ulb_idx, self.select_ulb_pseudo_label, self.select_ulb_label):
                            if ulb_idx.item() in select_ulb_idx_to_label:
                                select_ulb_idx_to_label[ulb_idx.item()].append(ulb_label.item())
                            else:
                                select_ulb_idx_to_label[ulb_idx.item()] = [ulb_label.item()]

                            if ulb_idx.item() in select_ulb_idx_to_pseudo_label:
                                select_ulb_idx_to_pseudo_label[ulb_idx.item()].append(ulb_pseudo_label.item())
                            else:
                                select_ulb_idx_to_pseudo_label[ulb_idx.item()] = [ulb_pseudo_label.item()]

                        select_ulb_unique_idx = torch.unique(self.select_ulb_idx)

                        mean_number_of_pseudo_label = []

                        for ulb_unique_idx in select_ulb_unique_idx:
                            mean_number_of_pseudo_label.append(len(select_ulb_idx_to_label[ulb_unique_idx.item()]))

                        select_ulb_unique_label = []
                        select_ulb_unique_pseudo_label = []
                        select_ulb_unique_pseudo_label_distribution = []

                        for ulb_unique_idx in select_ulb_unique_idx:
                            ulb_unique_label = select_ulb_idx_to_label[ulb_unique_idx.item()]
                            ulb_unique_pseudo_label = select_ulb_idx_to_pseudo_label[ulb_unique_idx.item()]

                            ulb_unique_pseudo_label_distribution = torch.zeros(self.num_classes)
                            for item in ulb_unique_pseudo_label:
                                ulb_unique_pseudo_label_distribution[item] += 1.0
                            ulb_unique_pseudo_label_distribution = ulb_unique_pseudo_label_distribution / torch.sum(ulb_unique_pseudo_label_distribution)

                            # process the ground-truth label                                                        
                            select_ulb_unique_label.append(torch.tensor([ulb_unique_label[0]]))

                            # process the pseudo-label
                            if len(ulb_unique_pseudo_label) > 12:
                                most_common_label = Counter(ulb_unique_pseudo_label).most_common(1)[0][0]
                                most_common_number = Counter(ulb_unique_pseudo_label).most_common(1)[0][1]
                                if most_common_number > 0.8 * len(ulb_unique_pseudo_label):
                                    select_ulb_unique_pseudo_label.append(torch.tensor([most_common_label]))
                                else:
                                    select_ulb_unique_pseudo_label.append(torch.tensor([-1]))
                            else:
                                select_ulb_unique_pseudo_label.append(torch.tensor([-1]))

                            # process the pseudo-label distribution
                            select_ulb_unique_pseudo_label_distribution.append(ulb_unique_pseudo_label_distribution.unsqueeze(0))

                        select_ulb_unique_label = torch.cat(select_ulb_unique_label)
                        select_ulb_unique_pseudo_label = torch.cat(select_ulb_unique_pseudo_label)
                        select_ulb_unique_pseudo_label_distribution = torch.cat(select_ulb_unique_pseudo_label_distribution)

                        self.select_ulb_idx = torch.masked_select(select_ulb_unique_idx.cpu(), select_ulb_unique_pseudo_label != -1)
                        self.select_ulb_label = torch.masked_select(select_ulb_unique_label, select_ulb_unique_pseudo_label != -1)
                        self.select_ulb_pseudo_label = torch.masked_select(select_ulb_unique_pseudo_label, select_ulb_unique_pseudo_label != -1)
                        self.select_ulb_pseudo_label_distribution = select_ulb_unique_pseudo_label_distribution[select_ulb_unique_pseudo_label != -1]

                        self.select_ulb_dist = torch.zeros(self.num_classes).cuda(self.args.gpu)
                        for item in self.select_ulb_pseudo_label:
                            self.select_ulb_dist[int(item)] += 1

                        self.lb_select_ulb_dist = self.lb_dist + self.select_ulb_dist

                        self.print_fn('select_ulb_dist:\n' + np.array_str(np.array(self.select_ulb_dist.cpu())))
                        self.print_fn('lb_select_ulb_dist:\n' + np.array_str(np.array(self.lb_select_ulb_dist.cpu())))

                        # update the current labeled and pseudo labeled data
                        self.current_idx = np.concatenate((self.lb_idx, self.select_ulb_idx), axis=0)
                        self.current_x = self.data[self.current_idx]
                        self.current_y = np.concatenate((self.targets[self.lb_idx], self.select_ulb_pseudo_label), axis=0)
                        self.current_noise_y = np.concatenate((self.noised_targets[self.lb_idx], self.select_ulb_pseudo_label), axis=0)
                        current_one_hot_y = np.full((len(self.targets[self.lb_idx]), self.num_classes), self.args.smoothing / (self.num_classes - 1))
                        current_one_hot_y[np.arange(len(self.targets[self.lb_idx])), self.targets[self.lb_idx]] = 1.0 - self.args.smoothing
                        current_one_hot_noise_y = np.full((len(self.noised_targets[self.lb_idx]), self.num_classes), self.args.smoothing / (self.num_classes - 1))
                        current_one_hot_noise_y[np.arange(len(self.noised_targets[self.lb_idx])), self.noised_targets[self.lb_idx]] = 1.0 - self.args.smoothing
                        self.current_one_hot_y = np.concatenate((current_one_hot_y, self.select_ulb_pseudo_label_distribution), axis=0)
                        self.current_one_hot_noise_y = np.concatenate((current_one_hot_noise_y, self.select_ulb_pseudo_label_distribution), axis=0)

                        # reset select ulb idx and its pseudo label
                        self.select_ulb_idx = None
                        self.select_ulb_label = None
                        self.select_ulb_pseudo_label = None
                        self.select_ulb_pseudo_label_distribution = None

                if (self.epoch - self.gen_start_epoch - self.memory_step) % self.update_step == 0:
                    self.print_fn(str(self.epoch) + ': Update the labeled data.')
                    self._update_adaptive_loader()

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(self.adaptive_lb_dest_loader,
                                         self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def train_step(self, idx_lb, x_lb_w, x_lb_s, y_lb, y_lb_noised, idx_ulb, x_ulb_w, x_ulb_s, y_ulb):
        num_lb = y_lb.shape[0]
        if self.args.noise_ratio > 0:
            lb = y_lb_noised
        else:
            lb = y_lb

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb_w, x_lb_s, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb_w, logits_x_lb_s = outputs['logits'][:2 * num_lb].chunk(2)
                aux_logits_x_lb_w, aux_logits_x_lb_s = outputs['aux_logits'][:2 * num_lb].chunk(2)                
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][2 * num_lb:].chunk(2)
                aux_logits_x_ulb_w, aux_logits_x_ulb_s = outputs['aux_logits'][2 * num_lb:].chunk(2)                
                feats_x_lb_w, feats_x_lb_s = outputs['feat'][:2 * num_lb].chunk(2)
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][2 * num_lb:].chunk(2)
            else:
                outs_x_lb_w = self.model(x_lb_w)
                logits_x_lb_w = outs_x_lb_w['logits']
                aux_logits_x_lb_w = outs_x_lb_w['aux_logits']
                feats_x_lb_w = outs_x_lb_w['feat']
                outs_x_lb_s = self.model(x_lb_s)
                logits_x_lb_s = outs_x_lb_s['logits']
                aux_logits_x_lb_s = outs_x_lb_s['aux_logits']
                feats_x_lb_s = outs_x_lb_s['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                aux_logits_x_ulb_s = outs_x_ulb_s['aux_logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    aux_logits_x_ulb_w = outs_x_ulb_w['aux_logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb_w': feats_x_lb_w, 'x_lb_s': feats_x_lb_s, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

            logit_adjustment = torch.log(self.lb_select_ulb_dist / torch.sum(self.lb_select_ulb_dist))
            if self.epoch < self.gen_start_epoch:
                lb_smooth = torch.zeros(num_lb, self.num_classes).cuda(self.args.gpu)
                lb_smooth.fill_(self.args.smoothing / (self.num_classes - 1))
                lb_smooth.scatter_(1, lb.unsqueeze(1), 1.0 - self.args.smoothing)
                lb_target = lb_smooth

                sup_loss = self.ce_loss(logits_x_lb_w + logit_adjustment, lb_target, reduction='mean')       

                aux_pseudo_label_w = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=self.compute_prob(aux_logits_x_ulb_w.detach()), use_hard_label=self.use_hard_label, T=self.T, softmax=False)
                aux_pseudo_label_w_smooth = torch.zeros(self.args.uratio * num_lb, self.num_classes).cuda(self.args.gpu)
                aux_pseudo_label_w_smooth.fill_(self.args.smoothing / (self.num_classes - 1))
                aux_pseudo_label_w_smooth.scatter_(1, aux_pseudo_label_w.unsqueeze(1), 1.0 - self.args.smoothing)

                aux_loss = self.ce_loss(aux_logits_x_ulb_s, aux_pseudo_label_w_smooth, reduction='mean') + self.ce_loss(aux_logits_x_lb_w + logit_adjustment, lb_target, reduction='mean')

                mask = torch.tensor([False]).cuda(self.args.gpu)
            
            else:
                lb_target = y_lb
                sup_loss = self.ce_loss(logits_x_lb_w + logit_adjustment, lb_target, reduction='mean')
                aux_pseudo_label_w = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=self.compute_prob(aux_logits_x_ulb_w.detach()), use_hard_label=self.use_hard_label, T=self.T, softmax=False)
                aux_loss = self.ce_loss(aux_logits_x_ulb_s, aux_pseudo_label_w, reduction='mean') + self.ce_loss(aux_logits_x_lb_w + torch.log(self.lb_select_ulb_dist / torch.sum(self.lb_select_ulb_dist)), lb.argmax(dim=1), reduction='mean')

                if self.epoch >= self.ulb_start_epoch:
                    before_refined_select_ulb_dist = torch.where(self.select_ulb_dist <= min(self.lb_dist), 0, self.select_ulb_dist)

                    sorted_select_ulb_dist, _ = torch.sort(torch.unique(before_refined_select_ulb_dist))
                    if len(sorted_select_ulb_dist) == 1:
                        refined_select_ulb_dist = torch.ones_like(before_refined_select_ulb_dist)
                    else:
                        refined_select_ulb_dist = torch.where(before_refined_select_ulb_dist == 0, sorted_select_ulb_dist[1], before_refined_select_ulb_dist)
                    
                    probs_x_ulb_w = self.compute_prob((logits_x_ulb_w + torch.log(refined_select_ulb_dist / torch.sum(refined_select_ulb_dist))).detach())
                    probs_x_ulb_s = self.compute_prob((logits_x_ulb_s + torch.log(refined_select_ulb_dist / torch.sum(refined_select_ulb_dist))).detach())

                    pseudo_label_w = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=probs_x_ulb_w, use_hard_label=self.use_hard_label, T=self.T, softmax=False)
                    pseudo_label_s = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=probs_x_ulb_s, use_hard_label=self.use_hard_label, T=self.T, softmax=False)
                    
                    mask_w_s = pseudo_label_w == pseudo_label_s

                    mask_w = probs_x_ulb_w.amax(dim=-1).ge(self.p_cutoff * (1.0 - self.args.smoothing))
                    mask_s = probs_x_ulb_s.amax(dim=-1).ge(self.p_cutoff * (1.0 - self.args.smoothing))

                    mask = torch.logical_and(torch.logical_and(mask_w, mask_s), mask_w_s)

                    # update select_ulb_idx and its pseudo_label
                    if self.select_ulb_idx is not None and self.select_ulb_pseudo_label is not None and self.select_ulb_label is not None:
                        self.select_ulb_idx = torch.cat([self.select_ulb_idx, idx_ulb[mask]], dim=0)
                        self.select_ulb_label = torch.cat([self.select_ulb_label, y_ulb[mask]], dim=0)
                        self.select_ulb_pseudo_label = torch.cat([self.select_ulb_pseudo_label, pseudo_label_w[mask]], dim=0)
                    else:
                        self.select_ulb_idx = idx_ulb[mask]
                        self.select_ulb_label = y_ulb[mask]
                        self.select_ulb_pseudo_label = pseudo_label_w[mask]
                else:
                    mask = torch.tensor([False]).cuda(self.args.gpu)

            total_loss = sup_loss + aux_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         aux_loss=aux_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--alpha', float, 1.0),
            SSL_Argument('--smoothing', float, 0.1),

            SSL_Argument('--gen_start_epoch', int, 30, 'Epoch to start 1st gen (Moderate Tail)'),
            SSL_Argument('--gen_mid_epoch', int, 45, 'Epoch to start 2nd gen (Extreme Tail)'),
            SSL_Argument('--ulb_start_epoch', int, 60, 'Epoch to start using unlabeled data'),
            
            SSL_Argument('--head_threshold', int, 100, 'Min samples for a class to be "Head"'),
            SSL_Argument('--mid_threshold', int, 20, 'Min samples for a class to be "Moderate Tail"'),
            
            SSL_Argument('--gen_target_count', int, 100, 'Target sample count for generated classes'),
            SSL_Argument('--gen_batch_size', int, 32, 'Batch size for diffusion generator'),
            SSL_Argument('--gen_steps', int, 50, 'Number of inference steps for diffusion'),
            SSL_Argument('--guidance_scale', float, 1.5, 'Strength of classifier guidance'),
            SSL_Argument('--model_path', str, './pretrained_models/cifar10-ddpm', 'Local path to diffusion model'),
        ]
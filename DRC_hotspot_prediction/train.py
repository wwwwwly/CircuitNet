# Copyright 2022 CircuitNet. All rights reserved.

import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm

from datasets.build_dataset import build_dataset
from utils.losses import build_loss
from models.build_model import build_model
from utils.configs import TrainParser
from math import cos, pi
import sys, os, subprocess
from pathlib import Path


def checkpoint(model, epoch, loss, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_out_path = f"./{save_path}/model_iters_{epoch}_{loss}.pth"
    torch.save({"state_dict": model.state_dict()}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


class CosineRestartLr(object):
    def __init__(
        self, base_lr, periods, restart_weights=[1], min_lr=None, min_lr_ratio=None
    ):
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        super().__init__()

        self.cumulative_periods = [
            sum(self.periods[0 : i + 1]) for i in range(0, len(self.periods))
        ]

        self.base_lr = base_lr

    def annealing_cos(
        self, start: float, end: float, factor: float, weight: float = 1.0
    ) -> float:
        cos_out = cos(pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def get_position_from_periods(self, iteration: int, cumulative_periods):
        for i, period in enumerate(cumulative_periods):
            if iteration < period:
                return i
        raise ValueError(
            f"Current iteration {iteration} exceeds "
            f"cumulative_periods {cumulative_periods}"
        )

    def get_lr(self, iter_num, base_lr: float):
        target_lr = self.min_lr  # type:ignore

        idx = self.get_position_from_periods(iter_num, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((iter_num - nearest_restart) / current_periods, 1)
        return self.annealing_cos(base_lr, target_lr, alpha, current_weight)

    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim in optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group["lr"] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group["lr"] = lr

    def get_regular_lr(self, iter_num):
        return [self.get_lr(iter_num, _base_lr) for _base_lr in self.base_lr]  # iters

    def set_init_lr(self, optimizer):
        for group in optimizer.param_groups:  # type: ignore
            group.setdefault("initial_lr", group["lr"])
            self.base_lr = [
                group["initial_lr"] for group in optimizer.param_groups  # type: ignore
            ]


def train(arg_dict):
    print("===> Loading datasets")
    # Initialize dataset
    train_set = build_dataset(arg_dict)

    print("===> Building model")
    # Initialize model parameters
    model = build_model(arg_dict)
    if not arg_dict["cpu"]:
        model = model.cuda()

    # Build loss
    loss = build_loss(arg_dict)

    # Build Optimzer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=arg_dict["lr"],
        betas=(0.9, 0.999),
        weight_decay=arg_dict["weight_decay"],
    )

    # Build lr scheduler
    cosine_lr = CosineRestartLr(arg_dict["lr"], [arg_dict["max_iters"]], [1], 1e-7)
    cosine_lr.set_init_lr(optimizer)

    # build validation set
    arg_dict["ann_file"] = arg_dict["ann_file_val"]
    arg_dict["test_mode"] = True
    val_set = build_dataset(arg_dict)
    
    train_epoch_loss = 0
    val_epoch_loss = 0
    iter_num = 0
    print_freq = 100
    save_freq = 10000

    while iter_num < arg_dict["max_iters"]:
        #-----------------Training-----------------#
        model.train()
        with tqdm(total=print_freq) as bar:
            for feature, label, _ in train_set:
                if arg_dict["cpu"]:
                    input, target = feature, label
                else:
                    input, target = feature.cuda(), label.cuda()

                regular_lr = cosine_lr.get_regular_lr(iter_num)
                cosine_lr._set_lr(optimizer, regular_lr)

                prediction = model(input)

                optimizer.zero_grad()

                # # 标签改为2值图像
                # target[target >= arg_dict["threshold"]] = 1
                # target[target < arg_dict["threshold"]] = 0

                pixel_loss = loss(prediction, target)
                train_epoch_loss += pixel_loss.item()
                pixel_loss.backward()

                optimizer.step()

                iter_num += 1

                bar.update(1)

                if iter_num % print_freq == 0:
                    break
        
        #-----------------Validation-----------------#
        model.eval()
        with tqdm(total=print_freq) as bar:
            with torch.no_grad():
                tmp_iter_num=iter_num
                for feature, label, _ in val_set:
                    if arg_dict["cpu"]:
                        input, target = feature, label
                    else:
                        input, target = feature.cuda(), label.cuda()

                    prediction = model(input)
                    pixel_loss = loss(prediction, target)
                    val_epoch_loss += pixel_loss.item()

                    tmp_iter_num += 1

                    bar.update(1)

                    if tmp_iter_num % print_freq == 0:
                        break

        log_message = "===> Iters[{}]({}/{}): Train Loss: {:.4f}\t\tValidation Loss: {:4f}".format(
            iter_num, iter_num, arg_dict["max_iters"], train_epoch_loss / print_freq, val_epoch_loss / print_freq
        )
        print(log_message)

        log_file_path = (
            Path(arg_dict["save_path"])
            / Path(arg_dict["task_description"])
            / f"{arg_dict['task_description']}_training_log.txt"
        )
        log_file_path.touch(exist_ok=True)
        with log_file_path.open("a") as log_file:
            log_file.write(log_message + "\n")

        if iter_num % save_freq == 0:
            checkpoint(model,iter_num, val_epoch_loss / print_freq,
                       arg_dict["save_path"] + f"/{arg_dict['task_description']}")
            
        train_epoch_loss = 0
        val_epoch_loss = 0


if __name__ == "__main__":
    argp = TrainParser()
    arg = argp.parse_args()
    arg_dict = vars(arg)

    if arg.arg_file is not None:
        with open(arg.arg_file, "rt") as f:
            arg_dict.update(json.load(f))

    save_dir = os.path.join(arg_dict["save_path"], arg_dict["task_description"])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(
        os.path.join(save_dir, "arg.json"),
        "wt",
    ) as f:
        json.dump(arg_dict, f, indent=4)

    arg_dict["ann_file"] = arg_dict["ann_file_train"]
    arg_dict["test_mode"] = False

    train(arg_dict)

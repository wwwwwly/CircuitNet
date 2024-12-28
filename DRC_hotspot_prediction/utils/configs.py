# Copyright 2022 CircuitNet. All rights reserved.

import argparse
import os
import sys

sys.path.append(os.getcwd())


class Parser(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--task", default="drc_routenet")
        self.parser.add_argument("--save_path", default="./drc_routenet")
        self.parser.add_argument("--max_iters", default=200000)
        self.parser.add_argument(
            "--plot_roc", default="true", action="store_true"
        )  # 如果提供plot_roc参数，值会设置为true
        self.parser.add_argument("--arg_file", default=None)
        self.parser.add_argument("--cpu", action="store_true")
        self.get_remainder()

    def get_remainder(self):
        if self.parser.parse_args().task == "drc_routenet":
            self.parser.add_argument("--dataroot", default="../training_set/DRC")
            self.parser.add_argument(
                "--ann_file_train", default="./files/train_N28.csv"
            )
            self.parser.add_argument("--ann_file_test", default="./files/test_N28.csv")
            self.parser.add_argument("--dataset_type", default="DRCDataset")
            self.parser.add_argument("--batch_size", default=8)
            self.parser.add_argument("--aug_pipeline", default=["Flip"])

            self.parser.add_argument("--model_type", default="RouteNet")
            self.parser.add_argument("--in_channels", default=9)
            self.parser.add_argument("--out_channels", default=1)
            self.parser.add_argument("--lr", default=2e-4)
            self.parser.add_argument("--weight_decay", default=1e-4)
            self.parser.add_argument("--loss_type", default="MSELoss")
            self.parser.add_argument("--eval-metric", default=["NRMS", "SSIM"])
            self.parser.add_argument("--threshold", default=0.1)

        else:
            raise ValueError

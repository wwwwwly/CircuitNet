# Copyright 2022 CircuitNet. All rights reserved.

from __future__ import print_function

import os
import os.path as osp
import json
import numpy as np

from tqdm import tqdm

from datasets.build_dataset import build_dataset
from utils.metrics import build_metric, build_roc_prc_metric
from models.build_model import build_model
from utils.configs import TestParser
from pathlib import Path
from datetime import datetime


def test(arg_dict):
    print("===> Loading datasets")
    # Initialize dataset
    dataset = build_dataset(arg_dict)

    print("===> Building model")
    # Initialize model parameters
    model = build_model(arg_dict)
    if not arg_dict["cpu"]:
        model = model.cuda()

    model.eval()
    # Build metrics
    metrics = {k: build_metric(k) for k in arg_dict["eval_metric"]}
    avg_metrics = {k: 0 for k in arg_dict["eval_metric"]}

    count = 0
    with tqdm(total=len(dataset)) as bar:
        for feature, label, label_path in dataset:
            if arg_dict["cpu"]:
                input, target = feature, label
            else:
                input, target = feature.cuda(), label.cuda()

            prediction = model(input)
            for metric, metric_func in metrics.items():
                if not metric_func(target.cpu(), prediction.squeeze(1).cpu()) == 1:
                    avg_metrics[metric] += metric_func(
                        target.cpu(), prediction.squeeze(1).cpu()
                    )

            if arg_dict["plot_roc"]:
                save_path = osp.join(
                    arg_dict["save_path"],
                    arg_dict["task_description"],
                    # "test_result-" + arg_dict["pretrained"].split("/")[-1],
                    "test_result",
                )
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                file_name = osp.splitext(osp.basename(label_path[0]))[
                    0
                ]  # 获取npy的文件名(不包括扩展名) label_path[0]是因为dataloader将原来的路径字符串被打包成了元组
                save_path = osp.join(save_path, f"{file_name}.npy")
                output_final = prediction.float().detach().cpu().numpy()
                np.save(save_path, output_final)
                count += 1

            bar.update(1)

    log_file_path = (
        Path(arg_dict["save_path"])
        / Path(arg_dict["task_description"])
        / r"images/metrics_log.txt"
    )
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    log_file_path.touch(exist_ok=True)
    with log_file_path.open("a") as log_file:
        for metric, avg_metric in avg_metrics.items():
            metric_msg = "===> Avg. {}: {:.4f}".format(
                metric, avg_metric / len(dataset)
            )
            print(metric_msg)
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"{formatted_time} - {metric_msg}\n")

    # eval roc&prc
    # if arg_dict["plot_roc"]:
    #     roc_metric, _ = build_roc_prc_metric(**arg_dict)
    #     print("\n===> AUC of ROC. {:.4f}".format(roc_metric))


if __name__ == "__main__":
    # 测试前需要设置pretrained的值 如果模型不同记得修改模型 
    argp = TestParser()
    arg = argp.parse_args()

    arg_dict = vars(arg)
    arg_dict["task_description"] = arg_dict["pretrained"].split("/")[2]

    if arg.arg_file is not None:
        with open(arg.arg_file, "rt") as f:
            arg_dict.update(json.load(f))

    arg_dict["ann_file"] = arg_dict["ann_file_test"]
    arg_dict["test_mode"] = True

    test(arg_dict)

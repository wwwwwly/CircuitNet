# Copyright 2022 CircuitNet. All rights reserved.

import models
import torch


def build_model(opt):  # 传入arg_dict 即将arg parser转化后的字典形式
    model = models.__dict__[opt.pop("model_type")](**opt)
    model.init_weights(**opt)
    if opt["test_mode"]:
        model.eval()

    return model

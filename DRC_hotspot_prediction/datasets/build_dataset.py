# Copyright 2022 CircuitNet. All rights reserved.

from torch.utils.data import DataLoader
import sys
import os
# script_dir = os.path.dirname(os.path.abspath(__file__))  # a.py 的目录
# project_dir = os.path.dirname(script_dir)  # datasets 的父目录，即项目根目录
# sys.path.append(project_dir)
import datasets

import time

from .augmentation import Flip, Rotation


class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            time.sleep(2)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)
        return data

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        return self


def build_dataset(opt):
    aug_methods = {"Flip": Flip(), "Rotation": Rotation(**opt)}
    pipeline = (
        [aug_methods[i] for i in opt["aug_pipeline"]]
        if "aug_pipeline" in opt and not opt["test_mode"]
        else None
    )
    dataset = datasets.__dict__[opt["dataset_type"]](**opt, pipeline=pipeline)
    if opt["test_mode"]:
        return DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)
    else:
        return IterLoader(
            DataLoader(
                dataset=dataset,
                num_workers=6,# todo: 扩展性
                batch_size=opt.pop("batch_size"),
                shuffle=True,
                drop_last=True,
                pin_memory=True,
            )
        )

if __name__=="__main__":
    # test IterLoader 
    It=IterLoader([1,2,3])
    i=0
    while i<5:
        for data in It:
            print(data)
            break
        i+=1

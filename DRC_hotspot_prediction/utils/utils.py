import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import os
import numpy as np
from configs import TestParser
from torch.utils.data import random_split
from torch import Generator
import csv

def save_img(list_x, list_y, img_name, x_axis="x", y_axis="y", label="data", title="image", save_path=None):
    img_dir = os.path.join(os.path.dirname(__file__), "..", "images")

    if save_path == None:
        save_path = os.path.join(img_dir, img_name)
    else:
        save_path = os.path.join(save_path, "images", img_name)

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    plt.figure(figsize=(6, 4))
    plt.plot(list_x, list_y, marker="o", label=label)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=300)
    plt.close()


def visualize_npy(save_path, npy_path_pred, npy_path_label=None):
    pred = np.load(npy_path_pred)
    pred = np.squeeze(pred)

    label = None
    if npy_path_label is not None:
        label = np.load(npy_path_label)
        label = np.squeeze(label)

    colors = [
        "#0c099c",
        "#29fdf6",
        "#2dfd29",
        "#ffffff",
        "#f8ff38",
        "#ff5d38",
        "#ff0000",
    ]  # 蓝 浅蓝 绿 白 黄 浅红 红
    cmap = LinearSegmentedColormap.from_list("blue_red", colors, N=256)
    norm_pred = Normalize(vmin=np.min(pred), vmax=np.max(pred))

    if label is not None:
        norm_label = Normalize(vmin=np.min(label), vmax=np.max(label))

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(pred, cmap=cmap, norm=norm_pred)
        axes[0].set_title("Prediction")
        axes[0].axis("off")

        axes[1].imshow(label, cmap=cmap, norm=norm_label)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(pred, cmap=cmap, norm=norm_pred)
        ax.set_title("feature")
        ax.axis("off")

    # 保存图像
    plt.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.close()


def visualize_all(arg_dict):
    label_dir = os.path.join("..", arg_dict["dataroot"], "label")
    pred_dir = os.path.join(
        "..", arg_dict["save_path"], arg_dict["task_description"], "test_result"
    )
    save_dir = os.path.join(
        "..", arg_dict["save_path"], arg_dict["task_description"], "visualization"
    )
    os.makedirs(save_dir, exist_ok=True)

    preds = os.listdir(pred_dir)

    for i, pred in enumerate(preds):
        save_path = os.path.join(save_dir, os.path.splitext(pred)[0] + ".png")
        visualize_npy(
            save_path, os.path.join(pred_dir, pred), os.path.join(label_dir, pred)
        )

    print("done")


def dataset_split(data_dir, file_dir=None, length=10242, rate=[0.7, 0.2, 0.1]):
    data_dir=os.path.join("..",data_dir)
    save_dir=file_dir if file_dir else os.path.join(os.path.dirname(__file__),"..","files")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    train_path=os.path.join(save_dir,"train_N28.csv")
    val_path=os.path.join(save_dir,"val_N28.csv")
    test_path=os.path.join(save_dir,"test_N28.csv")
    
    file_list=os.listdir(os.path.join(data_dir,"feature"))
    rate=(np.array(rate) * length).astype(int)
    if rate.sum() != length:
        rate[0]+=length-rate.sum()
    train_set,val_set,test_set=random_split(file_list,rate,generator=Generator().manual_seed(0))

    with open(train_path, "w", newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        for file in train_set:
           writer.writerow(["feature/"+file,"label/"+file])
    
    with open(val_path, "w", newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        for file in val_set:
           writer.writerow(["feature/"+file,"label/"+file])
    
    with open(test_path, "w", newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        for file in test_set:
           writer.writerow(["feature/"+file,"label/"+file])

if __name__ == "__main__":
    argp = TestParser()
    arg = argp.parse_args()
    arg_dict = vars(arg)

    # visualize_all(arg_dict)
    dataset_split(arg_dict["dataroot"])

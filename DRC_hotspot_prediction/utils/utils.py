import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import os
import numpy as np
from configs import TestParser

img_dir = os.path.join(os.path.dirname(__file__), "..", "images")


def save_img(
    list_x,
    list_y,
    img_name,
    x_axis="x",
    y_axis="y",
    label="data",
    title="image",
    save_path=None,
):
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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap


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


def visualize_all(task_description):
    argp = TestParser()
    arg = argp.parse_args()
    arg_dict = vars(arg)
    arg_dict["task_description"] = task_description

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


if __name__ == "__main__":
    visualize_all("original")

import numpy as np
import matplotlib.pyplot as plt
import os

training_set = os.path.join(os.path.dirname(__file__), "../training_set/DRC")
test_result = os.path.join(os.path.dirname(__file__), "drc_routenet/test_result")


def normalize(path, file):
    try:
        data = np.load(os.path.join(path, file))
        print(f"Data shape: {data.shape}, dtype: {data.dtype}")

        # 将数据归一化到 0-255
        if np.issubdtype(data.dtype, np.floating):
            data = (
                255
                * (data - data.min(axis=(0, 1), keepdims=True))
                / (
                    data.max(axis=(0, 1), keepdims=True)
                    - data.min(axis=(0, 1), keepdims=True)
                )
            ).astype(np.uint8)
        return data

    except Exception as e:
        print(f"加载文件 {file} 时出错: {e}")
        return None


if __name__ == "__main__":
    feature_list = [
        "macro_region",
        "cell_density",
        "RUDY_long",
        "RUDY_short",
        "RUDY_pin_long",
        "congestion_eGR_horizontal_overflow",
        "congestion_eGR_vertical_overflow",
        "congestion_GR_horizontal_overflow",
        "congestion_GR_vertical_overflow",
    ]

    file = "9284-zero-riscy-b-1-c2-u0.85-m1-p5-f1.npy"
    data = normalize(os.path.join(training_set, "feature"), file)

    fig, axes = plt.subplots(2, 5, figsize=(15, 5))
    for i, feature in enumerate(feature_list):

        axes[i // 5, i % 5].imshow(data[:, :, i], cmap="gray")
        axes[i // 5, i % 5].axis("off")
        axes[i // 5, i % 5].set_title(feature)
    axes[1, 4].axis("off")

    plt.tight_layout()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    data = np.load(os.path.join(training_set, "label", file))
    # data = (data > 0.01).astype(int)
    print(data.shape)
    data = (
        255
        * (data - data.min(axis=(0, 1), keepdims=True))
        / (data.max(axis=(0, 1), keepdims=True) - data.min(axis=(0, 1), keepdims=True))
    ).astype(np.uint8)

    axes[0].imshow(data[:, :, 0], cmap="gray", aspect="auto")
    axes[0].axis("off")
    axes[0].set_title("DRC label")

    data = np.load(os.path.join(test_result, file))
    # data = (data >= 0.01).astype(int)
    print(data.shape)
    data = (
        255
        * (data - data.min(axis=(2, 3), keepdims=True))
        / (data.max(axis=(2, 3), keepdims=True) - data.min(axis=(2, 3), keepdims=True))
    ).astype(np.uint8)

    axes[1].imshow(data[0, 0, :, :], cmap="gray", aspect="auto")
    axes[1].axis("off")
    axes[1].set_title("DRC prediction")

    plt.show()

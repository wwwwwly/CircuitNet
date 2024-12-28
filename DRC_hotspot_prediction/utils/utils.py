import matplotlib.pyplot as plt
import os

img_dir = os.path.join(os.path.dirname(__file__), "..", "images")


def save_img(
    list_x, list_y, img_name, x_axis="x", y_axis="y", label="data", title="image"
):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    save_path = os.path.join(img_dir, img_name)
    # 创建图像
    plt.figure(figsize=(6, 4))
    plt.plot(list_x, list_y, marker="o", label=label)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig(save_path, dpi=300)
    plt.close()  # 关闭图形以释放资源

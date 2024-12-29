# Copyright 2022 CircuitNet. All rights reserved.

from functools import wraps
from inspect import getfullargspec

import os
import os.path as osp
import cv2
import numpy as np
import torch
import multiprocessing as mul
import uuid
import psutil
import time
import csv
import shutil

# from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix
from scipy.interpolate import make_interp_spline
from functools import partial
from mmcv import scandir

from scipy.stats import wasserstein_distance
from skimage.metrics import normalized_root_mse
import math
import utils.metrics as metrics
import utils.utils as utils

__all__ = ["psnr", "ssim", "nrms", "emd"]


def confusion_matrix(label, pred):
    label, pred = label.float(), pred.float()
    tp = torch.logical_and(label, pred).count_nonzero().item()
    tn = torch.logical_and(1 - label, 1 - pred).count_nonzero().item()
    fp = torch.logical_and(1 - label, pred).count_nonzero().item()
    fn = torch.logical_and(label, 1 - pred).count_nonzero().item()

    return tn, fp, fn, tp


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == "":
        return
    dir_name = osp.expanduser(dir_name)  # 处理 ~ 符号，转化为用户当前目录
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def input_converter(apply_to=None):
    def input_converter_wrapper(old_func):
        @wraps(old_func)
        def new_func(*args, **kwargs):
            args_info = getfullargspec(old_func)
            args_to_cast = args_info.args if apply_to is None else apply_to
            new_args = []
            if args:
                arg_names = args_info.args[: len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(tensor2img(args[i]))
                    else:
                        new_args.append(args[i])

            return old_func(*new_args)

        return new_func

    return input_converter_wrapper


@input_converter(apply_to=("img1", "img2"))
def psnr(img1, img2, crop_border=0):  # 峰值信噪比（PSNR，Peak Signal-to-Noise Ratio）
    assert (
        img1.shape == img2.shape
    ), f"Image shapes are different: {img1.shape}, {img2.shape}."

    img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    mse_value = np.mean((img1 - img2) ** 2)
    if mse_value == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse_value))


def _ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


@input_converter(
    apply_to=("img1", "img2")
)  # 结构相似性指数（SSIM,Structural Similarity Index Measurement）
def ssim(img1, img2, crop_border=0):
    assert (
        img1.shape == img2.shape
    ), f"Image shapes are different: {img1.shape}, {img2.shape}."
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


@input_converter(apply_to=("img1", "img2"))
def nrms(img1, img2, crop_border=0):  # NRMS (Normalized Root Mean Squared Error)
    assert (
        img1.shape == img2.shape
    ), f"Image shapes are different: {img1.shape}, {img2.shape}."

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    nrmse_value = normalized_root_mse(
        img1.flatten(), img2.flatten(), normalization="min-max"
    )
    if math.isinf(nrmse_value):
        return 0.05
    return nrmse_value


def get_histogram(img):
    h, w = img.shape
    hist = [0.0] * 256
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
    return np.array(hist) / float(h * w)


def normalize_exposure(img):
    img = img.astype(int)
    hist = get_histogram(img)
    cdf = np.array([sum(hist[: i + 1]) for i in range(len(hist))])
    sk = np.uint8(255 * cdf)
    height, width = img.shape
    normalized = np.zeros_like(img)
    for i in range(0, height):
        for j in range(0, width):
            normalized[i, j] = sk[img[i, j]]
    return normalized.astype(int)


@input_converter(apply_to=("img1", "img2"))
def emd(img1, img2, crop_border=0):  # 动土距离 EMD (Earth Mover's Distance)
    assert (
        img1.shape == img2.shape
    ), f"Image shapes are different: {img1.shape}, {img2.shape}."

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    # change here
    img1 = normalize_exposure(np.squeeze(img1, axis=2))
    img2 = normalize_exposure(np.squeeze(img2, axis=2))
    hist_1 = get_histogram(img1)
    hist_2 = get_histogram(img2)

    emd_value = wasserstein_distance(hist_1, hist_2)
    return emd_value


def tpr(tp, fn):  # 召回率/查全率
    return tp / (tp + fn)


def fpr(fp, tn):  #
    return fp / (fp + tn)


def precision(tp, fp):  # 精确率/查准率
    return tp / (tp + fp)


def accuracy(tp, tn, fp, fn):  # 准确率
    return (tp + tn) / (tp + fp + tn + fn)


def calculate_all(csv_path):
    tpr_sum_List = []
    fpr_sum_List = []
    precision_sum_List = []
    threshold_remain_list = []
    num = 0
    tpr_sum = 0
    fpr_sum = 0
    precision_sum = 0

    with open(os.path.join(csv_path), "r") as csv_file:

        first_flag = False
        for line in csv_file:
            threshold, idx, tn, fp, fn, tp = line.strip().split(",")
            if threshold not in threshold_remain_list:
                if first_flag:
                    if num != 0:
                        tpr_sum_List.append(tpr_sum / num)
                        fpr_sum_List.append(fpr_sum / num)
                        precision_sum_List.append(precision_sum / num)
                threshold_remain_list.append(threshold)
                tpr_sum = 0
                fpr_sum = 0
                precision_sum = 0
                num = 0
                first_flag = True

            if int(fp) == 0 and int(tn) == 0:  # 阴性样本为0
                continue
            elif int(tp) == 0 and int(fn) == 0:  # 阳性样本为0
                continue
            elif int(tp) == 0 and int(fp) == 0:  # 预测值全0
                continue
            # todo: 是否应该直接放弃上述无效的pred   /num of effective preds or /num of all preds
            # todo: 拆分为3个if & 3个num分别统计求均值
            # todo: 或者将同一阈值下的所有preds的tp,tn,fp,fn全部各自求和之后再计算相应指标
            else:
                tpr_sum += tpr(int(tp), int(fn))
                fpr_sum += fpr(int(fp), int(tn))
                precision_sum += precision(int(tp), int(fp))
                num += 1
        if num != 0:
            tpr_sum_List.append(tpr_sum / num)
            fpr_sum_List.append(fpr_sum / num)
            precision_sum_List.append(precision_sum / num)

    return tpr_sum_List, fpr_sum_List, precision_sum_List


def calculated_score(
    threshold_idx=None,
    temp_path=None,
    label_path=None,
    save_path=None,
    threshold_label=None,
    preds=None,
):
    with open(
        os.path.join(temp_path, f"tpr_fpr_{threshold_idx}.csv"), "w", newline=""
    ) as file:
        f_csv = csv.writer(file, delimiter=",")
        for idx, pred in enumerate(preds):
            # numpy array -> tensor
            target_test = torch.tensor(
                np.load(os.path.join(label_path, pred)).reshape(-1)
            ).cuda()
            target_probabilities = torch.tensor(
                np.load(os.path.join(save_path, "test_result", pred)).reshape(-1)
            ).cuda()

            # 计算时是将label和输出转成2值图计算了
            target_test[target_test >= threshold_label] = 1
            target_test[target_test < threshold_label] = 0

            target_probabilities[target_probabilities >= threshold_idx] = 1
            target_probabilities[target_probabilities < threshold_idx] = 0

            # 标签和预测都全1 即全部为正例
            if target_probabilities.all() == 0 and target_test.all():
                tp = 256 * 256
                tn, fn, fp = 0, 0, 0
            # 标签和预测都全0 即全部为反例
            elif not (target_probabilities.any() or target_test.any()):
                tn = 256 * 256
                tp, fn, fp = 0, 0, 0
            # 计算混淆矩阵 并展平
            else:
                tn, fp, fn, tp = confusion_matrix(target_test, target_probabilities)

            f_csv.writerow(
                [str(threshold_idx)] + [str(i) for i in [idx, tn, fp, fn, tp]]
            )

    print(f"{threshold_idx}-done")


# 对于每个预测结果，0-1 200个阈值分别与label（只有一个阈值）计算混淆矩阵，并全部保存到1个csv文件中
def multi_process_score(out_name=None, threshold=0.0, label_path=None, save_path=None):
    uid = str(uuid.uuid4())
    suid = "".join(uid.split("-"))
    temp_path = f"./{suid}"

    # pool = mul.Pool(int(mul.cpu_count() * (1 - psutil.cpu_percent(None) / 100.0))) # mul.cpu_count()返回了逻辑核心数
    num_processes = int(4)
    with mul.Pool(num_processes) as pool:
        preds = scandir(
            os.path.join(save_path, "test_result"),  # todo: 文件名
            suffix="npy",
            recursive=True,
        )
        preds = [v for v in preds]

        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        threshold_list = np.linspace(0, 1, endpoint=False, num=200)

        calculated_score_parital = partial(
            calculated_score,
            temp_path=temp_path,
            label_path=label_path,
            save_path=save_path,
            threshold_label=threshold,
            preds=preds,
        )
        # 将 threshold_list 中的每个阈值（从 0 到 1）依次传递给 calculated_score_parital 函数并行执行
        rel = pool.map(calculated_score_parital, threshold_list)

    print(f"{suid}")

    for list_i in threshold_list:
        fr = open(os.path.join(temp_path, f"tpr_fpr_{list_i}.csv"), "r").read()
        with open(os.path.join(temp_path, f"{out_name}"), "a") as f:
            f.write(fr)
        f.close()

    # if not os.path.exists(os.path.join(os.getcwd(), 'out')):
    #     os.makedirs(os.path.join(os.getcwd(), 'out'))

    print("copying")
    source_path = os.path.join(temp_path, f"{out_name}")
    destination_path = os.path.join(os.getcwd(), save_path, f"{out_name}")
    shutil.copy(source_path, destination_path)

    print("remove temp files")
    if os.path.exists(temp_path) and os.path.isdir(temp_path):
        shutil.rmtree(temp_path)


def get_sorted_list(fpr_sum_List, tpr_sum_List):
    fpr_list = []
    tpr_list = []
    for i, j in zip(fpr_sum_List, tpr_sum_List):
        if i not in fpr_list:  # todo: 为什么去重？ 单纯为了绘图？
            fpr_list.append(i)
            tpr_list.append(j)

    fpr_list.reverse()
    tpr_list.reverse()
    fpr_list, tpr_list = zip(*sorted(zip(fpr_list, tpr_list)))  # 默认for升序
    return fpr_list, tpr_list


# 返回AUC-ROC和AUC-PR
def roc_prc(save_path):
    # 获得了200个阈值下的 平均tpr fpr precision
    tpr_sum_List, fpr_sum_List, precision_sum_List = calculate_all(
        os.path.join(save_path, "roc_prc.csv")
    )

    fpr_list, tpr_list = get_sorted_list(fpr_sum_List, tpr_sum_List)
    fpr_list = list(fpr_list)
    fpr_list.extend([1])

    tpr_list = list(tpr_list)
    tpr_list.extend([1])

    roc_numerator = 0
    for i in range(len(tpr_list) - 1):
        roc_numerator += (
            (tpr_list[i] + tpr_list[i + 1]) * (fpr_list[i + 1] - fpr_list[i]) / 2
        )

    utils.save_img(
        fpr_list,
        tpr_list,
        "roc.png",
        x_axis="FPR",
        y_axis="TPR",
        label=f"AUC-ROC={roc_numerator}",
        title="ROC",
        save_path=save_path,
    )

    tpr_list, p_list = get_sorted_list(tpr_sum_List, precision_sum_List)
    x_smooth = np.linspace(0, 1, 25)
    y_smooth = make_interp_spline(tpr_list, p_list, k=3)(x_smooth)

    prc_numerator = 0
    for i in range(len(y_smooth) - 1):
        prc_numerator += (
            (y_smooth[i] + y_smooth[i + 1]) * (x_smooth[i + 1] - x_smooth[i]) / 2
        )

    return roc_numerator, prc_numerator


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    if not (
        torch.is_tensor(tensor)
        or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))
    ):
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).squeeze(0)
        _tensor = _tensor.float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()

        if n_dim == 3:
            img_np = _tensor.numpy()
            img_np = np.transpose(img_np[:, :, :], (2, 0, 1))
            # img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
        elif n_dim == 2:
            img_np = _tensor.numpy()[..., None]
        else:
            raise ValueError(
                "Only support 4D, 3D or 2D tensor. "
                f"But received with dimension: {n_dim}"
            )
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    result = result[0] if len(result) == 1 else result
    return result


def build_metric(metric_name):
    return metrics.__dict__[metric_name.lower()]


def build_roc_prc_metric(
    threshold=None, dataroot=None, ann_file=None, save_path=None, **kwargs
):
    save_path = os.path.join(save_path, kwargs["task_description"])
    multi_process_score(
        out_name="roc_prc.csv",
        threshold=threshold,
        label_path=os.path.join(dataroot, "label"),  # ./training_set/DRC/label
        save_path=save_path,
    )

    return roc_prc(save_path)

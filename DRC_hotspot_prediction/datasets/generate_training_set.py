import os
import argparse
import numpy as np
import cv2
from scipy import ndimage
from multiprocessing import Process


def get_sub_path(path):
    sub_path = []
    if isinstance(path, list):
        for p in path:
            if os.path.isdir(p):
                for file in os.listdir(p):
                    sub_path.append(os.path.join(p, file))
            else:
                continue
    else:
        for file in os.listdir(path):
            sub_path.append(os.path.join(path, file))
    return sub_path


def resize(input):
    dimension = input.shape
    result = ndimage.zoom(input, (256 / dimension[0], 256 / dimension[1]), order=3)
    return result


def resize_cv2(input):
    output = cv2.resize(input, (256, 256), interpolation=cv2.INTER_AREA)
    return output


def std(input):
    if input.max() == 0:
        return input
    else:
        result = (input - input.min()) / (input.max() - input.min())
        return result


def save_npy(out_list, save_path, name):
    output = np.array(out_list)
    output = np.transpose(output, (1, 2, 0))
    np.save(os.path.join(save_path, name), output)


def divide_list(list, n):
    for i in range(0, len(list), n):
        yield list[i : i + n]


def pack_data(args, name_list, read_feature_list, read_label_list, save_path):
    os.makedirs(save_path, exist_ok=True)
    feature_save_path = os.path.join(args.save_path, args.task, "feature")

    os.makedirs(feature_save_path, exist_ok=True)
    label_save_path = os.path.join(args.save_path, args.task, "label")

    os.makedirs(label_save_path, exist_ok=True)

    for name in name_list:
        out_feature_list = []
        for feature_name in read_feature_list:
            name = os.path.basename(name)
            feature = np.load(os.path.join(args.data_path, feature_name, name))
            if args.task == "DRC":
                feature = std(resize(feature))
                out_feature_list.append(feature)
            else:
                raise ValueError("Task not implemented")

        save_npy(out_feature_list, feature_save_path, name)

        out_label_list = []
        for label_name in read_label_list:
            name = os.path.basename(name)
            label = np.load(os.path.join(args.data_path, label_name, name))

            if args.task == "DRC":
                label = np.clip(label, 0, 200)
                label = resize_cv2(label) / 200
                # label = label/200
                # label = np.clip(label, 0, 1)
                out_label_list.append(label)

            else:
                raise ValueError("Task not implemented")

        save_npy(out_label_list, label_save_path, name)


def parse_args():
    description = "you should add those parameter"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--task",
        default="DRC",
        type=str,
        help="select from congestion, DRC and IR_drop",
    )
    parser.add_argument(
        "--data_path",
        default="../../",
        type=str,
        help="path to the decompressed dataset",
    )
    parser.add_argument(
        "--save_path",
        default="../../training_set",
        type=str,
        help="path to save training set",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.task == "DRC":
        feature_list = [
            "routability_features/macro_region",
            "routability_features/cell_density",
            "routability_features/RUDY/RUDY_long",
            "routability_features/RUDY/RUDY_short",
            "routability_features/RUDY/RUDY_pin_long",
            "routability_features/congestion/congestion_early_global_routing/overflow_based/congestion_eGR_horizontal_overflow",
            "routability_features/congestion/congestion_early_global_routing/overflow_based/congestion_eGR_vertical_overflow",
            "routability_features/congestion/congestion_global_routing/overflow_based/congestion_GR_horizontal_overflow",
            "routability_features/congestion/congestion_global_routing/overflow_based/congestion_GR_vertical_overflow",
        ]
        label_list = ["routability_features/DRC/DRC_all"]

    else:
        raise ValueError("Please specify argument --task from DRC ")

    name_list = get_sub_path(os.path.join(args.data_path, label_list[0]))
    print("processing %s files" % len(name_list))
    save_path = os.path.join(args.save_path, args.task)

    nlist = divide_list(name_list, 1000)
    process = []
    for list in nlist:
        p = Process(
            target=pack_data, args=(args, list, feature_list, label_list, save_path)
        )
        process.append(p)
    for p in process:
        p.start()
    for p in process:
        p.join()

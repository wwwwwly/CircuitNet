# Copyright 2022 CircuitNet. All rights reserved.

import torch
import torch.nn as nn

from collections import OrderedDict


def generation_init_weights(module):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):

            if hasattr(m, "weight") and m.weight is not None:
                nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:

                nn.init.constant_(m.bias, 0)

    module.apply(init_func)


def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)

    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None

    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append(
            "unexpected key in source " f'state_dict: {", ".join(unexpected_keys)}\n'
        )
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n'
        )

    if len(err_msg) > 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)

        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    return missing_keys


class conv(nn.Module):
    def __init__(
        self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True
    ):  # size_out = size_in
        super(conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                dim_out,
                dim_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class upconv(nn.Module):
    def __init__(self, dim_in, dim_out):  # size_out = 2 * size_in
        super(upconv, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class Encoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=32):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.c1 = conv(in_dim, 32)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # size_out = size_in / 2
        self.c2 = conv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Sequential(  # size_out = size_ in
            nn.Conv2d(64, out_dim, 3, 1, 1),
            nn.BatchNorm2d(out_dim),
            nn.Tanh(),
        )

    def init_weights(self):
        generation_init_weights(self)

    def forward(self, input):  # input : 256 * 256 * C
        h1 = self.c1(input)  # 256 * 256 * 32
        h2 = self.pool1(h1)  # 128 *128 * 32
        h3 = self.c2(h2)  # 128 *128 * 64
        h4 = self.pool2(h3)  # 64 * 64 * 64
        h5 = self.c3(h4)  # 64 * 64 * 32

        return h5, h2  # shortpath from 2->7


class Decoder(nn.Module):
    def __init__(self, out_dim=2, in_dim=32):
        super(Decoder, self).__init__()
        self.conv1 = conv(in_dim, 32)

        self.upc1 = upconv(32, 16)  # size_out = 2 * size_in
        self.conv2 = conv(16, 16)
        self.upc2 = upconv(32 + 16, 4)
        self.conv3 = nn.Sequential(nn.Conv2d(4, out_dim, 3, 1, 1), nn.Sigmoid())

    def init_weights(self):
        generation_init_weights(self)

    def forward(self, input):  # input 64 *64 * 32
        feature, skip = input
        d1 = self.conv1(feature)  # 64 * 64 * 32
        d2 = self.upc1(d1)  # 128 * 128 * 16
        d3 = self.conv2(d2)  # 128 * 128 * 16
        d4 = self.upc2(torch.cat([d3, skip], dim=1))  # 128 * 128 *48 -> 256 * 256 * 4
        output = self.conv3(
            d4
        )  # shortpath from 2->7 # 256 * 256 * out_dim   drc中 out_dim=1

        return output


class GPDL(nn.Module):

    def __init__(self, in_channels=3, out_channels=2, **kwargs):  # drc 中out_channels=1

        super().__init__()

        self.encoder = Encoder(in_dim=in_channels)
        self.decoder = Decoder(out_dim=out_channels)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def init_weights(
        self, pretrained=None, pretrained_transfer=None, strict=False, **kwargs
    ):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location="cpu")["state_dict"]

            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            generation_init_weights(self)
        else:
            raise TypeError(
                "'pretrained' must be a str or None. "
                f"But received {type(pretrained)}."
            )

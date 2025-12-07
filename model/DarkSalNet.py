import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.MobileViT import mobile_vit_small
from einops import rearrange
from model.CIDNet import CIDNet
import cv2
import numpy as np


mob_conv1_2 = mob_conv2_2 = mob_conv3_3 = mob_conv4_3 = mob_conv5_3 = None


def conv_1_2_hook(module, input, output):
    global mob_conv1_2
    mob_conv1_2 = output
    return None


def conv_2_2_hook(module, input, output):
    global mob_conv2_2
    mob_conv2_2 = output
    return None


def conv_3_3_hook(module, input, output):
    global mob_conv3_3
    mob_conv3_3 = output
    return None


def conv_4_3_hook(module, input, output):
    global mob_conv4_3
    mob_conv4_3 = output
    return None


def conv_5_3_hook(module, input, output):
    global mob_conv5_3
    mob_conv5_3 = output
    return None


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        # self.mbv = shufflenet_v2_x2_0(pretrained=True).features
        self.mbv = models.mobilenet_v3_large(pretrained=True).features
        self.mbv[1].register_forward_hook(conv_1_2_hook)
        self.mbv[3].register_forward_hook(conv_2_2_hook)
        self.mbv[6].register_forward_hook(conv_3_3_hook)
        self.mbv[12].register_forward_hook(conv_4_3_hook)
        self.mbv[15].register_forward_hook(conv_5_3_hook)

    def forward(self, x: Tensor) -> Tensor:
        global mob_conv1_2, mob_conv2_2, mob_conv3_3, mob_conv4_3, mob_conv5_3
        self.mbv(x)

        return mob_conv1_2, mob_conv2_2, mob_conv3_3, mob_conv4_3, mob_conv5_3


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )

    def forward(self, x):
        return self.reduce(x)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class IEB(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEB, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.GELU = nn.GELU()

        self.CA = CA(hidden_features)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.GELU(self.dwconv1(x1))
        x2 = self.GELU(self.dwconv2(x2))
        x = x1 * x2 + x1 + x2
        x = self.CA(x) * x
        x = self.project_out(x)
        return x


class CCA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CCA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class SCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(SCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEB(dim)
        self.ffn = CCA(dim, num_heads, bias=bias)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = x + self.gdfn(self.norm(x))
        return x


class CA(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class LKB(nn.Module):

    def __init__(self, channels, large_kernel, split_group):
        super(LKB, self).__init__()
        self.channels = channels
        self.split_group = split_group
        self.split_channels = int(channels // split_group)
        self.CA = CA(channels // split_group)
        self.DWConv_Kx1 = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=(large_kernel, 1), stride=1,
                                    padding=(large_kernel // 2, 0), groups=self.split_channels)
        self.DWConv_1xK = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=(1, large_kernel), stride=1,
                                    padding=(0, large_kernel // 2), groups=self.split_channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        # channel shuffle
        B, C, H, W = x.size()
        x = x.reshape(B, self.split_channels, self.split_group, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B, C, H, W)

        x1, x2 = torch.split(x, (self.split_channels, self.channels - self.split_channels), dim=1)

        # channel attention
        x1 = self.CA(x1) * x1

        x1 = self.DWConv_Kx1(self.DWConv_1xK(x1))
        out = torch.cat((x1, x2), dim=1)
        out = self.act(self.conv1(out))
        return out




class Scaler(nn.Module):
    def __init__(self, channels, init_value=1e-5, requires_grad=True):
        super(Scaler, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones(1, channels, 1, 1),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class CMF(nn.Module):

    def __init__(self, channels, large_kernel, split_group):
        super(CMF, self).__init__()
        self.LKB = LKB(channels, large_kernel, split_group)
        self.DWConv_3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.conv1 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.scaler1 = Scaler(channels)
        self.scaler2 = Scaler(channels)
        self.act = nn.GELU()

    def forward(self, x, y):
        x1 = self.LKB(x)
        x1_scaler = self.scaler1(x + x1)

        y2 = self.DWConv_3(y)
        y2_scaler = self.scaler2(y + y2)

        x1 = x1 * y2_scaler
        y2 = y2 * x1_scaler

        out = self.act(self.conv1(torch.cat((x1, y2), dim=1)))
        return out


class NormUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2, use_norm=False):
        super(NormUpsample, self).__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
        self.up = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, y):
        x = self.up_scale(x)
        x = torch.cat([x, y], dim=1)
        x = self.up(x)
        x = self.prelu(x)
        if self.use_norm:
            return self.norm(x)
        else:
            return x


class MSG(nn.Module):
    def __init__(self, channel):
        super(MSG, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)
        self.S_conv = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )

    def forward(self, fl, fh, f5, f4=None, f3=None):
        fgl1 = F.interpolate(f5, size=fl.size()[2:], mode='bilinear')
        fh = self.conv1(fgl1 * fh) + fh
        fl = self.conv2(fgl1 * fl) + fl
        out = self.S_conv(torch.cat((fh, fl), 1))
        if f4 is not None:
            fgl1 = F.interpolate(f5, size=fl.size()[2:], mode='bilinear')
            fgl2 = F.interpolate(f4, size=fl.size()[2:], mode='bilinear')
            fh = self.conv1(fgl1 * fgl2 * fh) + fh
            fl = self.conv2(fgl1 * fgl2 * fl) + fl
            out = self.S_conv(torch.cat((fh, fl), 1))
        else:
            if f4 is not None:
                if f3 is not None:
                    fgl1 = F.interpolate(f5, size=fl.size()[2:], mode='bilinear')
                    fgl2 = F.interpolate(f4, size=fl.size()[2:], mode='bilinear')
                    fgl3 = F.interpolate(f3, size=fl.size()[2:], mode='bilinear')
                    fh = self.conv1(fgl1 * fgl2 * fgl3 * fh) + fh
                    fl = self.conv2(fgl1 * fgl2 * fgl3 * fl) + fl
                    out = self.S_conv(torch.cat((fh, fl), 1))

        return out


class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DarkSalNet(nn.Module):
    def __init__(self, channel=32, cidnet_ckpt_path = '/home/dell/HJL/remote/work/HVI-DARK/models/fivek.pth'):
        super(DarkSalNet, self).__init__()

        self.Encoder1 = MobileNet()
        self.Encoder2 = mobile_vit_small()

        self.Translayer1 = Reduction(16, channel)
        self.Translayer2 = Reduction(24, channel)
        self.Translayer3 = Reduction(40, channel)
        self.Translayer4 = Reduction(112, channel)
        self.Translayer5 = Reduction(160, channel)

        self.Translayery1 = Reduction(32, channel)
        self.Translayery2 = Reduction(64, channel)
        self.Translayery3 = Reduction(96, channel)
        self.Translayery4 = Reduction(128, channel)
        self.Translayery5 = Reduction(160, channel)

        self.CMF1 = CMF(channel, 3, 2)
        self.CMF2 = CMF(channel, 5, 2)
        self.CMF3 = CMF(channel, 7, 2)
        self.SCA4 = SCA(channel, 8)
        self.SCA5 = SCA(channel, 16)

        self.Decoder1 = NormUpsample(channel, channel)
        self.Decoder2 = NormUpsample(channel, channel)
        self.Decoder3 = NormUpsample(channel, channel)
        self.Decoder4 = NormUpsample(channel, channel)

        self.s_conv = nn.Conv2d(channel, 1, 3, padding=1)
        self.s_conv1 = nn.Conv2d(channel, 1, 3, padding=1)
        self.s_conv2 = nn.Conv2d(channel, 1, 3, padding=1)
        self.s_conv3 = nn.Conv2d(channel, 1, 3, padding=1)
        self.s_conv4 = nn.Conv2d(channel, 1, 3, padding=1)



        self.trans_conv = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                           padding=0, dilation=1, bias=False)
        self.trans_conv1 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                           padding=0, dilation=1, bias=False)
        self.trans_conv2 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                           padding=0, dilation=1, bias=False)
        self.trans_conv3 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                           padding=0, dilation=1, bias=False)
        self.trans_conv4 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                           padding=0, dilation=1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.CIDNet = CIDNet()

        if cidnet_ckpt_path is not None:
            self._load_cidnet_weights(cidnet_ckpt_path, strict=False)

        for param in self.CIDNet.parameters():
            param.requires_grad = True

    def _load_cidnet_weights(self, path, strict=False):
        ckpt = torch.load(path)
        state_dict = self.CIDNet.state_dict()
        pretrained_dict = {k: v for k, v in ckpt.items() if k in state_dict}
        state_dict.update(pretrained_dict)

        self.CIDNet.load_state_dict(state_dict, strict=strict)

    def UVIT(self, x, y):
        uvi = self.CIDNet.UVIT(x, y)
        return uvi


    def forward(self, dark_image, lab):
        size = dark_image.size()[2:]


        output_rgb, UVI, uvi = self.CIDNet(dark_image, lab)

        uvi1, uvi2, uvi3, uvi4, uvi5 = self.Encoder1(UVI)

        dark1 = self.Encoder2.conv_1(output_rgb)
        dark1 = self.Encoder2.layer_1(dark1)
        dark2 = self.Encoder2.layer_2(dark1)
        dark3 = self.Encoder2.layer_3(dark2)
        dark4 = self.Encoder2.layer_4(dark3)
        dark5 = self.Encoder2.layer_5(dark4)

        uvi1 = self.Translayer1(uvi1)
        uvi2 = self.Translayer2(uvi2)
        uvi3 = self.Translayer3(uvi3)
        uvi4 = self.Translayer4(uvi4)
        uvi5 = self.Translayer5(uvi5)

        dark1 = self.Translayery1(dark1)
        dark2 = self.Translayery2(dark2)
        dark3 = self.Translayery3(dark3)
        dark4 = self.Translayery4(dark4)
        dark5 = self.Translayery5(dark5)

        F1 = self.CMF1(uvi1, dark1)
        F2 = self.CMF2(uvi2, dark2)
        F3 = self.CMF3(uvi3, dark3)
        F4 = self.SCA4(uvi4, dark4)
        F5 = self.SCA5(uvi5, dark5)


        S4 = self.Decoder4(F5, F4)
        S3 = self.Decoder3(S4, F3)
        S2 = self.Decoder2(S3, F2)
        S1 = self.Decoder1(S2, F1)
        S = self.trans_conv(S1)


        Pre4 = self.s_conv4(S4)
        Pre3 = self.s_conv3(S3)
        Pre2 = self.s_conv2(S2)
        Pre1 = self.s_conv1(S1)
        Pre = self.s_conv(S)

        Pre1 = F.interpolate(Pre1, size=size, mode='bilinear', align_corners=True)
        Pre2 = F.interpolate(Pre2, size=size, mode='bilinear', align_corners=True)
        Pre3 = F.interpolate(Pre3, size=size, mode='bilinear', align_corners=True)
        Pre4 = F.interpolate(Pre4, size=size, mode='bilinear', align_corners=True)



        return Pre, Pre1, Pre2, Pre3, Pre4, self.sigmoid(Pre), self.sigmoid(Pre1), self.sigmoid(Pre2), self.sigmoid(Pre3), self.sigmoid(Pre4), output_rgb
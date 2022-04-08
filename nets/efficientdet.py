import torch
import torch.nn as nn
from utils.anchors import Anchors

from nets.efficientnet import EfficientNet as EffNet
from nets.layers import (Conv2dStaticSamePadding, MaxPool2dStaticSamePadding,
                         MemoryEfficientSwish, Swish)
from encode import DilateEncoder

from self_attention_cv.bottleneck_transformer import BottleneckAttention

from nets.attention import cbam_block, eca_block, se_block

attention_block = [se_block, cbam_block, eca_block]



class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=3, stride=1,
                                                      groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class BiFPN(nn.Module):
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        # self.mhsa_p3 = BottleneckAttention(dim=112,fmap_size=(96, 96),heads=4)
        # self.mhsa_p4 = BottleneckAttention(dim=112,fmap_size=(48, 48),heads=4)
        # self.mhsa_p5 = BottleneckAttention(dim=112, fmap_size=(24, 24), heads=16)
        self.mhsa_p6 = BottleneckAttention(dim=112, fmap_size=(12, 12), heads=16)
        self.mhsa_p7 = BottleneckAttention(dim=112, fmap_size=(6, 6),   heads=16)

        self.first_time = first_time
        if self.first_time:
   
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            self.p5_to_p6 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                MaxPool2dStaticSamePadding(3, 2)
            )
            self.p6_to_p7 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2)
            )


            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )


        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        """ bifpn
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """
        if self.attention:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward_fast_attention(inputs)
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward(inputs)

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward_fast_attention(self, inputs):
  
        #         C3  64, 64, 40
        #         C4  32, 32, 112
        #         C5  16, 16, 320

        if self.first_time:

            p3, p4, p5 = inputs  # d1: p3(40,80,80) p4(112,40,40) p5(320,20,20)

            p3_in = self.p3_down_channel(p3)  # d1:torch.Size([2, 88, 80, 80])

            #   C4 32, 32, 112 -> 32, 32, 64
   
 
            p4_in_1 = self.p4_down_channel(p4)  # d1:torch.Size([2, 88, 40, 40])
            p4_in_2 = self.p4_down_channel_2(p4)  # d1:torch.Size([2, 88, 40, 40])

 
            #   C5 16, 16, 320 -> 16, 16, 64
  
            p5_in_1 = self.p5_down_channel(p5)  # d1:torch.Size([2, 88, 20, 20])
            p5_in_2 = self.p5_down_channel_2(p5)  # d1:torch.Size([2, 88, 20, 20])

            #   C5 16, 16, 320 -> 8, 8, 64
   
            p6_in = self.p5_to_p6(p5)  # d1:torch.Size([2, 88, 10, 10])

            #   P6_in 8, 8, 64 -> 4, 4, 64

            p7_in = self.p6_to_p7(p6_in)  # d1:torch.Size([2, 88, 5, 5])

            # p3_in = self.mhsa_p3(p3_in)
            # p4_in_1 = self.mhsa_p4(p4_in_1)
            # p4_in_2 = self.mhsa_p4(p4_in_2)
            # p5_in_1 = self.mhsa_p5(p5_in_1)
            # p5_in_2 = self.mhsa_p5(p5_in_2)
            p6_in = self.mhsa_p6(p6_in)
            p7_in = self.mhsa_p7(p7_in)

            p6_w1 = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            p6_td = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            p5_td = self.conv5_up(self.swish(weight[0] * p5_in_1 + weight[1] * self.p5_upsample(p6_td)))

            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p4_td = self.conv4_up(self.swish(weight[0] * p4_in_1 + weight[1] * self.p4_upsample(p5_td)))

            p3_w1 = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))

            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            p4_out = self.conv4_down(
                self.swish(weight[0] * p4_in_2 + weight[1] * p4_td + weight[2] * self.p4_downsample(p3_out)))

            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            p5_out = self.conv5_down(
                self.swish(weight[0] * p5_in_2 + weight[1] * p5_td + weight[2] * self.p5_downsample(p4_out)))

            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p6_out = self.conv6_down(
                self.swish(weight[0] * p6_in + weight[1] * p6_td + weight[2] * self.p6_downsample(p5_out)))

            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))


        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

            # p3_in = self.mhsa_p3(p3_in)
            # p4_in = self.mhsa_p4(p4_in)
            # p5_in = self.mhsa_p5(p5_in)
            p6_in = self.mhsa_p6(p6_in)
            p7_in = self.mhsa_p7(p7_in)

            p6_w1 = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            p6_td = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            p5_td = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_td)))


            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p4_td = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_td)))

            p3_w1 = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))

            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            p4_out = self.conv4_down(
                self.swish(weight[0] * p4_in + weight[1] * p4_td + weight[2] * self.p4_downsample(p3_out)))

            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            p5_out = self.conv5_down(
                self.swish(weight[0] * p5_in + weight[1] * p5_td + weight[2] * self.p5_downsample(p4_out)))

            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p6_out = self.conv6_down(
                self.swish(weight[0] * p6_in + weight[1] * p6_td + weight[2] * self.p6_downsample(p5_out)))

            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        # phi=6、7
        if self.first_time:
 
            # p3_in p4_in p5_in p6_in p7_in
            p3, p4, p5 = inputs
            p3_in = self.p3_down_channel(p3)
            p4_in_1 = self.p4_down_channel(p4)
            p4_in_2 = self.p4_down_channel_2(p4)
            p5_in_1 = self.p5_down_channel(p5)
            p5_in_2 = self.p5_down_channel_2(p5)
            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p6_td = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

            p5_td = self.conv5_up(self.swish(p5_in_1 + self.p5_upsample(p6_td)))

            p4_td = self.conv4_up(self.swish(p4_in_1 + self.p4_upsample(p5_td)))

            p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_td)))

            p4_out = self.conv4_down(
                self.swish(p4_in_2 + p4_td + self.p4_downsample(p3_out)))

            p5_out = self.conv5_down(
                self.swish(p5_in_2 + p5_td + self.p5_downsample(p4_out)))

            p6_out = self.conv6_down(
                self.swish(p6_in + p6_td + self.p6_downsample(p5_out)))

            p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

            p6_td = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

            p5_td = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_td)))

            p4_td = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_td)))

            p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_td)))

            p4_out = self.conv4_down(
                self.swish(p4_in + p4_td + self.p4_downsample(p3_out)))

            p5_out = self.conv5_down(
                self.swish(p5_in + p5_td + self.p5_downsample(p4_out)))

            p6_out = self.conv6_down(
                self.swish(p6_in + p6_td + self.p6_downsample(p5_out)))

            p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out


class BoxNet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_layers, onnx_export=False):
        super(BoxNet, self).__init__()
        self.num_layers = num_layers

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])

        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(5)])

        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []

        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)

        return feats


class ClassNet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers, onnx_export=False):
        super(ClassNet, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])

        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(5)])

        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []

        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
        
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
                                          self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)
    
        feats = feats.sigmoid()

        return feats


class EfficientNet(nn.Module):
    def __init__(self, phi, load_weights=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{phi}', load_weights)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps[1:]


class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=80, phi=0, load_weights=False, ph=0):
        super(EfficientDetBackbone, self).__init__()

        self.phi = phi
        self.ph = ph

        self.backbone_phi = [0, 1, 2, 3, 4, 5, 6, 6]

        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]

        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]

        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]

        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]

        num_anchors = 9
        conv_channel_coef = {
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
        }
        #self.SPP = SPPBottleneck(conv_channel_coef[phi][2], conv_channel_coef[phi][2])
        # self.SPP = SPPBottleneck(112,112)
        # self.encode = DilateEncoder(conv_channel_coef[phi][2], conv_channel_coef[phi][2])
        #self.encode = DilateEncoder(112,112)

        if 1 <= self.ph and self.ph <= 3:
            # self.p3_att = attention_block[self.ph - 1](conv_channel_coef[self.phi][0])
            # self.p4_att = attention_block[self.ph - 1](conv_channel_coef[self.phi][1])
            # self.p5_att = attention_block[self.ph - 1](conv_channel_coef[self.phi][2])
            #self.bifpn_att = attention_block[self.ph - 1](88)
            self.bifpn_att = attention_block[self.ph - 1](112)

        # self.mhsa_p3 = BottleneckAttention(dim=88,fmap_size=(80, 80),heads=4)
        # self.mhsa_p4 = BottleneckAttention(dim=88,fmap_size=(40, 40),heads=4)
        # self.mhsa_p5 = BottleneckAttention(dim=88,fmap_size=(20, 20),heads=4)
        # self.mhsa_p6 = BottleneckAttention(dim=88,fmap_size=(10, 10),heads=4)
        # self.mhsa_p7 = BottleneckAttention(dim=88,fmap_size=(5, 5), heads=4)

        # self.mhsa_p5 = BottleneckAttention(dim=112, fmap_size=(24, 24), heads=4)
        # self.mhsa_p6 = BottleneckAttention(dim=112, fmap_size=(12, 12), heads=16)
        # self.mhsa_p7 = BottleneckAttention(dim=112, fmap_size=(6, 6), heads=16)


        #   P3_out      64,64,64
        #   P4_out      32,32,64
        #   P5_out      16,16,64
        #   P6_out      8,8,64
        #   P7_out      4,4,64
        
        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.phi],
                    conv_channel_coef[phi],
                    True if _ == 0 else False,
                    attention=True if phi < 6 else False)
              for _ in range(self.fpn_cell_repeats[phi])])

        self.num_classes = num_classes

        self.regressor = BoxNet(in_channels=self.fpn_num_filters[self.phi], num_anchors=num_anchors,
                                num_layers=self.box_class_repeats[self.phi])

        self.classifier = ClassNet(in_channels=self.fpn_num_filters[self.phi], num_anchors=num_anchors,
                                   num_classes=num_classes, num_layers=self.box_class_repeats[self.phi])

        self.anchors = Anchors(anchor_scale=self.anchor_scale[phi])

        self.backbone_net = EfficientNet(self.backbone_phi[phi], load_weights)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        _, p3, p4, p5 = self.backbone_net(inputs)
        #
        # if 1 <= self.ph and self.ph <= 3:
        #     p3 = self.p3_att(p3)
        #     p4 = self.p4_att(p4)
        #     p5 = self.p5_att(p5)

        # p5 = self.SPP(p5)
        # p5 = self.encode(p5)
        features = (p3, p4, p5)

        features = self.bifpn(features)

        #features = list(self.bifpn(features))
        # if 1 <= self.ph and self.ph <= 3:
        #     features[0] = self.bifpn_att(features[0])
        #     features[1] = self.bifpn_att(features[1])
        #     features[2] = self.bifpn_att(features[2])
        #     features[3] = self.bifpn_att(features[3])
        #     features[4] = self.bifpn_att(features[4])
        #     features = (features[0], features[1], features[2], features[3], features[4])

        # features[0] = self.mhsa_p3(features[0])
        # features[1] = self.mhsa_p4(features[1])
        # features[2] = self.mhsa_p5(features[2])
        # features[3] = self.mhsa_p6(features[3])
        # features[4] = self.mhsa_p7(features[4])

        # features[0] = self.encode(features[0])
        # features[1] = self.encode(features[1])
        # features[2] = self.encode(features[2])
        # features[3] = self.encode(features[3])
        # features[4] = self.encode(features[4])
        # features = (features[0], features[1], features[2], features[3], features[4])

        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(inputs)

        return features, regression, classification, anchors

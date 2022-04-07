import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act='relu'):
        super(Conv, self).__init__()
        if act is not None:
            if act == 'relu':
                self.convs = nn.Sequential(
                    nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.ReLU(inplace=True) if act else nn.Identity()
                )
            elif act == 'leaky':
                self.convs = nn.Sequential(
                    nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
                )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                nn.BatchNorm2d(c2)
            )

    def forward(self, x):
        return self.convs(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c, d=1, e=0.5, act='relu'):
        super(Bottleneck, self).__init__()
        c_ = int(c * e)
        self.branch = nn.Sequential(
            Conv(c, c_, k=1, act=act),
            Conv(c_, c_, k=3, p=d, d=d, act=act),
            Conv(c_, c, k=1, act=act)
        )

    def forward(self, x):
        return x + self.branch(x)

class DilateEncoder(nn.Module):
    """ DilateEncoder """
    def __init__(self, c1, c2, act='relu', dilation_list=[2, 4, 6, 8]):
        super(DilateEncoder, self).__init__()
        self.projector = nn.Sequential(
            Conv(c1, c2, k=1, act=None),
            Conv(c2, c2, k=3, p=1, act=None)
        )
        encoders = []
        for d in dilation_list:
            encoders.append(Bottleneck(c=c2, d=d, act=act))
        self.encoders = nn.Sequential(*encoders)

    def forward(self, x):
        x = self.projector(x)
        x = self.encoders(x)

        return x





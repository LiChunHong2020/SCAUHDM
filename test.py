import torch

from nets.efficientdet import EfficientDetBackbone
from nets.efficientnet import EfficientNet

if __name__ == '__main__':
    inputs = torch.randn(4, 3, 512, 512)
    model = EfficientDetBackbone(80,0)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    

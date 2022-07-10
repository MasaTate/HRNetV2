import torch
import torch.nn as nn

"""
There are 4 resolution levels in HRNet. We name each level as large, middle, small, tiny.
"""

class firstBlock(nn.Module):
    def __init__(self, in_channel, out_channel, identity_conv=None):
        super(firstBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_channel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channel)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.identity_conv = identity_conv
        
    def forward(self, x):
        identity = x.clone()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        # add 1x1conv to match number of channels
        if self.identity_conv is not None:
            identity = self.identity_conv(identity)
        x += identity
        x = self.relu(x)
        
        return x
    
class firstStage(nn.Module):
    def __init__(self, block, C):
        super(firstStage, self).__init__()
        
        self.units = self._make_units(block, 4)
        self.conv2large = nn.Conv2d(64, C, kernel_size=3, stride=1, padding=1)
        self.conv2middle = nn.Conv2d(64, 2*C, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.units(x)
        x_large = self.conv2large(x)
        x_middle = self.conv2middle(x)
        
        return x_large, x_middle
        
    def _make_units(self, block, num_units):
        layers = []
        
        # 1st unit
        identity_conv = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        layers.append(firstBlock(64, 256, identity_conv))
        
        # 2~num_units units
        for i in range(num_units - 1):
            layers.append(firstBlock(256, 256, identity_conv=None))
            
        return nn.Sequential(*layers)
        
class HRNetV2(nn.Module):
    def __init__(self, C):
        super(HRNetV2, self).__init__()
        
        # stem stage
        self.conv0_1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(64)
        self.conv0_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 1st stage
        self.firstStage = firstStage(firstBlock, C)
        
        # 2nd stage
        
        # 3rd stage
        
        # 4th stage
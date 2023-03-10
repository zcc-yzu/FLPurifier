import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.use_shortcut = stride != 1 or in_planes != self.expansion*planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)

        self.shortcut_conv = nn.Sequential()
        if self.use_shortcut:
            self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes, affine=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut_conv(x)
        if self.use_shortcut:
            shortcut = self.shortcut_bn(shortcut)
        out += shortcut
        return F.relu(out)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.use_shortcut = stride != 1 or in_planes != self.expansion*planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes, affine=True)

        self.shortcut_conv = nn.Sequential()
        if self.use_shortcut:
            self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes, affine=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        shortcut = self.shortcut_conv(x)
        if self.use_shortcut:
            shortcut = self.shortcut_bn(shortcut)
        out += shortcut
        return F.relu(out)

# Model class
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cfg=None):
        super(ResNet, self).__init__()
        self.train_sup = (num_classes > 0)
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.output_dim = 512*block.expansion
        if(self.train_sup):
            self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        if(self.train_sup):
            out = self.linear(out)
        return out

class ResNet_basic(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cfg=None):
        super(ResNet_basic, self).__init__()
        self.train_sup = (num_classes > 0)

        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16, affine=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.output_dim = 512*block.expansion
        if(self.train_sup):
            self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        if(self.train_sup):
            out = self.linear(out)
        return out


def get_block(block):
    if(block=="BasicBlock"):
        return BasicBlock
    elif(block=="Bottleneck"):
        return Bottleneck

def ResNet18(num_classes=10, block="BasicBlock"):
    return ResNet(get_block(block), [2,2,2,2], num_classes=num_classes)

def ResNet34(num_classes=10, block="BasicBlock"):
    return ResNet(get_block(block), [3,4,6,3], num_classes=num_classes)

def ResNet56(num_classes=10, block="BasicBlock"):
    return ResNet_basic(get_block(block), [9,9,9], num_classes=num_classes)

### Retrieval function for backbones ###
def create_backbone(name, num_classes=10, block='BasicBlock'):
    if(name == 'res18'):
        net = ResNet18(num_classes=num_classes, block=block)
    elif(name == 'res34'):
        net = ResNet34(num_classes=num_classes, block=block)
    elif(name == 'res56'):
        net = ResNet56(num_classes=num_classes, block=block)

    return net

class MLP_BYOL(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=512, is_pred=False):
        super(MLP_BYOL, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim, affine=True)
        self.layer2 = nn.Linear(hidden_dim, in_dim, bias=is_pred)

    def forward(self, x):
        x = self.layer2(F.relu(self.bn1(self.linear1(x))))
        return x

# BYOL
class byol(nn.Module):
    def __init__(self, bbone_arch):
        super(byol, self).__init__()
        # Online model

        self.backbone = create_backbone(bbone_arch, num_classes=0)
        self.projector = MLP_BYOL(in_dim=self.backbone.output_dim, hidden_dim=1024, out_dim=512)

        ### Predictor (should be defined last for divergence aware update)
        self.predictor = MLP_BYOL(in_dim=512, hidden_dim=1024, out_dim=512, is_pred=True)
        self.f = nn.Linear(512,10)

    def forward(self, x1):
        pro = self.projector(self.backbone(x1))
        pre = self.predictor(pro)
        y = self.f(pre)

        return pro,pre,y


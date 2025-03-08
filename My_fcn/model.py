import torch
from torch import nn

def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

class BottleBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, dilation=1, stride=1, downsample=None):
        super(BottleBlock, self).__init__()
        self.conv1 = conv1x1(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Resnet50(nn.Module):
    def __init__(self, block, num_classes, in_channels=3, dilation_rate=[1, 2]):
        super(Resnet50, self).__init__()
        self.in_channels = 64
        self.conv1 = conv3x3(in_channels, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_number = [3, 4, 6, 3]
        self.layer1 = self._make_layer(block, 64, self.layer_number[0], stride=1)
        self.layer2 = self._make_layer(block, 128, self.layer_number[1], stride=2)
        self.layer3 = self._make_layer(block, 256, self.layer_number[2], stride=2, dilation=dilation_rate[0])
        self.layer4 = self._make_layer(block, 512, self.layer_number[3], stride=2, dilation=dilation_rate[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, channels, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.in_channels != block.expansion * channels:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, block.expansion * channels, stride),
                nn.BatchNorm2d(block.expansion * channels)
            )
        layers = []
        layers.append(block(self.in_channels, channels, dilation, stride, downsample))
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # 不要展平特征图，直接返回
        return x

class FCNRes50(nn.Module):
    def __init__(self, block, num_classes=21, in_channels=3):
        super(FCNRes50, self).__init__()
        self.resnet50 = Resnet50(block, num_classes, in_channels)
        self.resnet50.fc = nn.Identity()  # 移除 ResNet 的全连接层
        self.fcn_head = FCNhead(512 * block.expansion, num_classes)
        self.upsample = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16)

    def forward(self, x):
        x = self.resnet50(x)
        x = self.fcn_head(x)
        x = self.upsample(x)
        return x

class FCNhead(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1)
        ]
        super(FCNhead, self).__init__(*layers)

def make_model():
    return FCNRes50(BottleBlock)

if __name__ == "__main__":
    x = torch.rand([2, 3, 224, 224])  # 输入通道数为3，尺寸为224x224
    model = make_model()
    y = model(x)
    print(y.shape)

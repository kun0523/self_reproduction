import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1 # 膨胀因子， 主分支的卷积核个数
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        # super(BasicBlock, self).__init__()  # Python2
        super().__init__()  # Python3
        # 当 stride!=1 时 对应论文中的虚线连接，代表下采样
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample  # ??
        return

    def forward(self, x):
        identity = x  # 保存输入数据 用于后续残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # 将输出与恒等映射相加  要求channel数一致
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4  # 指定膨胀因子，控制主分支的卷积核个数 最后一层会变成第一层的四倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=self.expansion*out_channel, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_channel)
        self.relu = nn.ReLU()
        self.downsample = downsample
        return

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        '''

        :param block: 选用的block  18/34 对应 BasicBlock
        :param blocks_num:  残差结构的列表  例如 resnet18 [2,2,2,2]
        :param num_classes:
        :param include_top: 是否包含分类头
        '''
        super().__init__()

        self.include_top = include_top  # 分类头
        self.in_channel = 64
        # 图像大小减半
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))  # 直接指定 output_size
            self.fc = nn.Linear(512*block.expansion, num_classes)  # 18/34 512channel  50以上是 2048channel  由expansion因子自动调控

        # 对卷积层初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        return

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        print(">>>", out.shape)
        out = self.layer1(out)
        print(">>>", out.shape)
        out = self.layer2(out)
        print(">>>", out.shape)
        out = self.layer3(out)
        print(">>>", out.shape)
        out = self.layer4(out)
        print(">>>", out.shape)

        if self.include_top:
            out = self.avgpool(out)
            out = torch.flatten(out, 1)  # 若输入张量形状为 (batch_size, C, H, W)（常见于卷积层输出），执行该操作后形状会变为 (batch_size, C*H*W)，即后续维度被合并为一个一维向量
            out = self.fc(out)

        return out

    def _make_layer(self, block_, channels, blocks_num, stride=1):
        """

        :param block_: 用于创建残差层 block_ 需要对应 网络层数进行选取
        :param channels:
        :param blocks_num:
        :param stride:
        :return:
        """

        downsample = None
        if stride != 1 or self.in_channel != channels*block_.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channels*block_.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels*block_.expansion)
            )

        layers = []
        # 添加每个残差块的第一层 都是在第一层做下采样
        layers.append(block_(            self.in_channel,
            channels,
            downsample=downsample,
            stride=stride,
        ))
        self.in_channel = channels * block_.expansion
        for _ in range(1, blocks_num):
            layers.append(block_(self.in_channel, channels))

        return nn.Sequential(*layers)


def resnet18(num_class=1000, include_top=True, pretrained=False):
    return ResNet(BasicBlock, [2,2,2,2], num_class, include_top)

def resnet34(num_class=1000, include_top=True, pretrained=False):
    return ResNet(BasicBlock, [3,4,6,3], num_class, include_top)

def resnet50(num_class=1000, include_top=True, pretrained=False):
    return ResNet(Bottleneck, [3,4,6,3], num_class, include_top)

def resnet101(num_class=1000, include_top=True, pretrained=False):
    return ResNet(Bottleneck, [3,4,23,3], num_class, include_top)


if __name__ == "__main__":
    # b = BasicBlock(4, 4)
    # # b = Bottleneck(in_channel=4, out_channel=8)
    # # t = torch.randn((1, 64, 200, 200), dtype=torch.float)
    # t = torch.ones((1,4,200,200), dtype=torch.float)
    # o = b(t)

    res = resnet18(10)
    # res = resnet50(10)

    x = torch.ones((1,3,224,224), dtype=torch.float32)
    y = res(x)

    print()
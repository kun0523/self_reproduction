import torch
import torch.nn as nn

def conv_bn(in_channel, out_channel, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU6(inplace=True)
    )

def conv_dw(in_channel, out_channel, stride=1):
    return nn.Sequential(
        # part 1 3x3分组卷积 提取特征图
        nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=stride, padding=1, groups=in_channel, bias=False),
        nn.BatchNorm2d(in_channel),
        nn.ReLU6(inplace=True),

        # part 2 1x1卷积 压缩通道维度
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU6(inplace=True),
    )

class MobileNetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),

            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),

            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )

        self.stage2 = nn.Sequential(
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )

        self.stage3 = nn.Sequential(
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )

        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, 1000)
        return

    def forward(self, x):
        print("Shape::", x.shape)
        out = self.stage1(x)
        print("Shape::", out.shape)
        out = self.stage2(out)
        print("Shape::", out.shape)
        out = self.stage3(out)
        print("Shape::", out.shape)
        out = self.avg(out)
        print("Shape::", out.shape)
        out = out.view(-1, 1024)
        print("Shape::", out.shape)
        out = self.fc(out)
        print("Shape::", out.shape)
        return out

if __name__ == "__main__":
    from torchsummary import summary

    model = MobileNetV1()
    x = torch.ones((10, 3, 224, 224), dtype=torch.float)
    y = model(x)
    model.to("cuda")
    summary(model, input_size=(3, 224, 224))
    print()

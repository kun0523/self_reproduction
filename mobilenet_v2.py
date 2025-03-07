import torch.nn as nn
from torch.hub import load_state_dict_from_url

model_urls = {
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
}

# ???
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v+divisor/2)//divisor*divisor)
    if new_v < 0.9*v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size-1)//2
        super().__init__(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )
        return

class InvertedResidual(nn.Module):
    def __init__(self, inc, outc, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in (1, 2)

        # 中间层 channel 数， 先扩大
        hidden_dim = int(round(inc*expand_ratio))
        self.use_res_connect = self.stride == 1 and inc==outc

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inc, hidden_dim, kernel_size=1))

        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, outc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outc),
        ])
        self.conv = nn.Sequential(*layers)
        return

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                # 208,208,32 -> 208,208,16
                [1, 16, 1, 1],
                # 208,208,16 -> 104,104,24
                [6, 24, 2, 2],
                # 104,104,24 -> 52,52,32
                [6, 32, 3, 2],

                # 52,52,32 -> 26,26,64
                [6, 64, 4, 2],
                # 26,26,64 -> 26,26,96
                [6, 96, 3, 1],

                # 26,26,96 -> 13,13,160
                [6, 160, 3, 2],
                # 13,13,160 -> 13,13,320
                [6, 320, 1, 1],
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty or a 4-element list, got {}".format(inverted_residual_setting))

        input_channel = _make_divisible(input_channel*width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel*max(1.0, width_mult), round_nearest)

        features = [ConvBNReLU(3, input_channel, stride=2)]
        # t expand_ratio  c width  n block_num  s stride
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c*width_mult, round_nearest)
            for i in range(n):
                stride = s if i==0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self. classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        return

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2,3])
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    from torchsummary import summary

    model = MobileNetV2()
    summary(model, input_size=(3, 224,224), device="cpu")
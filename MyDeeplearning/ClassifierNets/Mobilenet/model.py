import torch
from torch import nn


def _make_divisiable(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# 定义常用卷积模块
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2  # 保持特征尺寸
        # groups=1表示正常卷积
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.useshortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:  # 是否对features进行通道增加
            # 1x1 点卷积
            layers.append((ConvBNReLU(in_channel, hidden_channel, kernel_size=1)))
        layers.extend([
            # 3x3卷积
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1点卷积线性激活函数
            nn.Conv2d(hidden_channel, out_channels=out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.useshortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisiable(32 * alpha, round_nearest)
        last_channel = _make_divisiable(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        features = []

        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # 根据列表构建到残差结构

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisiable(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(in_channel=input_channel, out_channel=output_channel, stride=stride, expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, last_channel, 1))

        # 结合特征
        self.features = nn.Sequential(*features)

        # 建立分类器
        self.avgpool = nn.AdaptiveAvgPool1d((1, 1))
        # 具体来说，nn.AdaptiveAvgPool1d层会将输入张量中的每个通道分别进行平均池化，确保输出的每个通道只有一个元素
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = MobileNetV2()
    print(model)

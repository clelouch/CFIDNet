import torch
from torch import nn
import torch.nn.functional as F


def norm_layer(channel, norm_name='gn'):
    if norm_name == 'bn':
        return nn.BatchNorm2d(channel)
    elif norm_name == 'gn':
        return nn.GroupNorm(min(32, channel // 4), channel)


class ChannelAttention(nn.Module):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Linear(channel, channel // 4, bias=False), nn.ReLU(inplace=True))
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Sequential(nn.Linear(channel // 4, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        n, c, h, w = x.shape
        y1 = self.avg_pool(x)
        y1 = y1.reshape(n, -1)
        y = self.fc2(self.fc1(y1))
        y = y.reshape(n, c, 1, 1).expand_as(x).clone()
        y = x * y
        return y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        try:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            scale = torch.cat([avg_out, max_out], dim=1)
            scale = self.conv(scale)
            out = x * self.sigmoid(scale)
        except Exception as e:
            print(e)
            out = x

        return out


class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        channel = config.channels * 3
        ratio = config.ratio
        self.conv_query = nn.Conv2d(channel, channel // ratio, 1, bias=False)
        self.conv_key = nn.Conv2d(channel, channel // ratio, 1, bias=False)
        self.conv_value = nn.Conv2d(channel, channel, 1, bias=False)

    def forward(self, x):
        n, c, h, w = x.shape
        feature_q = self.conv_query(x).view(n, -1, h * w).permute(0, 2, 1)
        feature_k = self.conv_key(x).view(n, -1, h * w)
        mask = torch.bmm(feature_q, feature_k)
        mask = torch.softmax(mask, dim=-1)
        feature_v = self.conv_value(x).view(n, c, -1)
        feat = torch.bmm(feature_v, mask.permute(0, 2, 1))
        feat = feat.view(n, c, h, w)
        return feat


class SelfRefine(nn.Module):
    def __init__(self, config):
        super(SelfRefine, self).__init__()
        channel = config.channels * 3
        self.att = SelfAttention(config)
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(self.att(x)))


class CrossModality(nn.Module):
    def __init__(self, config):
        super(CrossModality, self).__init__()
        channel = config.channels
        self.ca = ChannelAttention(channel)

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel)
        )
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, rgb_feature, depth_feature):
        depth_feature = self.ca(depth_feature)
        depth_feature = self.relu(depth_feature + self.conv1(depth_feature))
        rgb_feature = rgb_feature + self.conv2(torch.cat([rgb_feature, depth_feature], dim=1))
        return self.relu2(rgb_feature)


class ReduceChannnel(nn.Module):
    def __init__(self, config, ch_list):
        super(ReduceChannnel, self).__init__()
        self.channel = config.channels
        convs = []
        for i in range(len(ch_list)):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(ch_list[i], self.channel, 1, 1, bias=False),
                    norm_layer(self.channel),
                    nn.ReLU(inplace=True)
                )
            )

        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        ret = []
        for i in range(len(x)):
            ret.append(self.convs[i](x[i]))
        return ret


class MS(nn.Module):
    def __init__(self, config):
        super(MS, self).__init__()
        self.config = config
        convs = []
        convs.append(nn.Sequential(
            nn.Conv2d(config.channels * 3, config.channels, 3, 2, 1, bias=False),
            norm_layer(config.channels)
        ))
        for branch in range(2, config.branches):
            convs.append(nn.Sequential(
                nn.Conv2d(config.channels, config.channels, 3, 2, 1, bias=False),
                norm_layer(config.channels)
            ))
        if config.branches != 1:
            convs.append(nn.Sequential(
                nn.Conv2d(config.channels, config.channels, 3, 1, 1, dilation=2, bias=False),
                norm_layer(config.channels)
            ))

        convs2 = []
        for i in range(1, config.branches):
            convs2.append(nn.Sequential(
                nn.Conv2d(config.channels, config.channels, 3, 1, 1, bias=False),
                norm_layer(config.channels)
            ))
        self.convs = nn.ModuleList(convs)
        self.convs2 = nn.ModuleList(convs2)
        self.final = nn.Sequential(
            nn.Conv2d(config.channels, config.channels, 3, 1, 1, bias=False),
            norm_layer(config.channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = []
        target_shape = x.shape[2:]
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            res.append(x)

        for i in range(len(res) - 1, 0, -1):
            res[i] = F.interpolate(res[i], res[i - 1].shape[2:], mode='bilinear', align_corners=True)
            res[i - 1] = res[i] + res[i - 1]
            res[i - 1] = self.convs2[i - 1](res[i - 1])
        return self.final(F.interpolate(res[0], target_shape, mode='bilinear', align_corners=True))


class IAM(nn.Module):
    def __init__(self, config):
        super(IAM, self).__init__()
        self.config = config
        self.refine = SelfRefine(config)
        self.low_before = nn.Sequential(
            nn.Conv2d(config.channels, config.channels, 3, 2, 1, bias=False),
            norm_layer(config.channels),
            nn.ReLU(inplace=True)
        )
        self.high_before = nn.Sequential(
            nn.Conv2d(config.channels, config.channels, 3, 1, 1, bias=False),
            norm_layer(config.channels),
            nn.ReLU(inplace=True)
        )
        self.ms = MS(config)
        self.low_after = nn.Sequential(
            nn.Conv2d(config.channels, config.channels, 3, 1, 1, bias=False),
            norm_layer(config.channels),
            nn.ReLU(inplace=True)
        )
        self.high_after = nn.Sequential(
            nn.Conv2d(config.channels, config.channels, 3, 2, 1, bias=False),
            norm_layer(config.channels),
            nn.ReLU(inplace=True)
        )
        self.middle_after = nn.Sequential(
            nn.Conv2d(config.channels, config.channels, 3, 1, 1, bias=False),
            norm_layer(config.channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, low, middle, high):
        high_tmp = F.interpolate(high, middle.shape[2:], mode='bilinear', align_corners=True)
        high_tmp = self.high_before(high_tmp)
        middle_tmp = middle
        low_tmp = F.interpolate(self.low_before(low), middle.shape[2:], mode='bilinear', align_corners=True)

        fuse_tmp = torch.cat([high_tmp, middle_tmp, low_tmp], dim=1)
        fuse_tmp = self.refine(fuse_tmp)
        fuse = self.ms(fuse_tmp)

        high = high + F.interpolate(self.high_after(fuse), high.shape[2:], mode='bilinear', align_corners=True)
        middle = middle + self.middle_after(fuse)
        low = low + self.low_after(F.interpolate(fuse, low.shape[2:], mode='bilinear', align_corners=True))
        return low, middle, high

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from resnet import resnet50
from modules import CrossModality, ReduceChannnel, norm_layer, IAM

channel_list = [64, 256, 512, 1024, 2048]


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        channel = config.channels

        ms_blockes = []
        for i in range(3):
            ms_blockes.append(IAM(config))
        self.ms_blockes = nn.ModuleList(ms_blockes)

    def forward(self, rgb_features):
        rgb_features[2], rgb_features[3], rgb_features[4] = self.ms_blockes[2](*rgb_features[2:])
        rgb_features[1], rgb_features[2], rgb_features[3] = self.ms_blockes[1](*rgb_features[1:4])
        rgb_features[0], rgb_features[1], rgb_features[2] = self.ms_blockes[0](*rgb_features[:3])
        return rgb_features


class ScoreLayer(nn.Module):
    def __init__(self, channel):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2d(channel, 1, 1, 1)

    def forward(self, x, x_size):
        x = self.score(x)
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.depth_net = resnet50()
        self.rgb_net = resnet50()

        # reduce the channel number of depth net
        self.reduce_depth = ReduceChannnel(config, channel_list)
        self.reduce_rgb = ReduceChannnel(config, channel_list)

        crosses = []
        for i in range(5):
            crosses.append(CrossModality(config))
        self.crosses = nn.ModuleList(crosses)

        ############################ sal refine componnet ############################
        decoders = []
        for i in range(2):
            decoders.append(Decoder(config))
        self.decoders = nn.ModuleList(decoders)

        ############################ sal predict componnet ############################
        score = []
        for i in range(4 + 2):
            score.append(ScoreLayer(config.channels))
        self.scores = nn.ModuleList(score)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, rgb, depth):
        image_size = rgb.size()

        rgb_features = self.rgb_net(rgb)
        depth_features = self.depth_net(depth)

        # reduce channel number
        depth_features = self.reduce_depth(depth_features)
        rgb_features = self.reduce_rgb(rgb_features)

        # cross modality fusion
        for i in range(5):
            rgb_features[i] = self.crosses[i](rgb_features[i], depth_features[i])
        del depth_features

        sal_preds = []
        # sal refinement
        for i in range(2):
            rgb_features = self.decoders[i](rgb_features)
            sal_preds.append(rgb_features[0])

        # merge multi-level features for prediction
        for i in range(1, 5):
            rgb_features[i] = F.interpolate(rgb_features[i], rgb_features[0].shape[2:], mode='bilinear',
                                            align_corners=True)
            sal_preds.append(rgb_features[i])
        for i in range(len(sal_preds)):
            sal_preds[i] = torch.sigmoid(self.scores[i](sal_preds[i], image_size))
        return sal_preds


def print_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(num_params))

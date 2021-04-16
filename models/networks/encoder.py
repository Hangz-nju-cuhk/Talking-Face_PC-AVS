import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
import torchvision.models.mobilenet
from util import util
from models.networks.audio_network import ResNetSE, SEBasicBlock
import torch
from models.networks.FAN_feature_extractor import FAN_use
from torchvision.models.vgg import vgg19_bn
from models.networks.vision_network import ResNeXt50


class ResSEAudioEncoder(BaseNetwork):
    def __init__(self, opt, nOut=2048, n_mel_T=None):
        super(ResSEAudioEncoder, self).__init__()
        self.nOut = nOut
        # Number of filters
        num_filters = [32, 64, 128, 256]
        if n_mel_T is None: # use it when use audio identity
            n_mel_T = opt.n_mel_T
        self.model = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters, self.nOut, n_mel_T=n_mel_T)
        self.fc = nn.Linear(self.nOut, opt.num_classes)

    def forward_feature(self, x):

        input_size = x.size()
        if len(input_size) == 5:
            bz, clip_len, c, f, t = input_size
            x = x.view(bz * clip_len, c, f, t)
        out = self.model(x)
        return out

    def forward(self, x):
        out = self.forward_feature(x)
        score = self.fc(out)
        return out, score


class ResSESyncEncoder(ResSEAudioEncoder):
    def __init__(self, opt):
        super(ResSESyncEncoder, self).__init__(opt, nOut=512, n_mel_T=1)


class ResNeXtEncoder(ResNeXt50):
    def __init__(self, opt):
        super(ResNeXtEncoder, self).__init__(opt)


class VGGEncoder(BaseNetwork):
    def __init__(self, opt):
        super(VGGEncoder, self).__init__()
        self.model = vgg19_bn(num_classes=opt.num_classes)

    def forward(self, x):
        return self.model(x)


class FanEncoder(BaseNetwork):
    def __init__(self, opt):
        super(FanEncoder, self).__init__()
        self.opt = opt
        pose_dim = self.opt.pose_dim
        self.model = FAN_use()
        self.classifier = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, opt.num_classes))

        # mapper to mouth subspace
        self.to_mouth = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))
        self.mouth_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, 512-pose_dim))
        self.mouth_fc = nn.Sequential(nn.ReLU(), nn.Linear(512*opt.clip_len, opt.num_classes))

        # mapper to head pose subspace
        self.to_headpose = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))
        self.headpose_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, pose_dim))
        self.headpose_fc = nn.Sequential(nn.ReLU(), nn.Linear(pose_dim*opt.clip_len, opt.num_classes))

    def load_pretrain(self):
        check_point = torch.load(self.opt.FAN_pretrain_path)
        print("=> loading checkpoint '{}'".format(self.opt.FAN_pretrain_path))
        util.copy_state_dict(check_point, self.model)

    def forward_feature(self, x):
        net = self.model(x)
        return net

    def forward(self, x):
        x0 = x.view(-1, self.opt.output_nc, self.opt.crop_size, self.opt.crop_size)
        net = self.forward_feature(x0)
        scores = self.classifier(net.view(-1, self.opt.num_clips, 512).mean(1))
        return net, scores

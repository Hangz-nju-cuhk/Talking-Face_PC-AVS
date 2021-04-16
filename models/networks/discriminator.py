import torch.nn as nn
import numpy as np
from models.networks.base_network import BaseNetwork
import util.util as util
import torch
from models.networks.architecture import get_nonspade_norm_layer
import torch.nn.functional as F


class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt):
        super(MultiscaleDiscriminator, self).__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt):

        super(NLayerDiscriminator, self).__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        if opt.D_input == "concat":
            input_nc = opt.label_nc + opt.output_nc
            if opt.contain_dontcare_label:
                input_nc += 1
            if not opt.no_instance:
                input_nc += 1
        else:
            input_nc = 3
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():

            # intermediate_output = checkpoint(submodel, results[-1])
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[0:]
        else:
            return results[-1]


class AudioSubDiscriminator(BaseNetwork):
    def __init__(self, opt, nc, audio_nc):
        super(AudioSubDiscriminator, self).__init__()
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        sequence = []
        sequence += [norm_layer(nn.Conv1d(nc, nc, 3, 2, 1)),
                      nn.ReLU()
                      ]
        sequence += [norm_layer(nn.Conv1d(nc, audio_nc, 3, 2, 1)),
                      nn.ReLU()
                      ]

        self.conv = nn.Sequential(*sequence)
        self.cosine = nn.CosineSimilarity()
        self.mapping = nn.Linear(audio_nc, audio_nc)

    def forward(self, result, audio):
        region = result[result.shape[3] // 2:result.shape[3] - 2, result.shape[4] // 3:  2 * result.shape[4] // 3]
        visual = self.avgpool(region)
        cos = self.cosine(visual, self.mapping(audio))
        return cos


class ImageDiscriminator(BaseNetwork):
    """Defines a PatchGAN discriminator"""
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(ImageDiscriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        if opt.D_input == "concat":
            input_nc = opt.label_nc + opt.output_nc
        else:
            input_nc = opt.label_nc
        ndf = 64
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class FeatureDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super(FeatureDiscriminator, self).__init__()
        self.opt = opt
        self.fc = nn.Linear(512, opt.num_labels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x0 = x.view(-1, 512)
        net = self.dropout(x0)
        net = self.fc(net)
        return net



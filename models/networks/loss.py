import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.architecture import VGG19, VGGFace19


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, opt, vgg=VGG19()):
        super(VGGLoss, self).__init__()
        self.vgg = vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y, layer=0):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            if i >= layer:
                loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class CrossEntropyLoss(nn.Module):
    """Cross Entropy Loss

    It will calculate cross_entropy loss given cls_score and label.
    """

    def forward(self, cls_score, label):
        loss_cls = F.cross_entropy(cls_score, label)
        return loss_cls


class SumLogSoftmaxLoss(nn.Module):

    def forward(self, x):
        out = F.log_softmax(x, dim=1)
        loss = - torch.mean(out) + torch.mean(F.log_softmax(torch.ones_like(out), dim=1) )
        return loss


class L2SoftmaxLoss(nn.Module):
    def __init__(self):
        super(L2SoftmaxLoss, self).__init__()
        self.softmax = nn.Softmax()
        self.L2loss = nn.MSELoss()
        self.label = None

    def forward(self, x):
        out = self.softmax(x)
        self.label = (torch.ones(out.size()).float() * (1 / x.size(1))).cuda()
        loss = self.L2loss(out, self.label)
        return loss


class SoftmaxContrastiveLoss(nn.Module):
    def __init__(self):
        super(SoftmaxContrastiveLoss, self).__init__()
        self.cross_ent = nn.CrossEntropyLoss()

    def l2_norm(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        return x_norm

    def l2_sim(self, feature1, feature2):
        Feature = feature1.expand(feature1.size(0), feature1.size(0), feature1.size(1)).transpose(0, 1)
        return torch.norm(Feature - feature2, p=2, dim=2)

    @torch.no_grad()
    def evaluate(self, face_feat, audio_feat, mode='max'):
        assert mode in 'max' or 'confusion', '{} must be in max or confusion'.format(mode)
        face_feat = self.l2_norm(face_feat)
        audio_feat = self.l2_norm(audio_feat)
        cross_dist = 1.0 / self.l2_sim(face_feat, audio_feat)

        print(cross_dist)
        if mode == 'max':
            label = torch.arange(face_feat.size(0)).to(cross_dist.device)
            max_idx = torch.argmax(cross_dist, dim=1)
            # print(max_idx, label)
            acc = torch.sum(label == max_idx) * 1.0 / label.size(0)
        else:
            raise ValueError

        return acc

    def forward(self, face_feat, audio_feat, mode='max'):
        assert mode in 'max' or 'confusion', '{} must be in max or confusion'.format(mode)

        face_feat = self.l2_norm(face_feat)
        audio_feat = self.l2_norm(audio_feat)

        cross_dist = 1.0 / self.l2_sim(face_feat, audio_feat)

        if mode == 'max':
            label = torch.arange(face_feat.size(0)).to(cross_dist.device)
            loss = F.cross_entropy(cross_dist, label)
        else:
            raise ValueError
        return loss

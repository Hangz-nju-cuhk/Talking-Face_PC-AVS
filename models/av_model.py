import torch
import models.networks as networks
from models.networks.architecture import VGGFace19
import util.util as util
from models.networks.loss import CrossEntropyLoss
import os


class AvModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super(AvModel, self).__init__()
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
        self.netG, self.netD, self.netA, self.netA_sync, self.netV, self.netE = \
            self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.loss_cls = CrossEntropyLoss()
            self.criterionFeat = torch.nn.L1Loss()

            if opt.softmax_contrastive:
                self.criterionSoftmaxContrastive = networks.SoftmaxContrastiveLoss()
            if opt.train_recognition or opt.train_sync:
                pass

            else:
                self.criterionGAN = networks.GANLoss(
                    opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)

                if not opt.no_vgg_loss:
                    self.criterionVGG = networks.VGGLoss(self.opt)

                if opt.vgg_face:
                    self.VGGFace = VGGFace19(self.opt)
                    self.criterionVGGFace = networks.VGGLoss(self.opt, self.VGGFace)

            if opt.disentangle:
                self.criterionLogSoftmax = networks.L2SoftmaxLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    # |data|: dictionary of the input data
    def preprocessing(self, data):
        target_images = data['target'].cuda()
        input_image = data['input'].cuda()
        augmented = data['augmented'].cuda()
        spectrogram = data['spectrograms'].cuda() if self.opt.use_audio else None

        target_images = target_images.view(-1, self.opt.output_nc, self.opt.crop_size, self.opt.crop_size)
        augmented = augmented.view(-1, self.opt.output_nc, self.opt.crop_size, self.opt.crop_size)

        return input_image, target_images, augmented, spectrogram

    def forward(self, data, mode):
        labels = data['label']
        input_image, target_images, augmentated, spectrogram = self.preprocessing(data)
        if mode == 'generator':
            g_loss, generated, id_scores = self.compute_generator_loss(
                input_image, target_images, augmentated, spectrogram,
                 netD=self.netD, labels=labels, no_ganFeat_loss=self.opt.no_ganFeat_loss,
                no_vgg_loss=self.opt.no_vgg_loss, lambda_D=self.opt.lambda_D)
            return g_loss, generated, id_scores
        if mode == 'encoder':
            g_loss, cls_score = self.compute_encoder_loss(
                input_image, target_images, spectrogram, labels)
            return g_loss, cls_score
        if mode == 'sync':
            g_loss = self.sync(augmentated, spectrogram)
            return g_loss
        if mode == 'sync_D':
            d_loss = self.sync_D(spectrogram, labels)
            return d_loss
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_image, target_images, augmentated, spectrogram, netD=self.netD, labels=labels, lambda_D=self.opt.lambda_D)
            return d_loss
        elif mode == 'inference':
            assert self.opt.use_audio, 'must use audio driven strategy.'
            driving_pose_frames = data['driving_pose_frames'].cuda()
            with torch.no_grad():
                fake_image_ref_pose_a, fake_image_driven_pose_a = self.inference(input_image, spectrogram,
                                                                                    driving_pose_frames)
            return fake_image_ref_pose_a, fake_image_driven_pose_a
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        optimizer_D = None
        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        if opt.train_recognition:

            util.freeze_model(self.netV)
            for param in self.netV.fc.parameters():
                param.requires_grad = True
            netV_params = list(self.netV.fc.parameters())
            netA_params = list(self.netA.parameters())
            G_params = netV_params + netA_params

        elif opt.train_sync:

            netA_sync_params = list(self.netA_sync.model.parameters())
            # netE_params = list(self.netE.model.parameters())
            netE_mouth_params = list(self.netE.to_mouth.parameters())
            G_params = netA_sync_params + netE_mouth_params

            D_params = list(self.netA_sync.fc.parameters()) + list(self.netE.classifier.parameters())
            optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        elif opt.train_dis_pose:
            netE_pure_pose_params = list(self.netE.pure_pose.parameters())+list(self.netE.headpose_embed.parameters())
            netG_params = list(self.netG.parameters())
            netV_params = list(self.netV.parameters())
            netE_params = list(self.netE.model.parameters())
            netA_sync_params = list(self.netA_sync.parameters()) if self.opt.use_audio else None
            netE_mouth_all_params = list(self.netE.to_mouth.parameters()) + list(self.netE.mouth_fc.parameters())

            G_params = []

            if not opt.fix_netE_mouth:
                G_params = G_params + netE_mouth_all_params
            else:
                util.freeze_model(self.netE.to_mouth)
                util.freeze_model(self.netE.mouth_fc)

            if not opt.fix_netE_headpose:
                G_params = G_params + netE_pure_pose_params
            else:
                util.freeze_model(self.netE.pure_pose)
                util.freeze_model(self.netE.headpose_embed)

            if not opt.fix_netG:
                G_params = G_params + netG_params
            else:
                util.freeze_model(self.netG)

            if not opt.fix_netV:
                G_params = G_params + netV_params
            else:
                util.freeze_model(self.netV)

            if not opt.fix_netE:
                G_params = G_params + netE_params
            else:
                util.freeze_model(self.netE.model)

            if self.opt.use_audio:
                if not opt.fix_netA_sync:
                    G_params = G_params + netA_sync_params
                else:
                    util.freeze_model(self.netA_sync)

            if opt.isTrain:
                D_params = list(self.netD.parameters())

                if opt.disentangle:

                    if not opt.fix_netE_headpose:
                        D_params = list(self.netE.headpose_fc.parameters()) + D_params
                    else:
                        util.freeze_model(self.netE.headpose_fc)

                if not opt.fix_netD:
                    optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
                else:
                    util.freeze_model(self.netD)

        else:
            netG_params = list(self.netG.parameters())
            netA_sync_params = list(self.netA_sync.model.parameters()) if opt.use_audio else 0
            netE_mouth_params = list(self.netE.to_mouth.parameters())
            netV_params = list(self.netV.parameters())
            netE_params = list(self.netE.model.parameters())

            G_params = netA_sync_params + netE_mouth_params
            if not opt.fix_netV:
                G_params = G_params + netV_params
            else:
                util.freeze_model(self.netV)

            if not opt.fix_netE:
                G_params = G_params + netE_params
            else:
                util.freeze_model(self.netE)

            if not opt.fix_netG:
                G_params = G_params + netG_params
            else:
                util.freeze_model(self.netG)

            if opt.isTrain:
                D_params = list(self.netD.parameters())

                if opt.disentangle:
                    D_params = list(self.netE.classifier.parameters()) + D_params

                if not opt.fix_netD:
                    optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
                else:
                    util.freeze_model(self.netD)

        if opt.optimizer == 'sgd':
            optimizer_G = torch.optim.SGD(G_params, lr=G_lr, momentum=0.9, nesterov=True)
        else:
            optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), amsgrad=True)

        return optimizer_G, optimizer_D

    def save(self, epoch):
        if self.opt.train_recognition:
            util.save_network(self.netV, 'V', epoch, self.opt)
        elif self.opt.train_sync:
            util.save_network(self.netE, 'E', epoch, self.opt)
            if self.opt.use_audio:
                util.save_network(self.netA_sync, 'A_sync', epoch, self.opt)
        else:
            util.save_network(self.netG, 'G', epoch, self.opt)
            # util.save_network(self.netD, 'D', epoch, self.opt)
            if self.opt.use_audio:
                if self.opt.use_audio_id:
                    util.save_network(self.netA, 'A', epoch, self.opt)
                util.save_network(self.netA_sync, 'A_sync', epoch, self.opt)
            util.save_network(self.netV, 'V', epoch, self.opt)
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################


    def initialize_networks(self, opt):
        netG = None
        netD = None
        netE = None
        netV = None
        netA = None
        netA_sync = None
        if opt.train_recognition:
            netV = networks.define_V(opt)
        elif opt.train_sync:
            netA_sync = networks.define_A_sync(opt) if opt.use_audio else None
            netE = networks.define_E(opt)
        else:

            netG = networks.define_G(opt)
            netA = networks.define_A(opt) if opt.use_audio and opt.use_audio_id else None
            netA_sync = networks.define_A_sync(opt) if opt.use_audio else None
            netE = networks.define_E(opt)
            netV = networks.define_V(opt)

            if opt.isTrain:
                netD = networks.define_D(opt)

        if not opt.isTrain or opt.continue_train:
            self.load_network(netG, 'G', opt.which_epoch)
            self.load_network(netV, 'V', opt.which_epoch)
            self.load_network(netE, 'E', opt.which_epoch)
            if opt.use_audio:
                if opt.use_audio_id:
                    self.load_network(netA, 'A', opt.which_epoch)
                self.load_network(netA_sync, 'A_sync', opt.which_epoch)

            if opt.isTrain and not opt.noload_D:
                self.load_network(netD, 'D', opt.which_epoch)
                # self.load_network(netD_rotate, 'D_rotate', opt.which_epoch, pretrained_path)

        else:
            if self.opt.pretrain:
                if opt.netE == 'fan':
                    netE.load_pretrain()
                netV.load_pretrain()
            if opt.load_separately:
                netG = self.load_separately(netG, 'G', opt)
                netA = self.load_separately(netA, 'A', opt) if opt.use_audio and opt.use_audio_id else None
                netA_sync = self.load_separately(netA_sync, 'A_sync', opt) if opt.use_audio else None
                netV = self.load_separately(netV, 'V', opt)
                netE = self.load_separately(netE, 'E', opt)
                if not opt.noload_D:
                    netD = self.load_separately(netD, 'D', opt)
        return netG, netD, netA, netA_sync, netV, netE

    def compute_encoder_loss(self, input_img, real_image, spectrogram, labels):
        G_losses = {}
        real_image = real_image.view(-1, self.opt.clip_len,  self.opt.output_nc, self.opt.crop_size, self.opt.crop_size)

        [image_feature, net_V_feature], cls_score_V = self.netV.forward(real_image)
        audio_feature, cls_score_A_2 = self.netA.forward(spectrogram)
        audio_feature = audio_feature.view(-1, self.opt.clip_len, audio_feature.shape[-1])
        audio_feature = torch.mean(audio_feature, 1)

        G_losses['loss_cls_V'] = self.loss_cls(cls_score_V, labels)
        cls_score_A = self.netV.fc.forward(audio_feature)
        G_losses['loss_cls_A'] = self.loss_cls(cls_score_A, labels)
        # G_losses['loss_cls_A_2'] = self.loss_cls(cls_score_A_2, labels)
        if not self.opt.no_cross_modal:
            G_losses['CrossModal'] = self.criterionFeat(image_feature.detach(), audio_feature) * self.opt.lambda_crossmodal

        if self.opt.softmax_contrastive:
            G_losses['SoftmaxContrastive'] = self.criterionSoftmaxContrastive(image_feature.detach(), audio_feature) * self.opt.lambda_contrastive

        return G_losses, cls_score_A

    def sync_D(self, spectrogram, labels):
        D_losses = {}
        with torch.no_grad():
            audio_content_feature = self.netA_sync.forward_feature(spectrogram)
            audio_content_feature = audio_content_feature.detach()
            audio_content_feature.requires_grad_()
        cls_score_A = self.netA_sync.fc.forward(audio_content_feature)
        labels = labels.unsqueeze(1)
        labels_expand = labels.expand(-1, self.opt.clip_len)
        labels_expand = labels_expand.contiguous().view(-1)
        D_losses['loss_cls_A'] = self.loss_cls(cls_score_A, labels_expand)
        return D_losses


    def encode_audiosync_feature(self, spectrogram):

        audio_content_feature = self.netA_sync.forward_feature(spectrogram)

        audio_content_feature = audio_content_feature.view(-1, self.opt.clip_len, audio_content_feature.shape[-1])
        return audio_content_feature

    def sync(self, augmented, spectrogram):
        G_losses = {}
        pose_feature = self.encode_noid_feature(augmented)

        audio_content_feature = self.encode_audiosync_feature(spectrogram)

        G_losses = self.compute_sync_loss(pose_feature, audio_content_feature, G_losses)
        return G_losses

    def compute_sync_loss(self, image_content_feature, audio_content_feature, G_losses, name=''):

        audio_content_feature_all = audio_content_feature.view(audio_content_feature.shape[0], -1)
        image_content_feature_all = image_content_feature.view(image_content_feature.shape[0], -1)

        if not self.opt.no_cross_modal:
            G_losses['CrossModal{}'.format(name)] = self.criterionFeat(image_content_feature_all.detach(),
                                                        audio_content_feature_all) * self.opt.lambda_crossmodal

        if self.opt.softmax_contrastive:
            G_losses['SoftmaxContrastive{}'.format(name)] = self.criterionSoftmaxContrastive(image_content_feature_all.detach(), audio_content_feature_all) * self.opt.lambda_contrastive
            G_losses['SoftmaxContrastive_v2a'] = self.criterionSoftmaxContrastive(audio_content_feature_all.detach(), image_content_feature_all) * self.opt.lambda_contrastive

        return G_losses

    def audio_identity_feature(self, id_mel, no_grad=True):
        id_mel = id_mel.view(-1, 1, id_mel.shape[-2], id_mel.shape[-1])
        if no_grad:
            with torch.no_grad():
                id_feature, id_scores = self.netA(id_mel)
        else:
            id_feature, id_scores = self.netA(id_mel)
        return id_feature, id_scores

    def encode_identity_feature(self, input_img):

        input_img = input_img.view(-1, self.opt.output_nc, self.opt.crop_size, self.opt.crop_size)
        if not self.opt.isTrain or self.opt.fix_netV:
            with torch.no_grad():
                id_feature, id_scores = self.netV(input_img)
        else:
            id_feature, id_scores = self.netV(input_img)

        id_feature[0] = id_feature[0].unsqueeze(1).repeat(1, self.opt.clip_len, 1).view(-1, *id_feature[0].shape[1:])
        id_feature[1] = id_feature[1].unsqueeze(1).repeat(1, self.opt.clip_len, 1, 1, 1).view(-1, *id_feature[1].shape[1:])

        return id_feature, id_scores

    def encode_ref_noid(self, input_img):
        input_img = input_img.view(-1, self.opt.output_nc, self.opt.crop_size, self.opt.crop_size)
        with torch.no_grad():
            ref_noid_feature = self.netE.forward_feature(input_img)
        ref_noid_feature = ref_noid_feature.view(-1, self.opt.num_inputs, ref_noid_feature.shape[-1])
        ref_noid_feature = ref_noid_feature.mean(1).unsqueeze(1).repeat(1, self.opt.clip_len, 1)
        return ref_noid_feature

    def compute_pose_diff(self, pose_feature, ref_noid_feature):
        pose_feature = pose_feature.view(-1, self.opt.clip_len, pose_feature.shape[-1])
        pose_differences = pose_feature - ref_noid_feature
        return pose_differences

    def compute_diff_loss(self, input_img, pose_feature, pose_feature_audio, G_losses):

        pose_feature_audio = pose_feature_audio.view(-1, self.opt.clip_len, pose_feature_audio.shape[-1])
        ref_noid_feature = self.encode_ref_noid(input_img)
        pose_differences = self.compute_pose_diff(pose_feature, ref_noid_feature)

        self.compute_sync_loss(pose_differences, pose_feature_audio, G_losses)

        pose_feature_audio = ref_noid_feature + pose_feature_audio

        return pose_feature_audio

    def encode_noid_feature(self, augmented):
        augmented = augmented.view(-1, self.opt.output_nc, self.opt.crop_size, self.opt.crop_size)
        if (not self.opt.isTrain) or self.opt.train_sync or self.opt.fix_netE:
            with torch.no_grad():
                noid_feature = self.netE.forward_feature(augmented)
        else:
            noid_feature = self.netE.forward_feature(augmented)

        noid_feature = noid_feature.view(-1, self.opt.clip_len, noid_feature.shape[-1])
        return noid_feature

    def select_frames(self, in_obj_ts):
        if len(in_obj_ts.shape) == 2:
            obj_ts = in_obj_ts.view(-1, self.opt.clip_len, in_obj_ts.shape[-1])
            obj_ts = obj_ts[:, ::self.opt.generate_interval, :].contiguous()
            obj_ts = obj_ts.view(-1, obj_ts.shape[-1])
        elif len(in_obj_ts.shape) == 3:
            obj_ts = in_obj_ts[:, ::self.opt.generate_interval, :].contiguous()
        elif len(in_obj_ts.shape) == 4:
            obj_ts = in_obj_ts.view(-1, self.opt.clip_len, *in_obj_ts.shape[1:])
            obj_ts = obj_ts[:, ::self.opt.generate_interval, :].contiguous()
            obj_ts = obj_ts.view(-1, *obj_ts.shape[2:])
        elif len(in_obj_ts.shape) == 5:
            obj_ts = in_obj_ts[:, ::self.opt.generate_interval, :].contiguous()
        else:
            raise ValueError
        return obj_ts

    def generate_fake(self, id_feature, pose_feature):
        pose_feature = pose_feature.view(-1, pose_feature.shape[-1])
        style = torch.cat([id_feature[0], pose_feature], 1)
        style = [style]
        if self.opt.input_id_feature:
            fake_image, style_rgb = self.netG(style, identity_style=id_feature[1])
        else:
            fake_image, style_rgb = self.netG(style)

        fake_image = fake_image.view(-1, self.opt.output_nc, self.opt.crop_size, self.opt.crop_size)

        return fake_image, style_rgb

    def merge_mouthpose(self, mouth_feature, headpose_feature, embed_headpose=False):

        mouth_feature = self.netE.mouth_embed(mouth_feature)
        if not embed_headpose:
            headpose_feature = self.netE.headpose_embed(headpose_feature)
        pose_feature = torch.cat((mouth_feature, headpose_feature), dim=2)

        return pose_feature

    def inference(self, input_img, spectrogram,
                  driving_pose_frames, mouth_feature_weight=1.2):

        ##### ***************** encode image feature and generate ******************************
        id_feature, _ = self.encode_identity_feature(input_img)

        fake_image_pose_driven_a = None
        if self.opt.generate_from_audio_only:
            assert self.opt.use_audio, 'must use audio in this case'

        A_mouth_feature = self.encode_audiosync_feature(spectrogram)
        A_mouth_feature = A_mouth_feature * mouth_feature_weight

        sel_id_feature = []
        sel_id_feature.append(self.select_frames(id_feature[0]))
        sel_id_feature.append(self.select_frames(id_feature[1]))

        V_noid_ref_feature = self.encode_ref_noid(input_img)
        V_headpose_ref_feature = self.netE.to_headpose(V_noid_ref_feature)

        ref_merge_feature_a = self.select_frames(self.merge_mouthpose(A_mouth_feature, V_headpose_ref_feature))
        fake_image_ref_pose_a, _ = self.generate_fake(sel_id_feature, ref_merge_feature_a)
        if self.opt.driving_pose:
            V_noid_driving_feature = self.encode_noid_feature(driving_pose_frames)
            V_headpose_feature = self.netE.to_headpose(V_noid_driving_feature)
            driven_merge_feature_a = self.merge_mouthpose(A_mouth_feature, V_headpose_feature)
            sel_driven_pose_feature_a = self.select_frames(driven_merge_feature_a)
            fake_image_pose_driven_a, _ = self.generate_fake(sel_id_feature, sel_driven_pose_feature_a)

        return fake_image_ref_pose_a, fake_image_pose_driven_a

    def compute_generator_loss(self, input_img, real_image, augmented, spectrogram,
                               netD, labels, no_ganFeat_loss=False, no_vgg_loss=False, lambda_D=1):

        G_losses = {}

        real_image = real_image.view(-1, self.opt.output_nc, self.opt.crop_size, self.opt.crop_size)

        ##### ***************** encode image feature and generate ******************************

        V_noid_feature = self.encode_noid_feature(augmented)

        V_mouth_feature = self.netE.to_mouth(V_noid_feature)
        V_headpose_feature = self.netE.to_headpose(V_noid_feature)
        id_feature, id_scores = self.encode_identity_feature(input_img)

        sel_id_feature = []
        sel_id_feature.append(self.select_frames(id_feature[0]))
        sel_id_feature.append(self.select_frames(id_feature[1]))

        sel_real_image = self.select_frames(real_image)

        fake_image_A, fake_image_V = None, None

        if self.opt.generate_from_audio_only:
            assert self.opt.use_audio, 'must use audio in this case'

        V_merge_feature = self.merge_mouthpose(V_mouth_feature, V_headpose_feature)

        sel_V_merge_feature = self.select_frames(V_merge_feature)
        if self.opt.use_audio: # use audio pose feature

            A_mouth_feature = self.encode_audiosync_feature(spectrogram)
            self.compute_sync_loss(V_mouth_feature, A_mouth_feature, G_losses)

            A_merge_feature = self.merge_mouthpose(A_mouth_feature, V_headpose_feature)
            sel_A_merge_feature = self.select_frames(A_merge_feature)
            fake_image_A, style_rgb_a = self.generate_fake(sel_id_feature, sel_A_merge_feature)
            pred_fake_audio = self.discriminate_single(fake_image_A, netD)

            if not self.opt.generate_from_audio_only: # use both audio and image pose feature
                fake_image_V, style_rgb_v = self.generate_fake(sel_id_feature, sel_V_merge_feature)

        else: # only use image pose feature
            fake_image_V, style_rgb_v = self.generate_fake(sel_id_feature, sel_V_merge_feature)

        pred_real = self.discriminate_single(sel_real_image, netD)

        ##### ****************************************************************************

        if (not self.opt.generate_from_audio_only) or (not self.opt.use_audio):
            pred_fake = self.discriminate_single(fake_image_V, netD)

        if not no_ganFeat_loss:
            if not self.opt.generate_from_audio_only:
                G_losses['GAN_Feat'] = self.compute_GAN_Feat_loss(pred_fake, pred_real)
            if self.opt.use_audio:
                G_losses['GAN_Feat_audio'] = self.compute_GAN_Feat_loss(pred_fake_audio, pred_real)

        if not self.opt.fix_netD:
            if not self.opt.generate_from_audio_only:
                G_losses['GANv'] = self.criterionGAN(pred_fake, True,
                                                    for_discriminator=False) * lambda_D
            if self.opt.use_audio:
                G_losses['GANa'] = self.criterionGAN(pred_fake_audio, True,
                                                for_discriminator=False) * lambda_D

        if not no_vgg_loss:
            if not self.opt.generate_from_audio_only:
                G_losses['VGGv'] = self.criterionVGG(fake_image_V, sel_real_image) \
                        * self.opt.lambda_vgg
            if self.opt.use_audio:
                G_losses['VGGa'] = self.criterionVGG(fake_image_A, sel_real_image) \
                                  * self.opt.lambda_vgg

        if self.opt.vgg_face:
            if not self.opt.generate_from_audio_only:
                G_losses['VGGFace_v'] = self.criterionVGGFace(fake_image_V, sel_real_image, layer=2) \
                                  * self.opt.lambda_vggface

            if self.opt.use_audio:
                G_losses['VGGFace_a'] = self.criterionVGGFace(fake_image_A, sel_real_image, layer=2) \
                                      * self.opt.lambda_vggface


        if not self.opt.no_id_loss or not self.fix_netV:
            G_losses['loss_cls'] = self.loss_cls(id_scores, labels)

        if self.opt.disentangle and self.opt.clip_len*self.opt.frame_interval >= 20:
            V_headpose_embed = self.netE.headpose_embed(V_headpose_feature)
            with torch.no_grad():
                V_all_headpose_embed = V_headpose_embed.view(-1, self.opt.clip_len * V_headpose_embed.shape[-1])
                headpose_word_scores = self.netE.headpose_fc(V_all_headpose_embed)
            G_losses['logSoftmax_v'] = self.criterionLogSoftmax(headpose_word_scores) * self.opt.lambda_softmax

        return G_losses, [sel_real_image, fake_image_V, fake_image_A,
                          ], id_scores


    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def compute_GAN_Feat_loss(self, pred_fake, pred_real):
        num_D = len(pred_fake)
        GAN_Feat_loss = self.FloatTensor(1).fill_(0)
        for i in range(num_D):  # for each discriminator
            # last output is the final prediction, so we exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                if j == 0:
                    unweighted_loss *= self.opt.lambda_image
                GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
        return GAN_Feat_loss

    def compute_discriminator_loss(self, input_img, real_image, augmented, spectrogram, netD, labels, lambda_D=1):
        D_losses = {}
        with torch.no_grad():
            ##### ***************** encode feature and generate ******************************

            id_feature, _ = self.encode_identity_feature(input_img)
            sel_id_feature = []
            sel_id_feature.append(self.select_frames(id_feature[0]))
            sel_id_feature.append(self.select_frames(id_feature[1]))

            sel_real_image = self.select_frames(real_image)
            sel_input_img = self.select_frames(input_img)

            V_noid_feature = self.encode_noid_feature(augmented)
            V_noid_feature = V_noid_feature.detach()
            V_noid_feature.requires_grad_()

            V_mouth_feature = self.netE.to_mouth(V_noid_feature)
            V_headpose_feature = self.netE.to_headpose(V_noid_feature)

            fake_image_audio, fake_image = None, None

            if self.opt.generate_from_audio_only:
                assert self.opt.use_audio, 'must use audio in this case'

            if not self.opt.generate_from_audio_only:
                V_merge_feature = self.merge_mouthpose(V_mouth_feature, V_headpose_feature)

                sel_V_merge_feature = self.select_frames(V_merge_feature)
            if self.opt.use_audio:

                A_mouth_feature = self.encode_audiosync_feature(spectrogram)
                A_pose_feature = self.merge_mouthpose(A_mouth_feature, V_headpose_feature)
                sel_A_pose_feature = self.select_frames(A_pose_feature)
                fake_image_audio, style_rgb_a = self.generate_fake(sel_id_feature, sel_A_pose_feature)
                fake_image = fake_image_audio

                if not self.opt.generate_from_audio_only:  # use both audio and image pose feature
                    fake_image, style_rgb_v = self.generate_fake(sel_id_feature, sel_V_merge_feature)
                    fake_image = torch.cat([fake_image_audio, fake_image], 0)

            else:  # only use image pose feature
                fake_image, style_rgb_v = self.generate_fake(sel_id_feature, sel_V_merge_feature)

            sel_real_image = torch.cat([sel_real_image,]*(len(fake_image)//len(sel_real_image)), 0)
            sel_input_img = torch.cat([sel_input_img,]*(len(fake_image)//len(sel_input_img)), 0)

            if fake_image is not None:
                fake_image = fake_image.detach()
                fake_image.requires_grad_()
            if fake_image_audio is not None:
                fake_image_audio = fake_image_audio.detach()
                fake_image_audio.requires_grad_()

            if self.opt.disentangle:
                V_headpose_embed = self.netE.headpose_embed(V_headpose_feature)
                V_headpose_embed = V_headpose_embed.detach()
                V_headpose_embed.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            sel_input_img, fake_image, sel_real_image, netD)

        if self.opt.stylegan_D:
            pred_fake_styleGAN, pred_real_styleGAN = self.discriminate(
                sel_input_img, fake_image, sel_real_image, self.net_styleGAN_D)
            if type(pred_fake) == list and type(pred_real) == list:
                pred_fake.append(pred_fake_styleGAN)
                pred_real.append(pred_real_styleGAN)
            else:
                pred_fake = [pred_fake]
                pred_fake.append(pred_fake_styleGAN)
                pred_real = [pred_real]
                pred_real.append(pred_real_styleGAN)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True) * lambda_D

        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True) * lambda_D

        if self.opt.disentangle and self.opt.clip_len*self.opt.frame_interval >= 20:
            V_all_headpose_embed = V_headpose_embed.view(-1, self.opt.clip_len * V_headpose_embed.shape[-1])
            headpose_word_scores = self.netE.headpose_fc(V_all_headpose_embed)
            D_losses['headpose_feature_cls'] = self.loss_cls(headpose_word_scores, labels)

        return D_losses

    def discriminate(self, input, fake_image, real_image, netD):
        if self.opt.D_input == "concat":
            fake_concat = torch.cat([input, fake_image], dim=1)
            real_concat = torch.cat([input, real_image], dim=1)
        else:
            fake_concat = fake_image
            real_concat = real_image

        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    def discriminate_single(self, single_image, netD):

        if single_image.dim() == 5:
            single_image = single_image.view(-1, self.opt.output_nc, self.opt.crop_size, self.opt.crop_size)

        pred_single = netD(single_image)

        return pred_single

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            # rotate_fake = pred[pred.size(0) // 3: pred.size(0) * 2 // 3]
            real = pred[pred.size(0)//2 :]

        return fake, real

    def load_separately(self, network, network_label, opt):
        load_path = None
        if network_label == 'G':
            load_path = opt.G_pretrain_path
        elif network_label == 'D':

            load_path = opt.D_pretrain_path
        elif network_label == 'D_rotate':
            load_path = opt.D_rotate_pretrain_path
        elif network_label == 'E':
            load_path = opt.E_pretrain_path
        elif network_label == 'A':
            load_path = opt.A_pretrain_path
        elif network_label == 'A_sync':
            load_path = opt.A_sync_pretrain_path
        elif network_label == 'V':
            load_path = opt.V_pretrain_path

        if load_path is not None:
            if os.path.isfile(load_path):
                print("=> loading checkpoint '{}'".format(load_path))
                checkpoint = torch.load(load_path)
                util.copy_state_dict(checkpoint, network, strip='MobileNet', replace='model')
        else:
            print("no load_path")
        return network

    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            if not self.opt.train_recognition:
                print('%s not exists yet!' % save_path)
                if network_label == 'G':
                    raise ('Generator must exist!')
        else:
            # network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:

                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print(
                            'Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    not_initialized = set()

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])

                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

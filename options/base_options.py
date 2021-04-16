import sys
import argparse
import math
import os
from util import util
import torch
import models
import data
import pickle


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--name', type=str, default='demo', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--filename_tmpl', type=str, default='{:06}.jpg', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--data_path', type=str, default='/home/SENSETIME/zhouhang1/Downloads/VoxCeleb2/voxceleb2_train.csv', help='where to load voxceleb train data')
        parser.add_argument('--lrw_data_path', type=str,
                            default='/home/SENSETIME/zhouhang1/Downloads/VoxCeleb2/voxceleb2_train.csv',
                            help='where to load lrw train data')

        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids')
        parser.add_argument('--num_classes', type=int, default=5830, help='num classes')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--model', type=str, default='av', help='which model to use, rotate|rotatespade')
        parser.add_argument('--trainer', type=str, default='audio', help='which trainer to use, rotate|rotatespade')
        parser.add_argument('--norm_G', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--norm_A', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # input/output sizes
        parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
        parser.add_argument('--preprocess_mode', type=str, default='resize_and_crop', help='scaling and cropping of images at load time.', choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
        parser.add_argument('--crop_size', type=int, default=224, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        parser.add_argument('--crop_len', type=int, default=16, help='Crop len')
        parser.add_argument('--target_crop_len', type=int, default=0, help='Crop len')
        parser.add_argument('--crop', action='store_true', help='whether to crop the image')
        parser.add_argument('--clip_len', type=int, default=1, help='num of imgs to process')
        parser.add_argument('--pose_dim', type=int, default=12, help='num of imgs to process')
        parser.add_argument('--frame_interval', type=int, default=1, help='the interval of frams')
        parser.add_argument('--num_clips', type=int, default=1, help='num of clips to process')
        parser.add_argument('--num_inputs', type=int, default=1, help='num of inputs to the network')
        parser.add_argument('--feature_encoded_dim', type=int, default=2560, help='dim of reduced id feature')

        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--audio_nc', type=int, default=256, help='# of output audio channels')
        parser.add_argument('--frame_rate', type=int, default=25, help='fps')
        parser.add_argument('--num_frames_per_clip', type=int, default=5, help='num of frames one audio bin')
        parser.add_argument('--hop_size', type=int, default=160, help='audio hop size')
        parser.add_argument('--generate_interval', type=int, default=1, help='select frames to generate')
        parser.add_argument('--dis_feat_rec', action='store_true', help='select frames to generate')

        parser.add_argument('--train_recognition', action='store_true', help='train recognition only')
        parser.add_argument('--train_sync', action='store_true', help='train sync only')
        parser.add_argument('--train_word', action='store_true', help='train word only')
        parser.add_argument('--train_dis_pose', action='store_true', help='train dis pose')
        parser.add_argument('--generate_from_audio_only', action='store_true', help='if specified, generate only from audio features')
        parser.add_argument('--noise_pose', action='store_true', help='noise pose to generate a talking face')
        parser.add_argument('--style_feature_loss', action='store_true', help='style_feature_loss')

        # for setting inputsf
        parser.add_argument('--dataset_mode', type=str, default='voxtest')
        parser.add_argument('--landmark_align', action='store_true', help='wether there is landmark_align')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
        parser.add_argument('--n_mel_T', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--num_bins_per_frame', type=int, default=4, help='n_melT')

        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
        parser.add_argument('--use_audio', type=int, default=1, help='use audio as driven input')
        parser.add_argument('--use_audio_id', type=int, default=0, help='use audio id')
        parser.add_argument('--augment_target',  action='store_true', help='whether to use checkpoint')
        parser.add_argument('--verbose', action='store_true', help='just add')

        parser.add_argument('--display_winsize', type=int, default=224, help='display window size')

        # for generator
        parser.add_argument('--netG', type=str, default='modulate', help='selects model to use for netG (modulate)')
        parser.add_argument('--netA', type=str, default='resseaudio', help='selects model to use for netA (audio | spade)')
        parser.add_argument('--netA_sync', type=str, default='ressesync', help='selects model to use for netA (audio | spade)')
        parser.add_argument('--netV', type=str, default='resnext', help='selects model to use for netV (mobile | id)')
        parser.add_argument('--netE', type=str, default='fan', help='selects model to use for netV (mobile | fan)')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image|projection)')
        parser.add_argument('--D_input', type=str, default='single', help='(concat|single|hinge)')
        parser.add_argument('--driven_type', type=str, default='face', help='selects model to use for netV (heatmap | face)')
        parser.add_argument('--landmark_type', type=str, default='min', help='selects model to use for netV (mobile | fan)')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--feature_fusion', type=str, default='concat', help='style fusion method')
        parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

        # for instance-wise features
        parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')
        parser.add_argument('--input_id_feature', action='store_true', help='if specified, use id feature as style gan input')
        parser.add_argument('--load_landmark', action='store_true', help='if specified, load landmarks')
        parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        parser.add_argument('--style_dim', type=int, default=2580, help='# of encoder filters in the first conv layer')

        ####################### weight settings ###################################################################

        parser.add_argument('--vgg_face', action='store_true', help='if specified, use VGG feature matching loss')

        parser.add_argument('--VGGFace_pretrain_path', type=str, default='', help='VGGFace pretrain path')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_image', type=float, default=1.0, help='weight for image reconstruction')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--lambda_vggface', type=float, default=5.0, help='weight for vggface loss')
        parser.add_argument('--lambda_rotate_D', type=float, default='0.1',
                                 help='rotated D loss weight')
        parser.add_argument('--lambda_D', type=float, default=1,
                                 help='D loss weight')
        parser.add_argument('--lambda_softmax', type=float, default=1000000, help='weight for softmax loss')
        parser.add_argument('--lambda_crossmodal', type=float, default=1, help='weight for softmax loss')

        parser.add_argument('--lambda_contrastive', type=float, default=100, help='if specified, use contrastive loss for img and audio embed')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')

        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--no_id_loss', action='store_true', help='if specified, do *not* use cls loss')
        parser.add_argument('--word_loss', action='store_true', help='if specified, do *not* use cls loss')
        parser.add_argument('--no_spectrogram', action='store_true', help='if specified, do *not* use mel spectrogram, use mfcc')

        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')

        ############################## optimizer #############################
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')

        parser.add_argument('--no_gaussian_landmark', action='store_true', help='whether to use no_gaussian_landmark (1.0 landmark) for rotatespade model')
        parser.add_argument('--label_mask', action='store_true', help='whether to use face mask')
        parser.add_argument('--positional_encode', action='store_true', help='whether to use positional encode')
        parser.add_argument('--use_transformer', action='store_true', help='whether to use transformer')
        parser.add_argument('--has_mask', action='store_true', help='whether to use mask in transformer')
        parser.add_argument('--heatmap_size', type=float, default=3, help='the size of the heatmap, used in rotatespade model')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        # modify dataset-related parser options
        dataset_mode = opt.dataset_mode
        dataset_modes = opt.dataset_mode.split(',')

        if len(dataset_modes) == 1:
            dataset_option_setter = data.get_option_setter(dataset_mode)
            parser = dataset_option_setter(parser, self.isTrain)
        else:
            for dm in dataset_modes:
                dataset_option_setter = data.get_option_setter(dm)
                parser = dataset_option_setter(parser, self.isTrain)

        opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # lt options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)
        # Set semantic_nc based on the option.
        # This will be convenient in many places
        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])


        self.opt = opt
        return self.opt

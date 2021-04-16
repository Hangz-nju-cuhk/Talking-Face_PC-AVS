from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
        parser.add_argument('--tensorboard', default=True, help='if specified, use tensorboard logging. Requires tensorflow installed')
        parser.add_argument('--load_pretrain', type=str, default='',
                                 help='load the pretrained model from the specified location')

        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--recognition', action='store_true', help='train only recognition')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--noload_D', action='store_true', help='whether to load D when continue training')
        parser.add_argument('--pose_noise', action='store_true', help='whether to use pose noise training')
        parser.add_argument('--load_separately', action='store_true', help='whether to continue train by loading separate models')
        parser.add_argument('--niter', type=int, default=10, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=1000, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')

        parser.add_argument('--G_pretrain_path', type=str, default='./checkpoints/100_net_G.pth', help='G pretrain path')
        parser.add_argument('--D_pretrain_path', type=str, default='', help='D pretrain path')
        parser.add_argument('--E_pretrain_path', type=str, default='', help='E pretrain path')
        parser.add_argument('--V_pretrain_path', type=str, default='', help='V pretrain path')
        parser.add_argument('--A_pretrain_path', type=str, default='', help='E pretrain path')
        parser.add_argument('--A_sync_pretrain_path', type=str, default='', help='E pretrain path')
        parser.add_argument('--netE_pretrain_path', type=str, default='', help='E pretrain path')

        parser.add_argument('--fix_netV', action='store_true', help='if specified, fix net V')
        parser.add_argument('--fix_netE', action='store_true', help='if specified, fix net E')
        parser.add_argument('--fix_netE_mouth', action='store_true', help='if specified, fix net E mapper, fc and mapper')
        parser.add_argument('--fix_netE_mouth_embed', action='store_true', help='if specified, fix net E mapper, fc and mapper')
        parser.add_argument('--fix_netE_headpose', action='store_true', help='if specified, fix net E headpose')
        parser.add_argument('--fix_netA_sync', action='store_true', help='if specified fix net A_sync')
        parser.add_argument('--fix_netG', action='store_true', help='if specified, fix net G')
        parser.add_argument('--fix_netD', action='store_true', help='if specified, fix net D')
        parser.add_argument('--no_cross_modal', action='store_true', help='if specified, do *not* use cls loss')
        parser.add_argument('--softmax_contrastive', action='store_true', help='if specified, use contrastive loss for img and audio embed')
        # for discriminators

        parser.add_argument('--baseline_sync', action='store_true', help='train baseline sync')
        parser.add_argument('--style_feature_loss', action='store_true', help='to use style feature loss')
        # parser.add_argument('--vggface_checkpoint', type=str, default='', help='pth to vggface ckpt')
        parser.add_argument('--pretrain', action='store_true', help='Use outsider pretrain')
        parser.add_argument('--disentangle', action='store_true', help='whether to use disentangle loss')
        self.isTrain = True
        return parser

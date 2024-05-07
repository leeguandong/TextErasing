from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--display_freq', type=int, default=200, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
        self.parser.add_argument('--valid_freq', type=int, default=200, help='frequency of saving the latest results')

        # self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

        # self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        # self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        self.parser.add_argument('--n_epochs', type=int, default=100, help='训练的轮数')
        self.parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')

        # baseline是比较重要的参数，dann/pcd:捕获域间特征一致性，afn：跨域特征规范，
        # aa：应用对抗性数据增强，
        # focal/domain/rsd/afn/dann
        # baseline中虽然可以有很多，但实际上用的很少，基本都是null，偶尔有个dann，在optimize_parameters=stypn
        self.parser.add_argument('--baseline', type=str, default="null", help='null, ohem, focal')

        self.parser.add_argument('--lambda1', type=float, default=1.0, help='Lm的权重数，aplha1')
        self.parser.add_argument('--lambda2', type=float, default=5, help='Ln的权重数，alpha2')
        self.parser.add_argument('--diff_range', type=int, default=2.0, help='Lm的一个系数')

        # 说不清楚，应该就是lm/ln的一个容器
        self.parser.add_argument('--ctl_M', type=int, default=3, help='number of strategy')
        # self.parser.add_argument('--ctl_layer', type=int, default=1, help='the freq of update controller')
        self.parser.add_argument('--ctl_freq', type=int, default=5, help='控制器训练更新的频率')
        self.parser.add_argument('--ctl_train_freq', type=int, default=5, help='在model训练过程中，经过10iter才会变更一次元素的组合')
        # self.parser.add_argument('--ctl_ratio', type=float, default=0.5, help='the ratio of datasets to update controller')

        # 训练擦除模型时，前向推理出来的polices
        self.parser.add_argument('--ctl_policy_num', type=int, default=2, help='the ratio of datasets to update controller')

        self.parser.add_argument('--ctl_update_num', type=int, default=6,
                                 help='它除以ctl_bs就是控制器训练的轮数，就是reward三次，两个reward都是前向结果')
        self.parser.add_argument('--ctl_batchSize', type=int, default=2, help='训练controller的bs，和文字擦除训练的model的bs没关系')  # 更新controller的数据

        self.parser.add_argument('--reward_norm', type=str, default="exp", help='mean, norm，exp')  # reward的数值形式
        # self.parser.add_argument('--aux_reward', action='store_true', help='number of strategy')

        self.parser.add_argument('--aux_dataset', action='store_true', help='是否有辅助数据集,这个参数没有什么意义，后面也没有该类的继承')
        self.parser.add_argument('--continual_loss', action='store_true', help='增量学习，和对比学习一样')  #
        self.parser.add_argument('--cont_weight', type=int, default=500, help='增量学习的权重')

        self.parser.add_argument('--sigmoid', default=False, action='store_true', help='对文字擦除的判别器的结果进行sigmoid')

        self.parser.add_argument('--contrastive_loss', action='store_true', help='对比学习')

        # netD_M的lr，这个netD_M是controller1的realistic reward，其中beta1和beta2分别优化器的beta参数
        # gan生成器也是用这个参数
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        # 生成器的优化策略，linear/sp/step/plateau/cosine
        self.parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        # linear层需要n_epoch，n_epoch_decay
        # step 需要这个参数
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        # gan判别器的优化器的lr
        self.parser.add_argument('--dlr', type=float, default=0.00001, help='initial learning rate for adam')
        # ctroller的优化器
        self.parser.add_argument('--clr', type=float, default=0.0005, help='initial learning rate for adam')

        # gan模型的类型 wgan-权重裁剪版本/lsgan/gan/wgangp -梯度惩罚版本/vanilla
        self.parser.add_argument('--gan_mode', type=str, default='vanilla',
                                 help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')

        self.isTrain = True

import argparse
import os

import torch

from ceyin.utils import util_list


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # basic parameters
        # -------------------------------------------------------------------------------
        # readme中虽然推荐了online的方式，但其实还是ddp训练还是有问题的
        self.parser.add_argument('--online', action='store_true', default=False, help="online会调用ddp，否则多卡是dp")
        self.parser.add_argument('--saveOnline', action='store_true', default=False, help="save on dev or offline")

        self.parser.add_argument('--dataroot', type=str, default="/home/ivms/local_disk/poster-erase",
                                 help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--checkpoints_dir', type=str, default="./results", help='where the model saved')
        # self.parser.add_argument('--load_dir', default=None, help='where the model saved')
        self.parser.add_argument('--name', type=str, default="item-sp8-r24", help='where the model saved')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # model parameters
        # 监督学习：pix2pix、ensnet、mtrnet++、erasenet、scenetextrase
        # 自监督合成数据模式：dann、pcd、afn、aa
        self.parser.add_argument('--model', type=str, default='erasenet', help='chooses which model to use. pix2pix')

        self.parser.add_argument('--init_type', type=str, default='normal',
                                 help='network initialization [normal | xavier | kaiming | orthogonal],权重初始化')
        self.parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')

        self.parser.add_argument('--adg_start', default=True, action='store_false', help='若是加载模型，则为False')
        # 是controller的控制的判别器，用来控制realistic reward分支的，当选择24reward_type时会自动赋值，这里搞个占位符其实就可以
        self.parser.add_argument('--netD_M', default=False, action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--reward_type', type=str, default="24",
                                 help='reward的形式，difficult_reward/realistic_reward/realistic_val_reawrd，其中difficult_reward不用训练，realistic_reward需要训练')

        self.parser.add_argument('--mask_sigmoid', action='store_false', help='mask分支是否进行sigmoid')
        self.parser.add_argument('--PasteImage', default=False, action='store_true', help='the output are paste to the gt for loss calcu')
        # False和gt是一对
        # self.parser.add_argument('--PasteText', action='store_true', help='the output are paste to the gt for loss calcu')
        self.parser.add_argument('--valid', type=int, default=1,
                                 help='valid for evalutions')  # =1，就不验证了，valid=2,test image when training; valid=3: valid image
        self.parser.add_argument('--domain_in', default=False, action='store_true')  # 跨域对齐

        # dataset process
        self.parser.add_argument('--dataset_mode', type=str, default='items_adg', help='chooses how datasets are loaded.')
        self.parser.add_argument('--gen_space', type=str, default='random8',
                                 help='sepcific1/2/3  random1/2/3/4/5/6/7/8/9/10/11/12')
        # 在数据初始中都是range，在controller的推理训练中是sift
        # space_config，作者定义的搜索空间，搜索空间中的空间和controller是要一一对应的，此处是默认到

        self.parser.add_argument('--serial_batches', action='store_true',  # 就是shuffle，默认False，按照顺序
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading datasets')  # num_workers
        self.parser.add_argument('--batchSize', type=int, default=2, help='input batch size')

        # load_size和下面的preprocess是一对的，带resize的items模式下的都是默认的768,512，load_size失效的，
        # 在scale模式下，scale_width/scale_width_and_crop下，loas_size是起作用的
        # 在crop模式下，crop_size是起作用的
        self.parser.add_argument('--load_size', type=int, default=512, help='scale images to this size')
        self.parser.add_argument('--crop_size', type=int, default=512, help='then crop to this size')
        # self.parser.add_argument('--gen_method', type=str, default='art', help='art / basic/ copy / art_copy')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),  # 控制加载数据的数量的,比如说我只加载多少数据
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--preprocess', type=str, default='resize',
                                 help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        # resize_and_crop/scale_width_and_crop/resize/scale_width/crop/none
        # 数据增强了，内置了一个random.random()是否小于filp和rotate的值，filp=RandomHorizontalFlip(),rotate=RandomRotation()
        self.parser.add_argument('--flip', type=float, default=0.0, help='flip probability')
        self.parser.add_argument('--rotate', type=float, default=0.3, help='rotate probability')

        # 生成text mask的模式，按照pixel方式生成，还是按照rect方式生成
        self.parser.add_argument('--mask_mode', type=int, default=1, help='the type of mask, 0 for pixel; 1 for rect')
        # self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        # 生成mask时的的dilate系数，raw_mask是生成原图的mask，mask是text mask.
        self.parser.add_argument('--raw_mask_dilate', type=int, default=4, help='valid for evalutions')
        self.parser.add_argument('--mask_dilate', type=int, default=3, help='valid for evalutions')

        # additional parameters
        self.parser.add_argument('--seed', type=int, default=66, help="random seed")
        self.parser.add_argument('--verbose', action='store_true', default=True,
                                 help='if specified, print more debugging information')  # print model
        # debug信息的打印输出

        # 加载模型时的前缀，可能是轮数，可能是latest
        self.parser.add_argument('--which_epoch', type=str, default='best',
                                 help='which epoch to load? set to latest to use latest cached model')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        # 把 0,1 存到一个组中
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        self.opt.data_norm = True

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        if self.isTrain:
            args = vars(self.opt)

            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.model, self.opt.name)
            util_list.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt

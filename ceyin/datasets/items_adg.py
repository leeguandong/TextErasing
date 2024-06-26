import json
import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from ceyin.datasets.base_dataset import get_params, get_transform
from ceyin.utils.util_list import generate_valid_img, generate_img_with_config, random_gen_config
from ceyin.utils.util_list import get_mask, gen_raw_mask, masktotensor


class ItemsAdgDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.gen_tr = False
        self.valid = (opt.valid == 2 or opt.valid == 3)
        self.aux = opt.aux_dataset
        self.real_val = opt.real_val
        self.domain = opt.domain_in  #
        self.train = (opt.isTrain and not self.valid and not self.aux)
        if self.train:
            if self.real_val:
                self.dirs = ["ps_valid.txt"]
            else:
                self.dirs = ["train.txt"]
        elif opt.valid == 3:  # valid image
            self.dirs = ["ps_valid.txt"]
        elif opt.valid == 2:  # test image when training
            self.dirs = ["ps_test.txt"]
        elif not opt.isTrain:  # image used when test
            self.dirs = ["ps_test.txt"]
        else:  # easy image
            self.dirs = ["ps_valid.txt"]

        self.imageFiles = []
        self.labelFiles = []
        self.infos = []
        cnt = 0
        self.ps = 0
        for dir in self.dirs:
            dir = os.path.join(self.root, dir)
            with open(dir, "r", encoding='utf-8') as s:
                for line in s.readlines():
                    cnt += 1
                    #
                    if cnt > self.opt.max_dataset_size and opt.isTrain and not self.valid:
                        break
                    seq = line.rstrip().split('\t')
                    self.imageFiles.append(seq[1])
                    self.infos.append(seq[2])
                    if "ps" in dir:
                        self.ps = 1
                        self.labelFiles.append(seq[3])
            if cnt > self.opt.max_dataset_size and opt.isTrain and not self.valid:
                break

        self.backup_item = None
        self.img_size = len(self.imageFiles)
        # -----------------------------------------------------------------
        self.gen_configs = [random_gen_config(opt.gen_space), random_gen_config(opt.gen_space)]

        self.basic_configs = {
            "T": 0, "F": 10, "C": 0, "A": 0, "S": 1,
            "A1": 0, "A1_S": 0, "A1_C": 0,
            "A2": 0, "A2_R": 0, "A2_D": 0, "A2_C": 0,
        }
        self.mask_transform = transforms.Compose([
            # transforms.Resize(size=[704, 480], interpolation=Image.BICUBIC),
            transforms.Resize(size=[768, 512], interpolation=Image.BICUBIC),
            transforms.Lambda(lambda img: masktotensor(img)),
        ])

        self.input_transform = transforms.Compose([
            # transforms.Resize(size=[704, 480], interpolation=Image.BICUBIC),
            transforms.Resize(size=[768, 512], interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        self.norm_transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.toPIL = transforms.ToPILImage()
        if not opt.isTrain:
            self.back_transform = transforms.Compose([
                transforms.Resize(size=[750, 513], interpolation=Image.BICUBIC),
            ])

    def __getitem__(self, index):
        # item = self.load_item(index)
        try:
            item = self.load_item(index)
            self.backup_item = item
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            """Handling errors introduced by random mask generation step introduced in dataloader."""
            print('loading error: item ' + self.imageFiles[index])
            if self.backup_item is not None:
                item = self.backup_item
            else:
                item = self.__getitem__(index + 1)

        return item

    def load_item(self, index):
        gt_path = os.path.join(self.root, self.imageFiles[index])
        gt = Image.open(gt_path).convert('RGB')
        info = json.loads(self.infos[index])
        if self.train:
            if self.gen_tr:  # True
                # imgs = []
                right = max(1, int(len(self.gen_configs)))
                config = self.gen_configs[np.random.randint(0, right)]  # 在0-2中只取了一个
                # imgs = [generate_img_with_config(gt, info, config)]
                imgs = [generate_img_with_config(gt, info, config, False)]
                if self.domain:
                    basic_img = generate_img_with_config(gt, info, self.basic_configs, False)
            else:
                imgs = []
                for config in self.gen_configs:
                    imgs.append(generate_img_with_config(gt, info, config))
        else:
            if self.ps == 1:
                imgs = [gt]
                gt = Image.open(os.path.join(self.root, self.labelFiles[index])).convert('RGB')
            else:
                imgs = [generate_valid_img(gt, info, index, self.opt.gen_method)]

        # img = generate_img_m(gt, info["place"])
        # img = torch.ones(1)
        # raw_mask = torch.ones(1)
        raw_mask = gen_raw_mask(imgs[0].size, info["mask"], self.opt.raw_mask_dilate)
        if self.ps == 1:
            masks = [raw_mask]
        else:
            masks = []
            for im in imgs:
                masks.append(get_mask(gt, im, mode=self.opt.mask_mode, dilate=self.opt.mask_dilate))

        transforms_params = get_params(self.opt, imgs[0].size)
        trans = get_transform(self.opt, transforms_params, convert=False)

        if self.train:
            gt = trans(gt)
            raw_mask = trans(raw_mask)

        # ------------------------- 相当于在trans的基础上又做了一次 trans------------------
        gt = self.input_transform(gt)
        raw_mask = self.mask_transform(raw_mask)
        if self.opt.data_norm:
            gt = self.norm_transform(gt)

        imgs_tensor = torch.ones(len(imgs), gt.shape[0], gt.shape[1], gt.shape[2])
        masks_tensor = torch.ones(len(imgs), gt.shape[0], gt.shape[1], gt.shape[2])
        for j in range(len(imgs)):
            if self.train:
                img = trans(imgs[j])
                mask = trans(masks[j])
            else:
                img = imgs[j]
                mask = masks[j]
            img = self.input_transform(img)
            mask = self.mask_transform(mask)
            if self.opt.data_norm:
                img = self.norm_transform(img)
            imgs_tensor[j] = img
            masks_tensor[j] = mask

        if self.ps == 1:
            gt = gt * (1 - masks_tensor[0]) + imgs_tensor[0] * masks_tensor[0]

        if self.train and self.domain and self.gen_tr:
            basic_img = trans(basic_img)
            basic_img = self.input_transform(basic_img)
            return {'img': imgs_tensor, 'gt': gt, 'basic_img': basic_img, 'raw_mask': raw_mask, 'mask': masks_tensor, 'path': gt_path}

        return {'img': imgs_tensor, 'gt': gt, 'raw_mask': raw_mask, 'mask': masks_tensor, 'path': gt_path}

    def __len__(self):
        return self.img_size

    def name(self):
        return 'itemsADGDataset'

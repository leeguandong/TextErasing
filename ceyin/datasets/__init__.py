import torch.utils.data


def CreateDataLoader(opt, D=None, valid=False):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt, D, valid)
    return data_loader


def CreateDataset(opt):
    # dataset = None
    if opt.dataset_mode == 'items':
        from ceyin.datasets import ItemsDataset
        dataset = ItemsDataset()  # poster数据
    elif opt.dataset_mode == 'items_adg':
        from ceyin.datasets.items_adg import ItemsAdgDataset
        dataset = ItemsAdgDataset()  # poster+controller
    # elif opt.dataset_mode == 'items_adg_fea':
    #     from ceyin.datasets.items_adg_fea import ItemsAdgFeaDataset
    #     dataset = ItemsAdgFeaDataset()
    # elif opt.dataset_mode == 'ens':
    #     from ceyin.datasets import EnsDataset
    #     dataset = EnsDataset()
    # elif opt.dataset_mode == 'ens_adg':
    #     from ceyin.datasets.ens_adg import EnsAdgDataset
    #     dataset = EnsAdgDataset()
    # elif opt.dataset_mode == 'ens_adg_fea':
    #     from ceyin.datasets import EnsAdgFeaDataset
    #     dataset = EnsAdgFeaDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader():
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, D=None, valid=False):
        self.opt = opt
        self.dataset = CreateDataset(opt)
        if not opt.online or valid:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,  # 文字擦除模型的bs
                shuffle=not opt.serial_batches,  # shuffle
                num_workers=int(opt.nThreads))  # num_works
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset,
                num_replicas=D.get_world_size(),
                rank=D.get_rank(),
                shuffle=not opt.serial_batches)

            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                num_workers=int(opt.nThreads),
                sampler=train_sampler)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)

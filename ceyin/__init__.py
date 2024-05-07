def create_model(opt):
    # model = None
    # print(opt.model)
    # if opt.model == 'pix2pix' or opt.model == 'Pix2pix':
    #     from ceyin.models.pix2pix_model import Pix2PixModel
    #     model = Pix2PixModel()
    # elif opt.model == 'disc':
    #     from .disc_model import DiscModel
    #     model = DiscModel()
    # elif opt.model == 'erase_fix':
    #     from .erase_model_re import EraseModel
    #     model = EraseModel()       
    # elif opt.model == 'gateconv':
    #     from ceyin.models.gateconv_model import GatedConvModel
    #     model = GatedConvModel()
    if opt.model == 'erase' or opt.model == 'Erase' or opt.model == 'erasenet':
        from ceyin.models.erase_model import EraseModel
        model = EraseModel()   
    # elif opt.model == 'mtrnet':
    #     from .mtrnet_model import MTRNetModel
    #     model = MTRNetModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    # --------------------- 给模型传参数的initialize ---------------------------------------
    model.initialize(opt)
    # print("model [%s] was created" % (model.name()))
    return model

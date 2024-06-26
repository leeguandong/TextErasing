import torch
import torch.nn as nn
from torchvision import models
import os
from io import BytesIO


# from utils.oss import get_bucket

# VGG16 feature extract
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        # vgg16 = ceyin.vgg16(pretrained=True)
        # print(os.path.isfile('/home/gangwei.jgw/.cache/torch/hub/checkpoints/vgg16-397923af.pth'))
        # if os.path.isfile(r'E:\open_source_project\classification\Grad_CAM_Pytorch-1.01-Stephenfang51-patch-1\Grad_CAM_Pytorch-1.01-Stephenfang51-patch-1\vgg16-397923af.pth'):
        #     vgg16 = ceyin.vgg16(pretrained=True)
        # else:
        #     vgg16 = ceyin.vgg16(pretrained=False)
        #     path_to_load_model = "gangwei.jgw/pre_model/vgg16-397923af.pth"
        #     bucket = get_bucket(online=True)
        #     buffer = BytesIO(bucket.get_object(path_to_load_model).read())
        #     vgg16.load_state_dict(torch.load(buffer))
        vgg16 = models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load(
            '/home/ivms/net_disk_project/19045845/weights/vgg16-397923af.pth',
            map_location='cpu'))
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

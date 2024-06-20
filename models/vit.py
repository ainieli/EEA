import numpy as np
from timm.models import create_model

import torch
import torch.nn.functional as F

from transforms.RA import RandAugment as RA
from transforms import Normalize


class ViT(torch.nn.Module):

    def __init__(self, model_name, pretrained=False, init_c=3, num_classes=1000, dropout=0.0, droppath=0.1,
                 aug_depth=2,
                 resolution=224, augment_space='RA', p_min_t=0.2, p_max_t=0.8,
                 norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):
        super(ViT, self).__init__()
        assert 'deit' in model_name or 'swin' in model_name

        self.model = create_model(
            model_name,
            pretrained=pretrained,
            in_chans=init_c,
            num_classes=num_classes,
            drop_rate=dropout,
            drop_path_rate=droppath,
        )

        self.ra = RA(depth=aug_depth,
                       resolution=resolution, augment_space=augment_space,
                       p_min_t=p_min_t, p_max_t=p_max_t
                       )
        self.normalize = Normalize(norm_mean, norm_std)

    def net_forward(self, x):
        x = self.model(x)
        return x

    def forward(self, x, training=False, y=None, cos=None):
        assert y is None or cos is None
        if training:
            aug = None
            ori = None
            cos_sim = None
            if y is not None:
                ori = x
                ori = self.normalize(ori)
                ori = self.net_forward(ori)

                if y.shape[-1] != ori.shape[-1]:
                    y = torch.zeros(ori.shape).to(y.device).scatter_(-1, y[..., None], 1.0)
                cos_sim = F.cosine_similarity(F.softmax(ori.detach(), dim=-1), y)
            else:
                aug = x
                aug = self.ra(aug, cos)
                aug = self.normalize(aug)
                aug = self.net_forward(aug)
            return aug, ori, cos_sim
        else:
            x = self.normalize(x)
            x = self.net_forward(x)
            return x, None


def deit_ti(**kwargs):
    return ViT('deit_tiny_patch16_224', **kwargs)


def swin_ti(**kwargs):
    return ViT('swin_tiny_patch4_window7_224', **kwargs)


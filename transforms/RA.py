import collections
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from transforms.operations import ShearX, ShearY, TranslateX, TranslateY, Rotate, Brightness, Color, \
    Sharpness, Contrast, Solarize, Posterize, Equalize, AutoContrast, Identity

CANDIDATE_OPS_DICT_32 = CANDIDATE_OPS_DICT_224 = collections.OrderedDict({
    'ShearX': ShearX(-0.3, 0.3),
    'ShearY': ShearY(-0.3, 0.3),
    'TranslateX': TranslateX(-0.45, 0.45),
    'TranslateY': TranslateY(-0.45, 0.45),
    'Rotate': Rotate(-30, 30),
    'Brightness': Brightness(0.1, 1.9),
    'Color': Color(0.1, 1.9),
    'Sharpness': Sharpness(0.1, 1.9),
    'Contrast': Contrast(0.1, 1.9),
    'Solarize': Solarize(0, 256),
    'Posterize': Posterize(0, 4),
    'Equalize': Equalize(None, None),
    'AutoContrast': AutoContrast(None, None),
    'Identity': Identity(None, None),
})

RA_OP_NAME = CANDIDATE_OPS_DICT_32.keys()

RA_glioma = [
    'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
    'Brightness', 'Sharpness', 'Solarize', 'Posterize', 'Equalize', 'AutoContrast',
    'Identity'
]

RA_SPACE = {
    'RA': RA_OP_NAME,
    'RA-glioma': RA_glioma,
}


def one_hot(value, n_elements, axis=-1):
    one_h = torch.zeros(n_elements).to(value.device).scatter_(axis, value[..., None], 1.0)
    return one_h


def gumbel_softmax(logits, tau=1.0, hard=False, axis=-1):
    u = np.random.uniform(low=0., high=1., size=logits.shape)
    gumbel = torch.from_numpy(-np.log(-np.log(u))).to(logits.dtype).to(logits.device)
    gumbel = (logits + gumbel) / tau
    y_soft = F.softmax(gumbel, dim=-1)
    if hard:
        index = torch.argmax(y_soft, dim=axis).to(logits.device)
        y_hard = one_hot(index, y_soft.shape[axis], axis)
        ret = y_hard + y_soft - y_soft.detach()
    else:
        ret = y_soft
    return ret


class RandAugment(torch.nn.Module):

    def __init__(self, depth=2,
                 resolution=224, augment_space='RA', p_min_t=0.2, p_max_t=0.8):
        super(RandAugment, self).__init__()
        assert augment_space in RA_SPACE.keys()

        if resolution == 224:
            CANDIDATE_OPS_LIST = [CANDIDATE_OPS_DICT_224[op] for op in RA_SPACE[augment_space]]
        elif resolution == 32:
            CANDIDATE_OPS_LIST = [CANDIDATE_OPS_DICT_32[op] for op in RA_SPACE[augment_space]]
        else:
            print('Use 224 space')
            CANDIDATE_OPS_LIST = [CANDIDATE_OPS_DICT_224[op] for op in RA_SPACE[augment_space]]

        self.candidate_ops = CANDIDATE_OPS_LIST
        num_candidate_ops = len(self.candidate_ops)
        self.sp_magnitudes_mean = None

        self.depth = depth
        self.num_candidate_ops = num_candidate_ops
        self.p_min_t = p_min_t
        self.p_max_t = p_max_t

    def learnable_params(self):
        return []

    def _apply_op(self, x, prob, s_op, mag_mean):
        magnitude = mag_mean
        p = np.random.uniform(low=0., high=1.)
        if p < prob:
            x = self.candidate_ops[s_op](x, magnitude)
        return x

    def forward(self, x, ra_deform):
        # self.clip_value() # Shouldn't be used in the forward
        # to avoid learnable tensors being non-leaf
        assert len(x.shape) == 4  # x: (N, C, H, W), float32, 0-1
        assert ra_deform.shape[-1] == x.shape[0]
        if torch.max(x) <= 1:
            x = x * 255
        x = x.to(torch.float32)

        if len(ra_deform.shape) == 1:
            ra_deform_list = [ra_deform] * self.depth  # "depth" layers share the same magnitude for each sample
            try:
                ra_deform = torch.stack(ra_deform_list, dim=0)  # (depth, N)
            except:
                pass    # depth=0
        self.sp_magnitudes_mean = ra_deform

        x_batch = []
        for n in range(x.shape[0]):
            x_temp = x[n][None]
            for i in range(self.depth):
                j = torch.randint(low=0, high=self.num_candidate_ops, size=())
                p = np.random.uniform(low=self.p_min_t, high=self.p_max_t)
                mag = self.sp_magnitudes_mean[i, n]
                x_temp = self._apply_op(x_temp, p, j, mag)

            x_batch.append(x_temp)
        x = torch.cat(x_batch, dim=0)

        x = torch.clamp(x / 255., 0., 1.)
        return x

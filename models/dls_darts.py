from collections import namedtuple

import torch
from torch.nn import Sequential, BatchNorm2d, Module, Conv2d, Linear, ReLU, AvgPool2d, MaxPool2d, Identity, ModuleList
import torch.nn.functional as F

from transforms.RA import RandAugment as RA
from transforms import Normalize


Genotype = namedtuple('Genotype', 'fused_head fl_head normal normal_concat reduce reduce_concat')

PRIMITIVES_dlsdarts = [
    'none',
    'skip_connect',
    'max_pool_3x3',
    'avg_pool_3x3',
    'nor_conv_1x1',
    'nor_conv_3x3',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
]

CONFIG = {
    'primitives': PRIMITIVES_dlsdarts,
}

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride: AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride: MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride: Identity() if stride == 1 else FactorizedReduce(C, C),
    'sep_conv_3x3': lambda C, stride: SepConv(C, C, 3, stride, 1),
    'sep_conv_5x5': lambda C, stride: SepConv(C, C, 5, stride, 2),
    # 'sep_conv_7x7': lambda C, stride: SepConv(C, C, 7, stride, 3),
    'dil_conv_3x3': lambda C, stride: DilConv(C, C, 3, stride, 2, 2),
    'dil_conv_5x5': lambda C, stride: DilConv(C, C, 5, stride, 4, 2),
    'nor_conv_1x1': lambda C, stride: ReLUConvBN(C, C, 1, stride, 0),
    'nor_conv_3x3': lambda C, stride: ReLUConvBN(C, C, 3, stride, 1),
}

# Genotype in DLS-DARTS
DEFAULT_GENO = Genotype(
        fused_head=[
            ('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 2),
            ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 3),
        ],
        fl_head=[
            ('nor_conv_3x3', 0), ('avg_pool_3x3', 1),
            ('max_pool_3x3', 0), ('avg_pool_3x3', 1),
        ],
        normal=[
            ('skip_connect', 0), ('nor_conv_3x3', 1),
            ('sep_conv_3x3', 1), ('dil_conv_3x3', 2),
            ('nor_conv_3x3', 0), ('skip_connect', 1),
        ], normal_concat=[2, 3, 4],
        reduce=[
            ('nor_conv_3x3', 0), ('sep_conv_5x5', 1),
            ('sep_conv_5x5', 0), ('skip_connect', 2),
            ('skip_connect', 0), ('dil_conv_5x5', 3),
        ], reduce_concat=[2, 3, 4],
    )


def get_primitives():
    return CONFIG['primitives']


class ReLUConvBN(Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = Sequential(
            ReLU(inplace=False),
            Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            BatchNorm2d(C_out, affine=affine)
        ).cuda()

    def forward(self, x):
        return self.op(x)


class DilConv(Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = Sequential(
            ReLU(inplace=False),
            Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            BatchNorm2d(C_out, affine=affine),
        ).cuda()

    def forward(self, x):
        return self.op(x)


class SepConv(Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = Sequential(
            ReLU(inplace=False),
            Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            BatchNorm2d(C_in, affine=affine),
            ReLU(inplace=False),
            Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            BatchNorm2d(C_out, affine=affine),
        ).cuda()

    def forward(self, x):
        return self.op(x)


class Zero(Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = ReLU(inplace=False)
        self.conv_1 = Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False).cuda()
        self.conv_2 = Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False).cuda()
        self.bn = BatchNorm2d(C_out, affine=affine).cuda()

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class DropPath(Module):
    def __init__(self, p=0.2):
        super(DropPath, self).__init__()
        assert 0 <= p <= 1, "drop probability has to be between 0 and 1, but got %f" % p
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep_prob = 1.0 - self.p
        batch_size = x.size(0)
        t = torch.rand(batch_size, 1, 1, 1, dtype=x.dtype, device=x.device) > keep_prob
        x = (x / keep_prob).masked_fill(t, 0)
        return x


class GlobalAvgPool(Module):

    def __init__(self, keep_dim=False):
        super(GlobalAvgPool, self).__init__()
        self.keep_dim = keep_dim

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, 1)
        if not self.keep_dim:
            x = x.view(x.size(0), -1)
        return x


class SelectedOP(Module):

    def __init__(self, C, stride, droppath, op_name):
        super(SelectedOP, self).__init__()
        self.stride = stride
        primitives = get_primitives()
        op_idx = primitives.index(op_name)
        primitive = primitives[op_idx]
        if droppath:
            op = Sequential(
                OPS[primitive](C, stride),
            )
            if 'pool' in primitive:
                op.add_module('bn', BatchNorm2d(C))
            op.add_module('droppath', DropPath(droppath))
        else:
            op = OPS[primitive](C, stride)
            if 'pool' in primitive:
                op = Sequential(
                    op,
                    BatchNorm2d(C)
                )
        self._ops = op.cuda()

    def forward(self, x):
        return self._ops(x)


class Cell(Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, droppath, geno):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = ModuleList()
        self._pre_node_idx = []
        start = 0
        for i in range(self._steps):
            end = start + 2
            geno_i = geno[start:end]  # tuples
            for j in range(2):
                op_name, pre_node_idx = geno_i[j]
                stride = 2 if reduction and pre_node_idx < 2 else 1
                op = SelectedOP(C, stride, droppath, op_name)
                self._ops.append(op)
                self._pre_node_idx.append(pre_node_idx)
            start = end

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum([self._ops[offset + j](states[self._pre_node_idx[offset + j]]) for j in range(2)])
            offset += 2
            states.append(s)
        return torch.cat(states[-self._multiplier:], axis=1)


# For WL, NIR-I and NIR-II
class FusedHeadCell(Module):

    def __init__(self, step, multiplier, stem_C, C, droppath, geno):
        super(FusedHeadCell, self).__init__()
        self.preprocess0 = ReLUConvBN(stem_C, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(stem_C, C, 1, 1, 0)
        self.preprocess2 = ReLUConvBN(stem_C, C, 1, 1, 0)
        self.step = step
        self.multiplier = multiplier

        self._ops = ModuleList()
        self._pre_node_idx = []
        start = 0
        for i in range(self.step):
            end = start + 3
            geno_i = geno[start:end]  # tuples
            for j in range(3):
                op_name, pre_node_idx = geno_i[j]
                stride = 1
                op = SelectedOP(C, stride, droppath, op_name)
                self._ops.append(op)
                self._pre_node_idx.append(pre_node_idx)
            start = end
        self.channel_reduction = ReLUConvBN(C * (3 + self.step), C * self.multiplier, 1, 1, 0)

    def forward(self, wl, niri, nirii):
        s0 = self.preprocess0(wl)
        s1 = self.preprocess1(niri)
        s2 = self.preprocess2(nirii)

        states = [s0, s1, s2]
        offset = 0
        for i in range(self.step):
            s = sum([self._ops[offset + j](states[self._pre_node_idx[offset + j]]) for j in range(3)])
            offset += 3
            states.append(s)
        return self.channel_reduction(torch.cat(states, axis=1))


# For NIR-I and NIR-II
class FLHeadCell(Module):

    def __init__(self, step, multiplier, stem_C, C, droppath, geno):
        super(FLHeadCell, self).__init__()
        self.preprocess0 = ReLUConvBN(stem_C, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(stem_C, C, 1, 1, 0)
        self.step = step
        self.multiplier = multiplier

        self._ops = ModuleList()
        self._pre_node_idx = []
        start = 0
        for i in range(self.step):
            end = start + 2
            geno_i = geno[start:end]    # tuples
            for j in range(2):
                op_name, pre_node_idx = geno_i[j]
                stride = 1
                op = SelectedOP(C, stride, droppath, op_name)
                self._ops.append(op)
                self._pre_node_idx.append(pre_node_idx)
            start = end

    def forward(self, niri, nirii):
        s0 = self.preprocess0(niri)
        s1 = self.preprocess1(nirii)

        states = [s0, s1]
        offset = 0
        for i in range(self.step):
            s = sum([self._ops[offset + j](states[self._pre_node_idx[offset + j]]) for j in range(2)])
            offset += 2
            states.append(s)
        return torch.cat(states[-self.multiplier:], axis=1)


class DLS_DARTS(Module):

    def __init__(self, C, layers=6, steps=3, multiplier=3, stem_multiplier=4, droppath=0., num_classes=2,
                 geno: Genotype=None):
        super(DLS_DARTS, self).__init__()
        self._C = C
        self._steps = steps
        self._multiplier = multiplier
        self.droppath = droppath
        self.geno = geno

        # layers=6: (HH)RNRNRN
        if layers == 6:
            reduction_id = [0, 2, 4]
        # layers=8: (HH)RNRNRNRN
        elif layers == 8:
            reduction_id = [0, 2, 4, 6]
        else:
            raise NotImplementedError("Model not defined!")
        C_curr = stem_multiplier * C

        # hyper-parameters to control head cell
        self.fused_head_step = 2
        self.fused_head_multiplier = 4
        self.fused_head_C = C_curr // 2
        self.fl_head_step = 2
        self.fl_head_multiplier = 4
        self.fl_head_C = C_curr // 2

        self.stem_wl = Sequential(
            Conv2d(3, C_curr, 3, stride=1, padding=1, bias=False),
            BatchNorm2d(C_curr),
        )  # out channel: C_curr
        self.stem_niri = Sequential(
            Conv2d(3, C_curr, 3, stride=1, padding=1, bias=False),
            BatchNorm2d(C_curr),
        )  # out channel: C_curr
        self.stem_nirii = Sequential(
            Conv2d(3, C_curr, 3, stride=1, padding=1, bias=False),
            BatchNorm2d(C_curr),
        )  # out channel: C_curr

        self.fused_head = FusedHeadCell(
            self.fused_head_step, self.fused_head_multiplier, stem_C=C_curr, C=self.fused_head_C, droppath=droppath,
            geno=geno.fused_head
        )   # out channel: self.fused_head_C * self.fused_head_multiplier
        self.fl_head = FLHeadCell(
            self.fl_head_step, self.fl_head_multiplier, stem_C=C_curr, C=self.fl_head_C, droppath=droppath,
            geno=geno.fl_head
        )   # out channel: self.fl_head_C * self.fused_head_multiplier

        C_prev_prev, C_prev, C_curr = self.fused_head_C * self.fused_head_multiplier, self.fl_head_C * self.fused_head_multiplier, C
        self.cells = ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in reduction_id:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            if reduction:
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, droppath,
                            geno=geno.reduce)
            else:
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, droppath,
                            geno=geno.normal)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.avg_pool = GlobalAvgPool()
        self.classifier = Linear(C_prev, num_classes)

    def forward(self, x):      # x has 9 channels
        x = x.float()
        input_C = x.shape[1]
        if input_C == 9:
            wl = x[:, :3, :, :]
            niri = x[:, 3:6, :, :]
            nirii = x[:, 6:, :, :]
        else:
            wl = niri = nirii = x

        wl = self.stem_wl(wl)
        niri = self.stem_niri(niri)
        nirii = self.stem_nirii(nirii)

        s0 = self.fused_head(wl, niri, nirii)
        s1 = self.fl_head(niri, nirii)

        for cnt, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)

        x = self.avg_pool(s1)
        logits = self.classifier(x)
        return logits


class DLS_DARTS_w_Aug(torch.nn.Module):

    def __init__(self, init_c=36, layers=6, num_classes=2, droppath=0.1,
                 aug_depth=2,
                 resolution=224, augment_space='RA', p_min_t=0.2, p_max_t=0.8,
                 norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):
        super(DLS_DARTS_w_Aug, self).__init__()

        self.model = DLS_DARTS(C=init_c, layers=layers, droppath=droppath, num_classes=num_classes, geno=DEFAULT_GENO)

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


def dls_darts_trans(**kwargs):
    return DLS_DARTS_w_Aug(layers=6, **kwargs)


import numpy as np
import torch
from transforms.SRA import CANDIDATE_OPS_DICT_32, CANDIDATE_OPS_DICT_224
from transforms.operations import Invert

POLICY_LIST_224 = [
            (0.4, "Posterize", 8, 0.6, "Rotate", 9),
            (0.6, "Solarize", 5, 0.6, "AutoContrast", 5),
            (0.8, "Equalize", 8, 0.6, "Equalize", 3),
            (0.6, "Posterize", 7, 0.6, "Posterize", 6),
            (0.4, "Equalize", 7, 0.2, "Solarize", 4),

            (0.4, "Equalize", 4, 0.8, "Rotate", 8),
            (0.6, "Solarize", 3, 0.6, "Equalize", 7),
            (0.8, "Posterize", 5, 1.0, "Equalize", 2),
            (0.2, "Rotate", 3, 0.6, "Solarize", 8),
            (0.6, "Equalize", 8, 0.4, "Posterize", 6),

            (0.8, "Rotate", 8, 0.4, "Color", 0),
            (0.4, "Rotate", 9, 0.6, "Equalize", 2),
            (0.0, "Equalize", 7, 0.8, "Equalize", 8),
            (0.6, "Invert", 4, 1.0, "Equalize", 8),
            (0.6, "Color", 4, 1.0, "Contrast", 8),

            (0.8, "Rotate", 8, 1.0, "Color", 2),
            (0.8, "Color", 8, 0.8, "Solarize", 7),
            (0.4, "Sharpness", 7, 0.6, "Invert", 8),
            (0.6, "ShearX", 5, 1.0, "Equalize", 9),
            (0.4, "Color", 0, 0.6, "Equalize", 3),

            (0.4, "Equalize", 7, 0.2, "Solarize", 4),
            (0.6, "Solarize", 5, 0.6, "AutoContrast", 5),
            (0.6, "Invert", 4, 1.0, "Equalize", 8),
            (0.6, "Color", 4, 1.0, "Contrast", 8),
            (0.8, "Equalize", 8, 0.6, "Equalize", 3)
]

POLICY_LIST_32 = [
            (0.1, "Invert", 7, 0.2, "Contrast", 6),
            (0.7, "Rotate", 2, 0.3, "TranslateX", 9),
            (0.8, "Sharpness", 1, 0.9, "Sharpness", 3),
            (0.5, "ShearY", 8, 0.7, "TranslateY", 9),
            (0.5, "AutoContrast", 8, 0.9, "Equalize", 2),

            (0.2, "ShearY", 7, 0.3, "Posterize", 7),
            (0.4, "Color", 3, 0.6, "Brightness", 7),
            (0.3, "Sharpness", 9, 0.7, "Brightness", 9),
            (0.6, "Equalize", 5, 0.5, "Equalize", 1),
            (0.6, "Contrast", 7, 0.6, "Sharpness", 5),

            (0.7, "Color", 7, 0.5, "TranslateX", 8),
            (0.3, "Equalize", 7, 0.4, "AutoContrast", 8),
            (0.4, "TranslateY", 3, 0.2, "Sharpness", 6),
            (0.9, "Brightness", 6, 0.2, "Color", 8),
            (0.5, "Solarize", 2, 0.0, "Invert", 3),

            (0.2, "Equalize", 0, 0.6, "AutoContrast", 0),
            (0.2, "Equalize", 8, 0.6, "Equalize", 4),
            (0.9, "Color", 9, 0.6, "Equalize", 6),
            (0.8, "AutoContrast", 4, 0.2, "Solarize", 8),
            (0.1, "Brightness", 3, 0.7, "Color", 0),

            (0.4, "Solarize", 5, 0.9, "AutoContrast", 3),
            (0.9, "TranslateY", 9, 0.7, "TranslateY", 9),
            (0.9, "AutoContrast", 2, 0.8, "Solarize", 3),
            (0.8, "Equalize", 8, 0.1, "Invert", 3),
            (0.7, "TranslateY", 9, 0.9, "AutoContrast", 1)
]


class AutoAugment(torch.nn.Module):

    def __init__(self, resolution=224):
        super(AutoAugment, self).__init__()
        if resolution == 224:
            self.policy_list = POLICY_LIST_224
            self.candidate_ops = CANDIDATE_OPS_DICT_224
        elif resolution == 32:
            self.policy_list = POLICY_LIST_32
            self.candidate_ops = CANDIDATE_OPS_DICT_32
        else:
            print('Use 224 space')
            self.policy_list = POLICY_LIST_224
            self.candidate_ops = CANDIDATE_OPS_DICT_224

        # Add extra ops in AA
        self.candidate_ops['Invert'] = Invert(None, None)

        self.num_candidate_policy = len(self.policy_list)


    def _apply_op(self, x, prob, op_name, mag_level):
        mag = mag_level / 10.
        p = np.random.uniform(low=0., high=1.)
        if p < prob:
            x = self.candidate_ops[op_name](x, mag)
        return x

    def forward(self, x):
        assert len(x.shape) == 4  # x: (N, C, H, W), float32, 0-1
        if torch.max(x) <= 1:
            x = x * 255
        x = x.to(torch.float32)

        x_batch = []
        for n in range(x.shape[0]):
            x_temp = x[n][None]

            j = torch.randint(low=0, high=self.num_candidate_policy, size=())

            p1, o1, m1, p2, o2, m2 = self.policy_list[j]
            x_temp = self._apply_op(x_temp, p1, o1, m1)
            x_temp = self._apply_op(x_temp, p2, o2, m2)

            x_batch.append(x_temp)
        x = torch.cat(x_batch, dim=0)

        x = torch.clamp(x / 255., 0., 1.)
        return x
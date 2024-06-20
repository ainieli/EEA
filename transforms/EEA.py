import collections
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from transforms.SRA import one_hot
from transforms.SRA import CANDIDATE_OPS_DICT_32, CANDIDATE_OPS_DICT_224, RA_SPACE


class ExploreExploitAugment(torch.nn.Module):

    def __init__(self, depth=2, epsilon=0.0, learning_rate=0.01, reward_decay=0.98, num_bins=11,
                 resolution=224, augment_space='RA', p_min_t=1.0, p_max_t=1.0):
        super(ExploreExploitAugment, self).__init__()
        assert augment_space in RA_SPACE.keys()
        assert isinstance(num_bins, int) and num_bins > 0

        if resolution == 224:
            CANDIDATE_OPS_LIST = [CANDIDATE_OPS_DICT_224[op] for op in RA_SPACE[augment_space]]
        elif resolution == 32:
            CANDIDATE_OPS_LIST = [CANDIDATE_OPS_DICT_32[op] for op in RA_SPACE[augment_space]]
        else:
            print('Use 224 space')
            CANDIDATE_OPS_LIST = [CANDIDATE_OPS_DICT_224[op] for op in RA_SPACE[augment_space]]

        self.candidate_ops = CANDIDATE_OPS_LIST
        num_candidate_ops = len(self.candidate_ops)
        # self.Q_table = torch.ones(depth, num_bins) / num_bins    # (depth, bins)
        self.Q_table = torch.zeros(depth, num_bins)  # (depth, bins)
        self.reward = torch.zeros(depth, num_bins)  # (depth, bins)
        self.bin_cnts = torch.zeros(depth, num_bins)  # (depth, bins)
        self.sp_magnitudes = None  # (depth, N), N is recorded after forward

        self.epsilon = epsilon
        self.learning_rate = learning_rate  # alpha in Q table update
        self.reward_decay = reward_decay  # gamma in Q table update
        self.depth = depth
        self.num_bins = num_bins

        self.num_candidate_ops = num_candidate_ops
        self.p_min_t = p_min_t
        self.p_max_t = p_max_t

        self.r_w = torch.ones(depth, num_bins)
        self.Qmax = torch.zeros(1)

    # def learnable_params(self):
    #     return []

    def _cal_bin_cnts(self, batch_loss):
        bin_cnts = torch.zeros(self.depth, self.num_bins).to(batch_loss.device)
        for i in range(self.depth):
            bin_cnts[i] = torch.tensor(
                [torch.sum(torch.where(self.sp_magnitudes[i] == x, 1, 0)) for x in range(self.num_bins)])
        return bin_cnts
        # print(self.bin_cnts)

    def _cal_batch_avg_pred(self, batch_loss):
        return torch.exp(-torch.mean(batch_loss))

    def _cal_mag_avg_pred(self, batch_loss):
        #loss_sum = torch.zeros(self.depth, self.num_bins).to(batch_loss.device)  # (depth, bins)
        #for i, loss in enumerate(batch_loss):
        #    loss_sum[range(self.depth), self.sp_magnitudes[:, i]] += loss
        #bin_cnts = self._cal_bin_cnts(batch_loss)
        #mag_avg_pred = torch.exp(-loss_sum / bin_cnts)
        #return mag_avg_pred
        pred_sum = torch.zeros(self.depth, self.num_bins).to(batch_loss.device)  # (depth, bins)
        batch_pred = torch.exp(-batch_loss)
        for i, pred in enumerate(batch_pred):
            pred_sum[range(self.depth), self.sp_magnitudes[:, i]] += pred
        bin_cnts = self._cal_bin_cnts(batch_loss)
        mag_avg_pred = pred_sum / bin_cnts
        return mag_avg_pred

    def _apply_op(self, x, prob, s_op, mag):
        p = np.random.uniform(low=0., high=1.)
        if p < prob:
            x = self.candidate_ops[s_op](x, mag)
        return x

    def cal_bin_cnts(self, batch_loss):
        batch_loss = batch_loss.detach()
        self.bin_cnts = self._cal_bin_cnts(batch_loss)

    def cal_reward(self, batch_loss):
        batch_loss = batch_loss.detach()
        self.reward = self._cal_mag_avg_pred(batch_loss)
        self.reward = torch.stack(
            [torch.where(torch.isnan(self.reward[x]), torch.full_like(self.reward[x], 0), self.reward[x]) for x in
             range(self.depth)], dim=0)

    def cal_Qmax(self, batch_loss):
        batch_loss = batch_loss.detach()
        self.Qmax = self._cal_batch_avg_pred(batch_loss).to(batch_loss.device)
        # self.Qmax = torch.max(self.Q_table, dim=-1, keepdim=True)[0].to(batch_loss.device)

    def cal_reward_weights(self, batch_loss):
        batch_loss = batch_loss.detach()
        batch_avg_pred = self._cal_batch_avg_pred(batch_loss)
        mag_avg_pred = self._cal_mag_avg_pred(batch_loss)
        mag_avg_pred = torch.stack(
            [torch.where(torch.isnan(mag_avg_pred[x]), torch.full_like(mag_avg_pred[x], float(batch_avg_pred)),
                         mag_avg_pred[x]) for x in
             range(self.depth)], dim=0)
        self.r_w = batch_avg_pred / mag_avg_pred
        # self.r_w = -torch.log(mag_avg_pred)

    def update_Q_table(self, batch_loss):
        batch_loss = batch_loss.detach()
        # batch_loss: (N,)
        # mag: (depth, N)
        # bin_cnts: (depth, bins)

        # Bellman Equation: Q(a) ← Q(a) + alpha * (reward + gamma * max Q(a) - Q(a))
        self.Q_table = self.Q_table.to(batch_loss.device)
        self.r_w = self.r_w.to(batch_loss.device)
        # self.Q_table = self.Q_table + self.learning_rate * (self.r_w * self.reward + self.reward_decay * torch.max(self.Q_table, dim=-1)[0][:, None].repeat(1, self.num_bins) - self.Q_table)
        self.Q_table = self.Q_table + self.learning_rate * (
                    self.r_w * self.reward + self.reward_decay * self.Qmax - self.Q_table)

    # def _normalize_Q_table(self):
    #     if torch.sum(self.Q_table) != self.depth:
    #         self.Q_table = F.softmax(self.Q_table, dim=-1)

    def forward(self, x, use_Q_table=False):
        assert len(x.shape) == 4  # x: (N, C, H, W), float32, 0-1
        if torch.max(x) <= 1:
            x = x * 255
        x = x.to(torch.float32)
        N = x.shape[0]
        sp_magnitudes = torch.zeros(self.depth, N)

        x_batch = []
        # self._normalize_weights()
        for n in range(N):
            x_temp = x[n][None]
            for i in range(self.depth):
                value = torch.randint(low=0, high=self.num_candidate_ops, size=())
                hardwts = one_hot(value, self.num_candidate_ops)

                p = np.random.uniform(low=self.p_min_t, high=self.p_max_t)  # probability to apply the augmentation

                j = torch.argmax(hardwts)
                if self.num_bins > 1:
                    if not use_Q_table:
                        mag = torch.randint(low=0, high=self.num_bins, size=())
                        # print(mag)
                    else:
                        p = torch.zeros(1).uniform_()
                        # p = np.random.uniform()
                        if p < self.epsilon:
                            mag = torch.randint(low=0, high=self.num_bins, size=())
                        else:
                            mag = torch.argmax(self.Q_table[i])
                        # print(p, mag)

                    sp_magnitudes[i, n] = mag
                    x_temp = self._apply_op(x_temp, p, j, mag / (self.num_bins - 1))
                else:  # default: mag=1.0
                    mag = 1
                    sp_magnitudes[i, n] = mag
                    x_temp = self._apply_op(x_temp, p, j, mag)

            x_batch.append(x_temp)
        x = torch.cat(x_batch, dim=0)

        x = torch.clamp(x / 255., 0., 1.)
        self.sp_magnitudes = sp_magnitudes.long()
        return x

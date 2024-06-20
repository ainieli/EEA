import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=-1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1. / batch_size))
    if len(res) == 1:
        return res[0]
    return res


def _logits_to_prob(output):
    return F.softmax(output, dim=-1)


def _get_cm(output, target, logits=True):
    # Output shape: (N, 2), Target shape: (N,)
    if logits:
        output = _logits_to_prob(output)
    output[output > 0.5] = 1
    output[output < 1] = 0
    confusion = confusion_matrix(target.cpu().numpy(), output[:, 1].cpu().numpy())
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    return TP, TN, FP, FN


def sensitivity(output, target, logits=True):
    # Output shape: (N, 2), Target shape: (N,)
    if logits:
        output = _logits_to_prob(output)
    TP, TN, FP, FN = _get_cm(output, target)
    return TP / float(TP + FN)


def specificity(output, target, logits=True):
    # Output shape: (N, 2), Target shape: (N,)
    if logits:
        output = _logits_to_prob(output)
    TP, TN, FP, FN = _get_cm(output, target)
    return TN / float(TN + FP)


def f1(output, target, logits=True):
    if logits:
        output = _logits_to_prob(output)
    output[output > 0.5] = 1
    output[output < 1] = 0
    f1 = f1_score(target.cpu().numpy(), output[:, 1].cpu().numpy())
    return f1


def auc(output, target, logits=True):
    # Output shape: (N, 2), Target shape: (N,)
    if logits:
        output = _logits_to_prob(output)
    auc = roc_auc_score(target.cpu().numpy(), output[:, 1].cpu().numpy())
    return auc


def equal_to(x1, x2, eta=1e-9):
    return x1 > x2 - eta and x1 < x2 + eta


if __name__ == "__main__":
    import numpy as np
    import torch
    pred = np.array([[0.1, 0.3, 0.6],
            [0.4, 0.45, 0.15],
            [0.2, 0.6, 0.1]])
    label = np.array([0, 1, 2])
    pred = torch.from_numpy(pred)
    label = torch.from_numpy(label)
    acc = accuracy(pred, label)
    print(acc)
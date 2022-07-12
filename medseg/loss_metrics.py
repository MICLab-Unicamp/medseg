import sys
import torch
from torch import nn
from torch import Tensor


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lower_is_best = True
        assert(hasattr(self, "name"))


class DICELoss(Loss):
    '''
    Calculates DICE Loss, with modifications to support multiple channels and volumetric 
    '''
    def __init__(self, volumetric=False, negative_loss=False, per_channel=False):
        self.name = "DICE Loss"
        super(DICELoss, self).__init__()
        self.volumetric = volumetric
        self.negative_loss = negative_loss
        self.per_channel = per_channel

        print(f"DICE Loss initialized with volumetric={volumetric}, negative? {negative_loss}, per_channel {per_channel}")

    def __call__(self, probs, targets):
        '''
        probs: output of last convolution, sigmoided or not (use apply_sigmoid=True if not)
        targets: binary target mask
        '''
        p_min = probs.min()
        p_max = probs.max()
        assert p_max <= 1.0 and p_min >= 0.0, "FATAL ERROR: DICE loss input not bounded! Did you apply sigmoid?"

        score = 0

        if self.per_channel:
            assert len(targets.shape) >= 4, ("less than 4 dimensions makes no sense with multi channel in a batch of 2D or 3D"
                                             "volumes")
            nchannels = targets.shape[1]
            if self.volumetric:
                score = torch.stack([vol_dice(probs[:, c], targets[:, c]) for c in range(nchannels)]).mean()
            else:
                score = torch.stack([batch_dice(probs[:, c], targets[:, c]) for c in range(nchannels)]).mean()
        else:
            if self.volumetric:
                score = vol_dice(probs, targets)
            else:
                score = batch_dice(probs, targets)

        if self.negative_loss:
            loss = -score
        else:
            loss = 1 - score

        return loss


def vol_dice(inpt, target, smooth=1.0):
    '''
    Calculate DICE of volume
    '''
    # q = inpt.size(0)
    assert len(inpt) != 0, " trying to compute DICE of nothing"

    iflat = inpt.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    eps = 0
    if smooth == 0.0:
        eps = sys.float_info.epsilon

    iflat_sum = iflat.sum()
    tflat_sum = tflat.sum()

    if iflat_sum.item() == 0.0 and tflat_sum.item() == 0.0:
        # print("DICE Metric got black mask and prediction!")
        dice = torch.tensor(1.0, requires_grad=True, device=inpt.device)
    else:
        dice = (2. * intersection + smooth) / (iflat_sum + tflat_sum + smooth + eps)

    value = dice.item()
    assert value >= 0.0 or value <= 1.0, " DICE not between 0 and 1! something is wrong"

    return dice


def batch_dice(inpt, target, smooth=1.0):
    '''
    Calculate DICE of a batch of two binary masks
    Returns mean dice of all slices
    '''
    q = inpt.size(0)
    assert len(inpt) != 0, " trying to compute DICE of nothing"

    iflat = inpt.contiguous().view(q, -1)
    tflat = target.contiguous().view(q, -1)
    intersection = (iflat * tflat).sum(dim=1)

    eps = 0
    if smooth == 0.0:
        eps = sys.float_info.epsilon

    iflat_sum = iflat.sum(dim=1)
    tflat_sum = tflat.sum(dim=1)

    dice = (2. * intersection + smooth) / (iflat_sum + tflat_sum + smooth + eps)

    dice = dice.mean()
    value = dice.item()
    assert value >= 0.0 or value <= 1.0, " DICE not between 0 and 1! something is wrong"

    return dice


class CoUNet3D_metrics():
    def __init__(self, classes=["P", "L"]):
        self.dice = DICEMetric(per_channel_metric=True)
        self.classes = classes

    def __call__(self, preds, tgt):
        dices = self.dice(preds, tgt)
        report = {}

        for i, c in enumerate(self.classes):
            report[f"{c}_dice"] = dices[i]

        return report


class Metric():
    '''
    All metrics should extend this, and define if lower is better and its name.
    '''
    def __init__(self):
        assert hasattr(self, 'lower_is_best')
        assert hasattr(self, 'name')

    def __call__(self, outputs: Tensor, target: Tensor) -> float:
        raise NotImplementedError


class DICEMetric(Metric):
    '''
    Calculates DICE Metric
    '''
    def __init__(self, apply_sigmoid=False, mask_ths=0.5, skip_ths=False, per_channel_metric=False):
        self.name = "DICE"
        self.lower_is_best = False
        super(DICEMetric, self).__init__()
        self.apply_sigmoid = apply_sigmoid
        self.mask_ths = mask_ths
        self.skip_ths = skip_ths
        self.per_channel_metric = per_channel_metric
        print(f"DICE Metric initialized with apply_sigmoid={apply_sigmoid}, mask_ths={mask_ths}, skip_ths={skip_ths}, "
              f"per_channel={per_channel_metric}")

    def __call__(self, probs, target):
        '''
        Returns only DICE metric, as volumetric dice
        probs: output of last convolution, sigmoided or not (use apply_sigmoid=True if not)
        targets: float binary target mask
        '''
        probs = probs.type(torch.float32)
        target = target.type(torch.float32)

        if self.apply_sigmoid:
            probs = probs.sigmoid()

        p_min = probs.min()
        assert p_min >= 0.0, "FATAL ERROR: DICE metric input not positive! Did you apply sigmoid?"

        if self.skip_ths:
            mask = probs
        else:
            mask = (probs > self.mask_ths).float()

        if self.per_channel_metric:
            assert len(target.shape) >= 4, ("less than 4 dimensions makes no sense with multi channel in a batch of 2D or 3D"
                                            "volumes")
            nchannels = target.shape[1]
            return [vol_dice(mask[:, c], target[:, c], smooth=0.0).item() for c in range(nchannels)]
        else:
            return vol_dice(mask, target, smooth=0.0).item()

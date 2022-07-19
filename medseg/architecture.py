import torch
from torch import nn

from medseg.edet.modeling_efficientdet import EfficientDetForSemanticSegmentation


class MEDSeg(nn.Module):
    def __init__(self, nin=3, nout=3, apply_sigmoid=False, dropout=None, backbone="effnet", pretrained=True):
        super().__init__()
        if backbone == "effnet":
            self.model = EfficientDetForSemanticSegmentation(num_classes=nout, load_weights=pretrained,
                                                             apply_sigmoid=apply_sigmoid, expand_bifpn=True, dropout=dropout)
        elif backbone == "convnext":
            # TODO
            raise NotImplementedError

        self.nin = nin
        if self.nin not in [1, 3]:
            self.in_conv = nn.Conv2d(in_channels=self.nin, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)

        print(f"MEDSeg initialized. nin: {nin}, nout: {nout}, apply_sigmoid: {apply_sigmoid}")

    def forward(self, x):
        if self.nin == 1:
            x_in = torch.zeros(size=(x.shape[0], 3) + x.shape[2:], device=x.device, dtype=x.dtype)
            x_in[:, 0] = x[:, 0]
            x_in[:, 1] = x[:, 0]
            x_in[:, 2] = x[:, 0]
            x = x_in
        elif self.nin == 3:
            pass
        else:
            x = self.in_conv(x)

        return self.model(x)
        
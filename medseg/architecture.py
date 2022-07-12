import torch
from torch import nn

from medseg.edet.modeling_efficientdet import EfficientDetForSemanticSegmentation


class CoEDET(nn.Module):
    def __init__(self, head_type=None):
        super().__init__()
        num_classes = 2 if head_type is None else 4
        apply_sigmoid = True if head_type is None else False
        expand_bifpn = head_type is None
        self.model = EfficientDetForSemanticSegmentation(num_classes=num_classes, load_weights=False,
                                                         apply_sigmoid=apply_sigmoid, expand_bifpn=expand_bifpn)
        if head_type is None:
            self.head = None
        else:
            self.head = self.get_head(head_type)

        print(f"CoEDET initialized. Head: {head_type}. Expand BiFPN: {expand_bifpn}.")

    def get_head(self, head_type):
        if head_type == "baseline":
            return nn.Sequential(nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(4),
                                 nn.LeakyReLU(),
                                 nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=2, stride=2),
                                 nn.Sigmoid())
        else:
            raise ValueError(f"Unsupported head_type {head_type}")

    def forward(self, x):
        x_in = torch.zeros(size=(x.shape[0], 3) + x.shape[2:], device=x.device, dtype=x.dtype)
        x_in[:, 0] = x[:, 0]
        x_in[:, 1] = x[:, 0]
        x_in[:, 2] = x[:, 0]
        x = x_in

        if self.head is None:
            return self.model(x)
        else:
            return self.head(self.model(x))
            
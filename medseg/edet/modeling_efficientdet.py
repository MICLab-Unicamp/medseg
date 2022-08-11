# Author: Zylo117
# Modified by Israel and Diedre

import torch
from torch import nn

from .efficientnet.utils import MemoryEfficientSwish

from .efficientdet.model import BiFPN, Regressor, Classifier, EfficientNet, SegmentationClasssificationHead
from .efficientdet.utils import Anchors


class EfficientDet(nn.Module):
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, **kwargs):
        super(EfficientDet, self).__init__()
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]
        self.aspect_ratios = kwargs.get(
            'ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get(
            'scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef])
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef])

        self.anchors = Anchors(
            anchor_scale=self.anchor_scale[compound_coef], **kwargs)

        self.backbone_net = EfficientNet(
            self.backbone_compound_coef[compound_coef], load_weights)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def extract_backbone_features(self, inputs):
        max_size = inputs.shape[-1]

        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        return features

    def extract_bifpn_features(self, features):
        features = self.bifpn(features)
        return features

    def forward(self, inputs):
        features = self.extract_backbone_features(inputs)
        features = self.extract_bifpn_features(features)

        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(inputs, inputs.dtype).to(inputs.device)

        return features, regression, classification, anchors

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')


class EfficientDetForSemanticSegmentation(nn.Module):

    def __init__(self, load_weights=True, num_classes=2, apply_sigmoid=False, compound_coef=4, repeat=3, expand_bifpn=False, dropout=None, backbone="effnet"):
        super().__init__()
        assert backbone == "effnet", "Backbones other than effnet are not public yet."
        self.compound_coef = compound_coef
        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.num_classes = num_classes
        self.expand_bifpn = expand_bifpn
        self.backbone = backbone

        if self.expand_bifpn == True or self.expand_bifpn == "conv":
            print("Using transposed convolution for bifpn result expansion and upsample 2 for convnext features")
            self.expand_conv = nn.Sequential(nn.ConvTranspose2d(128, 128, 2, 2),
                                             nn.BatchNorm2d(128),
                                             MemoryEfficientSwish())
            self.convnext_upsample_scale = 2
        elif self.expand_bifpn == "upsample":
            print("Using upsample for bifpn result expansion and upsample 2 expansion of convnext features")
            self.expand_conv = nn.UpsamplingBilinear2d(scale_factor=2)
            self.convnext_upsample_scale = 2
        elif self.expand_bifpn == "upsample_4":
            print("Using upsample 4 for bifpn result expansion and no expansion of convnext features")
            self.expand_conv = nn.UpsamplingBilinear2d(scale_factor=4)
            self.convnext_upsample_scale = 0
        elif self.expand_bifpn == False:
            print("Bifpn expansion disabled")
            self.convnext_upsample_scale = 4
            pass
        else:
            raise ValueError(f"Expand bifpn {self.expand_bifpn} not supported!")
        
        conv_channel_coef = {
            # the channels of P2/P3/P4.
            0: [16, 24, 40],
            4: [24, 32, 56],
            "convnext": [96, 192, 384]
        }

        if self.backbone == "convnext":
            print("Changing compound coeff of BiFPN due to convnext backbone")
            compound_coef = "convnext"
            print(f"Convnext upsample scale {self.convnext_upsample_scale}")

        bifpn_channels = 128
        self.bifpn = nn.Sequential(
            *[BiFPN(bifpn_channels,
                    conv_channel_coef[compound_coef],
                    True if i == 0 else False,
                    attention=True if self.compound_coef < 6 else False)
              for i in range(repeat)])

        self.classifier = SegmentationClasssificationHead(in_channels=bifpn_channels,
                                                          num_classes=self.num_classes,
                                                          num_layers=repeat,  # should it be repeat - 1?
                                                          apply_sigmoid=apply_sigmoid,
                                                          dropout=dropout
                                                          )
        if self.backbone == "effnet":
            self.backbone_net = EfficientNet(self.backbone_compound_coef[self.compound_coef], load_weights)
        elif self.backbone == "convnext":
            raise NotImplementedError("Our implementation of convnext is not public yet.")

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def extract_backbone_features(self, inputs):
        max_size = inputs.shape[-1]

        if self.backbone == "effnet":
            p2, p3, p4, _ = self.backbone_net(inputs)
        elif self.backbone == "convnext":
            p2, p3, p4 = self.backbone_net.forward_seg_features(inputs, self.convnext_upsample_scale)

        features = (p2, p3, p4)
        return features

    def extract_bifpn_features(self, features):
        features = self.bifpn(features)
        return features

    def forward(self, inputs):
        self.input_shape = inputs.shape
        features = self.extract_backbone_features(inputs)
        feat_map = self.extract_bifpn_features(features)[0]

        if self.expand_bifpn:
            feat_map = self.expand_conv(feat_map)

            # Features must be same size as input
            diffX = inputs.shape[2] - feat_map.size()[2]
            diffY = inputs.shape[3] - feat_map.size()[3]
            feat_map = torch.nn.functional.pad(feat_map, (diffY // 2, diffY - diffY // 2, diffX // 2, diffX - diffX // 2))

        classification = self.classifier(feat_map)

        return classification

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')


class EfficientDetDoesEAST(nn.Module):

    def __init__(self, load_weights=True, compound_coef=4):
        super().__init__()
        self.num_classes = 40
        self.backbone = EfficientDetForSemanticSegmentation(
            load_weights=load_weights, num_classes=self.num_classes, apply_sigmoid=False, compound_coef=compound_coef)

        self.scores = nn.Conv2d(self.num_classes, 5, 1, groups=5)

    def forward(self, x):
        _, _, height, width = x.shape
        feats = self.backbone(x)

        scores = self.scores(feats)
        scores = torch.sigmoid(scores)

        score_map = scores[:, :1]
        geo_height = scores[:, 1:3] * height  # top and bottom
        geo_width = scores[:, 3:] * width  # left and right
        return torch.cat((score_map, geo_height, geo_width), dim=1)

# Author: Zylo117
# Modified by Israel and Diedre

from typing import List
from collections import OrderedDict
import torch
from torch import nn

from .efficientnet.utils import MemoryEfficientSwish

from .efficientdet.model import BiFPN, Regressor, Classifier, EfficientNet, SegmentationClasssificationHead
from .efficientdet.utils import Anchors
from ..convnext import convnext_tiny


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


class FeatureFusion(nn.Module):
    '''
    Feature fusion module that makes use of all BiFPN features for segmentation instead of only
    upsampling the highest spatial resolution.

    upsample_sum: upsamples and sums all features 
    (ESC) exponential_stride_compression: increases kernel size and dilation and exponentially increases the stride to compress features, from B, C, x, y into a B, C, x/256, y/256 array that can be linearized easily with reshape. Minimum input size 256x256.
    seg_exponential_stride_compression: use values derived from ESC to weight high resolution features
    '''
    SUPPORTED_STRATS = ["upsample_sum", "exponential_stride_compression", "seg_exponential_stride_compression", "nonlinear_esc"]
    def __init__(self, in_c: int, out_c: int, key: str):
        super().__init__()
        print(f"SELECTING FEATURE ADAPTER: {key}")
        self.key = key
        if key == "upsample_sum":
            self.feature_adapters =  nn.ModuleList([nn.UpsamplingBilinear2d(scale_factor=2),
                                                    nn.UpsamplingBilinear2d(scale_factor=4),
                                                    nn.UpsamplingBilinear2d(scale_factor=8),
                                                    nn.UpsamplingBilinear2d(scale_factor=16),
                                                    nn.UpsamplingBilinear2d(scale_factor=32)])
        elif key == "exponential_stride_compression":
            self.feature_adapters =  nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=11, padding=5, stride=128, dilation=6, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False)),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=9, padding=4, stride=64, dilation=5, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False)),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=7, padding=3, stride=32, dilation=4, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False)),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=5, padding=2, stride=16, dilation=3, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False)),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=3, padding=1, stride=8, dilation=2, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False))])
        elif key == "seg_exponential_stride_compression":
            self.feature_adapters =  nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=11, padding=5, stride=128, dilation=6, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False)),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=9, padding=4, stride=64, dilation=5, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False)),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=7, padding=3, stride=32, dilation=4, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False)),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=5, padding=2, stride=16, dilation=3, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False)),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=3, padding=1, stride=8, dilation=2, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False))])
            self.upsampler = nn.UpsamplingBilinear2d(scale_factor=2)
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif key == "nonlinear_esc":  # Save this for future embbedding building for transformers 
            # Reduced stride progression, trusting average pooling, makes network work with 128x128 inputs minimum
            self.feature_adapters =  nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=11, padding=5, stride=64, dilation=4, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False),
                                                                  nn.LeakyReLU()),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=9, padding=4, stride=32, dilation=3, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False),
                                                                  nn.LeakyReLU()),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=7, padding=3, stride=16, dilation=3, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False),
                                                                  nn.LeakyReLU()),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=5, padding=2, stride=8, dilation=2, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False),
                                                                  nn.LeakyReLU()),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=3, padding=1, stride=4, dilation=2, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False),
                                                                  nn.LeakyReLU())])
            self.upsampler = nn.UpsamplingBilinear2d(scale_factor=2)
            self.pooling = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"Unsupported feature adapter {key}. Use one of {FeatureFusion.SUPPORTED_STRATS}")
        self.latent_space = None
    
    def get_latent_space(self):
        # Save this for future transformer involvement
        B, C, _, _ = self.latent_space.shape
        return self.latent_space.reshape(B, C)

    def forward(self, in_features: List[torch.Tensor]) -> torch.Tensor:
        out_features = None
        for feature_adapter, in_feature in zip(self.feature_adapters, in_features):
            if out_features is None:
                out_features = feature_adapter(in_feature)
            else:
                out_features += feature_adapter(in_feature)
        
        if self.key in ["nonlinear_esc", "seg_exponential_stride_compression"]:
            self.latent_space = self.pooling(out_features)
            return self.upsampler(in_features[0]) * self.latent_space  # latent space weights channel contributions
        else:
            return out_features


class EfficientDetForSemanticSegmentation(nn.Module):

    def __init__(self, load_weights=True, num_classes=2, apply_sigmoid=False, compound_coef=4, repeat=3, expand_bifpn=False, dropout=None, backbone="effnet", 
                 num_classes_atm=None,    # airway tree modelling
                 num_classes_rec=None,    # reconstructing branch
                 num_classes_vessel=None, # vessel branch for future parse data (hopefully)
                 new_latent_space=False):
        '''
        load_weights: wether to load pre trained as backbone
        num_classes: number of classes for primary downstream segmentation task 
        apply_sigmoid: wether to apply sigmoid to output. DEPRECATED. do this outside depending on application.
        compound_coef: which efficientnet variation to base the architecture of, only supports 4.
        repeat: how many conv blocks on the segmentation head
        expand_bifpn: how to expand the bifpn features. Upsample is best
        dropout: add additional dropouts to segmentation heads, doesnt work well DEPRECATED.
        backbone: efficientnet or convnext as backbone
        num_classes_aux: number of classes for secondary segmentation task. If None will not initialize second output.
        '''
        super().__init__()
        self.compound_coef = compound_coef
        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.num_classes = num_classes
        self.expand_bifpn = expand_bifpn
        self.backbone = backbone

        feature_fusion = False
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
        elif self.expand_bifpn in FeatureFusion.SUPPORTED_STRATS:
            print(f"Enabling feature fusion through {self.expand_bifpn}")
            feature_fusion = True
        else:
            raise ValueError(f"Expand bifpn {self.expand_bifpn} not supported!")
        
        conv_channel_coef = {
            # the channels of P2/P3/P4.
            0: [16, 24, 40],
            4: [24, 32, 56],
            6: [32, 40, 72],
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

        # Main classifier
        self.classifier = SegmentationClasssificationHead(in_channels=bifpn_channels,
                                                          num_classes=self.num_classes,
                                                          num_layers=repeat,
                                                          apply_sigmoid=apply_sigmoid,
                                                          dropout=dropout
                                                          )

        # Reconstruction branch
        if num_classes_rec is not None:
            self.rec_classifier = SegmentationClasssificationHead(in_channels=bifpn_channels,
                                                                  num_classes=num_classes_rec, 
                                                                  num_layers=repeat,
                                                                  apply_sigmoid=apply_sigmoid,
                                                                  dropout=dropout
                                                                  )

        # ATM branch
        if num_classes_atm is not None:
            self.atm_classifier = SegmentationClasssificationHead(in_channels=bifpn_channels,
                                                                  num_classes=num_classes_atm, 
                                                                  num_layers=repeat,
                                                                  apply_sigmoid=apply_sigmoid,
                                                                  dropout=dropout
                                                                  )

        # vessel branch
        if num_classes_vessel is not None:
            self.vessel_classifier = SegmentationClasssificationHead(in_channels=bifpn_channels,
                                                                     num_classes=num_classes_vessel, 
                                                                     num_layers=repeat,
                                                                     apply_sigmoid=apply_sigmoid,
                                                                     dropout=dropout
                                                                     )

        # Where bifpn upsampling happens
        if feature_fusion:
            self.feature_adapters = FeatureFusion(bifpn_channels, bifpn_channels, key=self.expand_bifpn)   
        else:
            self.feature_adapters = None

        # Backbone derived latent space
        self.new_latent_space = new_latent_space
        if self.new_latent_space:
            raise DeprecationWarning("Latent space generation will be deprecated")
            self.backbone_latent_space_pooling = nn.AdaptiveAvgPool2d(2)
            self.backbone_latent_space_projection = nn.Sequential(nn.Linear(640, 320),
                                                                  nn.LeakyReLU(),
                                                                  nn.Linear(320, 160))
        self.latent_space = None

        if self.backbone == "effnet":
            self.backbone_net = EfficientNet(self.backbone_compound_coef[self.compound_coef], load_weights)
        elif self.backbone == "convnext":
            self.backbone_net = convnext_tiny(pretrained=load_weights)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_latent_space(self):
        raise DeprecationWarning("Latent space generation will be deprecated")
        if self.feature_adapters is None:
            return None
        else:
            return self.feature_adapters.get_latent_space()

    def extract_backbone_latent_space(self, last_conv_features):
        raise DeprecationWarning("Latent space generation will be deprecated")
        B = last_conv_features.shape[0]
        
        latent_space = self.backbone_latent_space_pooling(last_conv_features).reshape(B, -1)
        self.backbone_latent_space = self.backbone_latent_space_projection(latent_space)
    
    def get_backbone_latent_space(self):
        raise DeprecationWarning("Latent space generation will be deprecated")
        return self.backbone_latent_space
        
    def extract_backbone_features(self, inputs):
        if self.backbone == "effnet":
            p2, p3, p4, p5 = self.backbone_net(inputs)
            if self.new_latent_space:
                raise DeprecationWarning("Latent space generation will be deprecated")
                self.extract_backbone_latent_space(p5)
        elif self.backbone == "convnext":
            p2, p3, p4 = self.backbone_net.forward_seg_features(inputs, self.convnext_upsample_scale)

        features = (p2, p3, p4)
        return features

    def extract_bifpn_features(self, features):
        features = self.bifpn(features)
        return features

    def forward(self, inputs):
        features = self.extract_backbone_features(inputs)
        feat_map = self.extract_bifpn_features(features)
        
        # Here is where BIFPN feature fusion happens (ISBI paper?)
        if self.feature_adapters is not None:
            feat_map = self.feature_adapters(feat_map)
        else:
            feat_map = feat_map[0]

            if self.expand_bifpn:
                feat_map = self.expand_conv(feat_map)

                # Features must be same size as input
                # diffX = inputs.shape[2] - feat_map.size()[2]
                # diffY = inputs.shape[3] - feat_map.size()[3]
                # feat_map = torch.nn.functional.pad(feat_map, (diffY // 2, diffY - diffY // 2, diffX // 2, diffX - diffX // 2))
                # Align disabled!

        classification = self.classifier(feat_map)

        rdict = OrderedDict()
        rdict["main"] = classification
        return_dict = False
        for branch in ["rec", "atm", "vessel"]:
            branch_module = getattr(self, f"{branch}_classifier", None)
            if branch_module is not None:
                return_dict = True
                rdict[branch] = branch_module(feat_map)
        
        # Dict return form in case auxiliary branches are involved
        if return_dict:
            # LATENT SPACE DEPRECATED 
            # rdict["latent_space"] = self.get_backbone_latent_space()
            return rdict
        else:
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

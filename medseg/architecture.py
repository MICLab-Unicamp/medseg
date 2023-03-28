'''
If you use please cite:
CARMO, Diedre et al. Multitasking segmentation of lung and COVID-19 findings in CT scans using modified EfficientDet, UNet and MobileNetV3 models. In: 17th International Symposium on Medical Information Processing and Analysis. SPIE, 2021. p. 65-74.
'''
import torch
from torch import nn
from efficientnet_pytorch.utils import round_filters
from medseg.edet.modeling_efficientdet import EfficientDetForSemanticSegmentation


class MEDSeg(nn.Module):
    def __init__(self, nin=3, nout=3, apply_sigmoid=False, dropout=None, backbone="effnet", pretrained=True, expand_bifpn="upsample", imnet_norm=False,
                 num_classes_atm=None,
                 num_classes_rec=None,
                 num_classes_vessel=None,
                 stem_replacement=False,
                 new_latent_space=False,
                 compound_coef=4):  # compound always has been 4 by default before
        super().__init__()
        print("WARNING: default expand_bifpn changed to upsample!")
        self.model = EfficientDetForSemanticSegmentation(num_classes=nout, 
                                                         load_weights=pretrained,
                                                         apply_sigmoid=apply_sigmoid, 
                                                         expand_bifpn=expand_bifpn, 
                                                         dropout=dropout,
                                                         backbone=backbone,
                                                         compound_coef=compound_coef,
                                                         num_classes_atm=num_classes_atm,
                                                         num_classes_rec=num_classes_rec,
                                                         num_classes_vessel=num_classes_vessel,
                                                         new_latent_space=new_latent_space)

        self.feature_adapters = self.model.feature_adapters

        if imnet_norm:
            print("Performing imnet normalization internally, assuming inputs between 1 and 0")
            self.imnet_norm = ImNetNorm()
        else:
            self.imnet_norm = nn.Identity()

        self.nin = nin
        if self.nin not in [1, 3]:
            self.in_conv = nn.Conv2d(in_channels=self.nin, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)

        if stem_replacement:
            assert backbone == "effnet", "Stem replacement only valid for efficientnet"
            print("Performing stem replacement on EfficientNet backbone (this runs after initialization)")
            self.model.backbone_net.model._conv_stem = EffNet3DStemReplacement(self.model.backbone_net.model)

        print(f"MEDSeg initialized. nin: {nin}, nout: {nout}, apply_sigmoid: {apply_sigmoid}, dropout: {dropout}," 
              f"backbone: {backbone}, pretrained: {pretrained}, expand_bifpn: {expand_bifpn}, pad align DISABLED, stem_replacement {stem_replacement}"
              f"new latent space extraction {new_latent_space}")

    def extract_backbone_features(self, inputs):
        return self.model.extract_backbone_features(inputs)

    def extract_bifpn_features(self, features):
        return self.model.extract_bifpn_features(features)
    
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

        x = self.imnet_norm(x)

        return self.model(x)


class EffNet3DStemReplacement(nn.Module):
    def __init__(self, effnet_pytorch_instance):
        super().__init__()
        out_channels = round_filters(32, effnet_pytorch_instance._global_params)
        self.conv = nn.Conv3d(1, out_channels, kernel_size=3, stride=1, padding="valid", bias=False)
        self.pad = nn.ZeroPad2d(1)
        self.conv_pool = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        
    def forward(self, x):
        '''
        x is 4D batch but will be treated as 5D
        '''
        x = self.conv(x.unsqueeze(1)).squeeze(2)  # [B, 3, X, Y] -> [B, 1, 3, X, Y] 
                                                  # -> [B, OUT_CH, 1, X, Y] -> [B, OUT_CH, X, Y]
        x = self.pad(x)
        x = self.conv_pool(x)
        return x
        

class ImNetNorm():
    '''
    Assumes input between 1 and 0
    '''
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, xim):
        with torch.no_grad():
            for i in range(3):
                xim[:, i] = (xim[:, i] - self.mean[i])/self.std[i]
        
        return xim
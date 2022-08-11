'''
Multitask architecture lightning module
'''
import pytorch_lightning as pl
import torchmetrics
import random
from torch import nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import matplotlib.pyplot as plt
from medseg.utils import get_optimizer, DICELoss, DICEMetric
from medseg.architecture import MEDSeg
from medseg.unet_v2 import UNet


class Seg2DModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.pretraining = self.hparams.pretraining
        unet = getattr(self.hparams, "unet", False)
        dropout = getattr(self.hparams, "dropout", None)
        self.findings_only = getattr(self.hparams, "findings_only", False)
        self.weight_decay = getattr(self.hparams, "weight_decay", None)
        self.scheduling_factor = getattr(self.hparams, "scheduling_factor", None)
        self.scheduling = getattr(self.hparams, "scheduling", "step")
        self.scratch = getattr(self.hparams, "scratch", False)
        self.expand_bifpn = getattr(self.hparams, "expand_bifpn", "conv")
        self.backbone = getattr(self.hparams, "backbone", "effnet")

        if dropout == True:
            print("WARNING: Replacing old hparams true dropout by full dropout")
            dropout = "full"
        
        if unet:
            self.model = UNet(self.hparams.nin, self.hparams.seg_nout, True, "2d", 64, dropout=dropout)
        else:
            self.model = MEDSeg(self.hparams.nin, self.hparams.seg_nout, apply_sigmoid=False, backbone=self.backbone, expand_bifpn=self.expand_bifpn, dropout=dropout, pretrained=not self.scratch)
        
        self.pretrained_weights = self.hparams.pretrained_weights
        if self.pretrained_weights is not None:
            pretrained_module = Seg2DModule.load_from_checkpoint(self.pretrained_weights)
            if unet:
                print("Loading pre-trained encoder...")
                self.model.enc = pretrained_module.model.enc
            else:
                print("Loading pre-trained backbone and bifpn...")
                
                # Segmentation head will be trained from scratch. Load bakcbone and bifpn
                self.model.model.backbone_net = pretrained_module.model.model.backbone_net
                self.model.model.bifpn = pretrained_module.model.model.bifpn
            print("Done.")

        if self.pretraining:
            self.lossfn = nn.MSELoss()
            self.maer = torchmetrics.MeanAbsoluteError()
        else:
            self.lossfn = DICELoss(volumetric=False, per_channel=True)
            self.dicer = DICEMetric(per_channel_metric=True)

    def debug_target(self, x, y):
        B = x.shape[0]
        if self.pretraining:
            yd = y
        else:
            yd = y.argmax(dim=1, keepdim=True)
        
        if x.shape[1] not in [1, 3]:
            xd = x.mean(dim=1, keepdim=True)
        else:
            xd = x
        plt.figure(num="Input")
        b = random.randint(0, B-1)
        plt.imshow(xd.detach().cpu().numpy()[b, :, :, :].transpose(1, 2, 0), cmap="gray")
        plt.figure(num="Target")
        plt.imshow(yd.detach().cpu().numpy()[b, 0, :, :], cmap="gray")
        plt.show()

    def forward(self, x):
        if self.pretraining:
            return self.model(x)
        elif self.findings_only:
            return self.model(x).sigmoid()
        else:
            return self.model(x).softmax(dim=1)

    def compute_loss(self, x, y, y_hat, meta, prestr):
        # Extract multitask targets
        loss = self.lossfn(y_hat, y)

        if self.pretraining:
            self.log(f"{prestr}rec_loss", loss, on_step=False, on_epoch=True)
        else:
            self.log(f"{prestr}seg_loss", loss, on_step=False, on_epoch=True)
        
        self.log(f"{prestr}loss", loss, on_step=True, on_epoch=True)

        return loss

    def compute_metrics(self, x, y, y_hat, meta):
        if self.pretraining:
            self.maer(y_hat, y)
            self.log("mean_abs_err", self.maer, on_epoch=True, on_step=False, prog_bar=True)
        else:
            metrics = self.dicer(y_hat, y)
            if len(metrics) == 1 and self.findings_only:
                unhealthy_dice = metrics[0]        
                self.log("unhealthy_dice", unhealthy_dice, on_epoch=True, on_step=False, prog_bar=True)
            elif len(metrics) == 3:
                bg_dice, healthy_dice, unhealthy_dice = metrics    
                self.log("unhealthy_dice", unhealthy_dice, on_epoch=True, on_step=False, prog_bar=True)
                
                self.log("bg_dice", bg_dice, on_epoch=True, on_step=False, prog_bar=True)
                self.log("healthy_dice", healthy_dice, on_epoch=True, on_step=False, prog_bar=True)
            elif len(metrics) == 4:
                bg_dice, healthy_dice, ggo_dice, con_dice = metrics    
                self.log("ggo_dice", ggo_dice, on_epoch=True, on_step=False, prog_bar=True)
                self.log("con_dice", con_dice, on_epoch=True, on_step=False, prog_bar=True)

                self.log("bg_dice", bg_dice, on_epoch=True, on_step=False, prog_bar=True)
                self.log("healthy_dice", healthy_dice, on_epoch=True, on_step=False, prog_bar=True)

    def training_step(self, train_batch, batch_idx):
        x, y, meta = train_batch
        if self.findings_only and y.shape[1] == 3:
            y = y[:, 2:3]  
        if self.hparams.debug:
            print("Train step debug")
            self.debug_target(x, y)

        y_hat = self.forward(x)

        loss = self.compute_loss(x, y, y_hat, meta, prestr='')

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, meta = val_batch
        if self.findings_only and y.shape[1] == 3:
            y = y[:, 2:3]  
        if self.hparams.debug:
            print("Validation step debug")
            self.debug_target(x, y)
        
        y_hat = self.forward(x)
        
        self.compute_loss(x, y, y_hat, meta, prestr="val_")
        self.compute_metrics(x, y, y_hat, meta)

    def configure_optimizers(self):
        '''
        Select optimizer and scheduling strategy according to hparams.
        '''
        opt = getattr(self.hparams, "opt", "Adam")
        optimizer = get_optimizer(opt, self.model.parameters(), self.hparams.lr, wd=self.weight_decay)
        print(f"Opt: {opt}, Weight decay: {self.weight_decay}")

        if self.scheduling == "step" and self.scheduling_factor is None:
            print("Not using any scheduler")
            return optimizer
        elif self.scheduling_factor is not None and self.scheduling == "step":
            print(f"Using step LR {self.scheduling_factor}!")
            scheduler = StepLR(optimizer, 1, self.scheduling_factor, verbose=True)
            return [optimizer], [scheduler]
        elif self.scheduling == "cosine":
            print(f"Using CosineAnnealingLR with tmax {self.scheduling_factor}!")
            scheduler = CosineAnnealingLR(optimizer, T_max=self.scheduling_factor, verbose=True)
            return [optimizer], [scheduler]

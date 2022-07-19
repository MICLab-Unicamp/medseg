'''
Main Lightning Module definition
'''
import argparse
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR

from medseg.utils import check_params
from medseg.loss_metrics import DICELoss, CoUNet3D_metrics
from medseg.architecture import MEDSeg
from medseg.radam import get_optimizer


class MEDSegModule(pl.LightningModule):
    '''
    Generic classifier that can work with different architectures and datasets.
    '''
    def __init__(self, hparams: argparse.Namespace):
        '''
        hparams: Consult hparams_builder.parse_hparams for possible hyperparameter definitions.
        '''
        super().__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.save_hyperparameters(check_params(hparams))

        # Loss and metrics
        self.loss, self.metric = DICELoss(volumetric=True if '3d' in self.hparams.paradigm else False, per_channel=True), CoUNet3D_metrics()
        self.loss_str, self.metric_str = str(self.loss), str(self.metric)

        architecture = MEDSeg
        self.model = architecture()

    def forward(self, x):
        '''
        x: input is forwarded to the architecture in question (defined in __init__ according to hparams).
        '''
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        x, y, _ = train_batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, _ = val_batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        metrics = self.metric(y_hat, y)
        metrics["val_loss"] = loss

        for k, v in metrics.items():
            self.log(k, v, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        '''
        Select optimizer and scheduling strategy according to hparams.
        '''
        optimizer = get_optimizer(self.hparams.opt_name, self.model.parameters(), self.hparams.lr, self.hparams.wd)

        if self.hparams.scheduling_factor is None:
            return optimizer
        else:
            scheduler = StepLR(optimizer, 1, self.hparams.scheduling_factor)

            return [optimizer], [scheduler]
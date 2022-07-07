'''
Loads the best trainings to create the definitive lung segmentation pipeline

build nice pipeline involving all three weights above.
'''
import uuid
import torch
import numpy as np
import subprocess
from DLPT.post.labeling3d import get_connected_components
from coedet.seg_2d_module import Seg2DModule
from coedet.poly_seg_3d_module import PolySeg3DModule
from coedet.eval_2d_utils import stack_predict, multi_view_consensus
import SimpleITK as sitk
from torch.nn import functional as F
from DLPT.models.unet_v2 import UNetEncoder
from tqdm import tqdm


class PrintInterface():
    def __init__(self, tqdm_iter):
        self.tqdm_iter = tqdm_iter

    def write(self, x):
        self.tqdm_iter.put(("write", x))

    def progress(self, x):
        self.tqdm_iter.put(("iterbar", x))


class SegmentationPipeline():
    def __init__(self,
                 best_3d="/home/diedre/diedre_phd/phd/models/wd_step_poly_lung_softm-epoch=85-val_loss=0.03.ckpt",  # POLY LUNG SOFTM
                 best_25d="/home/diedre/diedre_phd/phd/models/wd_step_sme2d_coedet_fiso-epoch=72-val_loss=0.06-bg_dice=1.00-healthy_dice=0.92-unhealthy_dice=0.74.ckpt",  # SME2D COEDET FISO Full train wd step  better
                 best_25d_raw="/home/diedre/diedre_phd/phd/models/wd_step_raw_medseg_positive-epoch=27-val_loss=0.15-bg_dice=1.00-healthy_dice=0.84-unhealthy_dice=0.71.ckpt",  # Best raw axial 2.5D model, trained on positive 256 slices only
                 batch_size=4,
                 n=10,
                 cpu=False):  
        self.version = 'v1.1_wd_step'
        self.batch_size = batch_size
        self.n = n
        self.device = torch.device("cpu") if cpu else torch.device("cuda:0")
        self.model_3d = PolySeg3DModule.load_from_checkpoint(best_3d).eval()
        self.model_25d = Seg2DModule.load_from_checkpoint(best_25d).eval()
        self.model_25d_raw = Seg2DModule.load_from_checkpoint(best_25d_raw).eval()
        self.hparams = self.model_25d_raw.hparams
        self.hparams.experiment_name = '_'.join([self.model_3d.hparams.experiment_name,
                                                 self.model_25d.hparams.experiment_name,
                                                 self.model_25d_raw.hparams.experiment_name])

    def __call__(self, input_volume, spacing, tqdm_iter=None, minimum_return=False):
        assert input_volume.max() <= 1.0 and input_volume.min() >= 0.0
        
        if tqdm_iter is not None and not isinstance(tqdm_iter, tqdm):
            tqdm_iter = PrintInterface(tqdm_iter)

        with torch.no_grad():
            # 3D
            if tqdm_iter is not None:
                tqdm_iter.write("3D Prediction...")
                tqdm_iter.progress(20)
            self.model_3d = self.model_3d.to(self.device)
            input_volume = input_volume.to(self.device)
            pre_shape = input_volume.shape[2:]
            transformed_input = F.interpolate(input_volume, (128, 256, 256), mode="trilinear")
            output = self.model_3d(transformed_input, get_bg=True)[0]
            self.model_3d.cpu()
            input_volume = input_volume.cpu()
            output = F.interpolate(output, pre_shape, mode="nearest").squeeze().cpu().numpy()
            if self.n is not None:
                print("Computing attention")
                if tqdm_iter is not None:
                    tqdm_iter.write("Computing attention...")
                atts = get_atts_work(self.model_3d.model.enc, pre_shape)
            else:
                print("Skipping attention computation")
                if tqdm_iter is not None:
                    tqdm_iter.write("Skipping attention computation")
                atts = None
            tqdm_iter.progress(30)
                
            # 2.5D RAW
            if tqdm_iter is not None:
                tqdm_iter.write("2.5D Raw prediction...")
                tqdm_iter.progress(40)
            if self.n is None:
                output_f = stack_predict(self.model_25d_raw.to(self.device), input_volume, self.batch_size, extended_2d=1, get_uncertainty=self.n, device=self.device)[0]
            else:
                output_f, epistemic_uncertainties, mean_predictions = stack_predict(self.model_25d_raw.to(self.device), input_volume, self.batch_size, extended_2d=1, get_uncertainty=self.n, device=self.device)
            self.model_25d_raw.cpu()
            input_volume = input_volume.cpu()

            # Isometric single model 2.5D consensus
            if tqdm_iter is not None:
                tqdm_iter.write("2.5D Single model isometric consensus prediction...")
                tqdm_iter.progress(60)

            if not isinstance(spacing, np.ndarray):
                spacing = np.array([x.item() for x in spacing])
            dest_shape = (spacing*pre_shape).astype(int).tolist()
            input_volume.to(self.device)
            input_iso = F.interpolate(input_volume, size=dest_shape, mode="trilinear", align_corners=False).cpu()
            input_volume.cpu()
            orientations = [input_iso, 
                            input_iso.permute(0, 1, 3, 2, 4), 
                            input_iso.permute(0, 1, 4, 2, 3)]
            
            output_f_iso = multi_view_consensus([self.model_25d for _ in range(3)],
                                                 orientations=orientations,
                                                 tqdm_iter=None,
                                                 batch_size=self.batch_size,
                                                 extended_2d=1,
                                                 device=self.device)

            output_f_sm_consensus = F.interpolate(torch.from_numpy(output_f_iso).to(self.device), pre_shape, mode="nearest").squeeze().cpu().numpy()
       
        if tqdm_iter is not None:
            tqdm_iter.write("Post-processing...")
            tqdm_iter.progress(80)
        output_logits = {"3d_bg": output[0],
                         "3d_left_lung": output[1],
                         "3d_right_lung": output[2],
                         "3d_lung": output[1] + output[2],

                         "25d_bg": output_f[0],
                         "25d_lung": output_f[1],
                         "25d_findings": output_f[2],

                         "sm25d_bg": output_f_sm_consensus[0],
                         "sm25d_lung": output_f_sm_consensus[1],
                         "sm25d_findings": output_f_sm_consensus[2]}

        for k, v in output_logits.items():
            print(k, v.shape)
        
        output_logits["bg_ensemble"] = (output_logits["3d_bg"] + output_logits["25d_bg"] + output_logits["sm25d_bg"])/3
        output_logits["lung_ensemble"] = (output_logits["3d_lung"] + output_logits["25d_lung"] + output_logits["sm25d_lung"])/3
        output_logits["findings_ensemble"] = (output_logits["25d_findings"] + output_logits["sm25d_findings"])/2
        
        ensemble_consensus = np.zeros((3,) + pre_shape)
        ensemble_consensus[0] = output_logits["bg_ensemble"]
        ensemble_consensus[1] = output_logits["lung_ensemble"]
        ensemble_consensus[2] = output_logits["findings_ensemble"]

        lung = ensemble_consensus[1] + ensemble_consensus[2]
        findings = ensemble_consensus[2]
        lung, findings = (lung > 0.5).astype(np.int32), (findings > 0.5).astype(np.int32)
        lung, lung_lc, lung_labeled = get_connected_components(lung, return_largest=2)

        # Filter by lung region
        findings = findings*lung
        if self.n is not None:
            epistemic_uncertainties = epistemic_uncertainties*(np.expand_dims(lung, axis=0))
            mean_predictions = mean_predictions*(np.expand_dims(lung, axis=0))
        else:
            epistemic_uncertainties, mean_predictions = None, None

        inverse_lung = np.ones_like(lung) - lung
        ensemble_consensus[0] = ensemble_consensus[0] * inverse_lung
        ensemble_consensus[1] = ensemble_consensus[1] * lung
        ensemble_consensus[2] = ensemble_consensus[2] * lung

        ensemble_consensus_label = ensemble_consensus.argmax(axis=0)
        left_right_label = output.argmax(axis=0)

        torch.cuda.empty_cache()
        if minimum_return:
            return ensemble_consensus
        else:
            return ensemble_consensus_label, ensemble_consensus, left_right_label, output, lung, findings, atts, epistemic_uncertainties, mean_predictions 


def get_atts_work(encoder: UNetEncoder, pre_shape):
    '''
    Gets attention from unt encoder specifically
    '''
    atts = torch.stack([torch.from_numpy(x) for x in encoder.return_atts()]).unsqueeze(0)
    atts = F.interpolate(atts, pre_shape, mode="trilinear", align_corners=False).squeeze().numpy()
    return atts


def surface_render_itksnap(img: np.ndarray, int_tgt: np.ndarray, label='', block=False):
    uid = uuid.uuid4()
    img_path = f"/tmp/{label}_itksnap_{uid}.nii.gz"
    tgt_path = f"/tmp/{label}_tgt_itksnap_{uid}.nii.gz"
    sitk_image, sitk_tgt = sitk.GetImageFromArray(img), sitk.GetImageFromArray(int_tgt)
    sitk.WriteImage(sitk_image, img_path)
    sitk.WriteImage(sitk_tgt, tgt_path)

    if block:
        subprocess.run(["itksnap", "-g", img_path, "-s", tgt_path])
    else:
        subprocess.Popen(["itksnap", "-g", img_path, "-s", tgt_path])

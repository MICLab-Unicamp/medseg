'''
Loads the best trainings to create the definitive lung segmentation pipeline

build nice pipeline involving all three weights above.
'''
from queue import Empty
import uuid
import torch
import numpy as np
import subprocess
from medseg.postprocess import get_connected_components
from medseg.seg_2d_module import Seg2DModule
from medseg.poly_seg_3d_module import PolySeg3DModule
from medseg.eval_2d_utils import stack_predict, multi_view_consensus
import SimpleITK as sitk
from torch.nn import functional as F
from tqdm import tqdm


class PrintInterface():
    def __init__(self, tqdm_iter):
        self.tqdm_iter = tqdm_iter
        self.rot90 = False

    def write(self, x):
        self.tqdm_iter.put(("write", x))

    def progress(self, x):
        self.tqdm_iter.put(("iterbar", x))

    def image_to_front_end(self, x):
        if self.rot90:
            x = np.rot90(x, k=2, axes=(0, 1))

        self.tqdm_iter.put(("slice", x))

    def icon(self):
        self.tqdm_iter.put(("icon", ''))


class SegmentationPipeline():
    def __init__(self,
                 best_3d="/home/diedre/diedre_phd/phd/models/wd_step_poly_lung_softm-epoch=85-val_loss=0.03.ckpt",  # POLY LUNG SOFTM
                 best_25d="/home/diedre/diedre_phd/phd/models/wd_step_sme2d_coedet_fiso-epoch=72-val_loss=0.06-bg_dice=1.00-healthy_dice=0.92-unhealthy_dice=0.74.ckpt",  # SME2D COEDET FISO Full train wd step  better
                 best_25d_raw="/home/diedre/diedre_phd/phd/models/sing_a100_up_awd_step_raw_medseg_pos-epoch=99-val_loss=0.15-bg_dice=1.00-healthy_dice=0.84-unhealthy_dice=0.71.ckpt",  # Best raw axial 2.5D model, trained on positive 256 slices only
                 airway="/home/diedre/diedre_phd/phd/models/atm_baseline-epoch=75-val_loss=0.10-bg_dice=0.00-healthy_dice=0.00-unhealthy_dice=0.90.ckpt",  # Air way model used in ATM22 challenge
                 parse="/home/diedre/diedre_phd/phd/models/parse_baseline-epoch=103-val_loss=0.26-bg_dice=0.00-healthy_dice=0.00-unhealthy_dice=0.74.ckpt",  # First attempt at training in parse data
                 batch_size=4,
                 n=10,
                 cpu=False):  
        self.version = 'v1.2_wd_step_atm'
        self.batch_size = batch_size
        self.n = n
        self.device = torch.device("cpu") if cpu else torch.device("cuda:0")
        self.model_3d = PolySeg3DModule.load_from_checkpoint(best_3d).eval()
        if best_25d is None:
            self.model_25d = None
        else:
            self.model_25d = Seg2DModule.load_from_checkpoint(best_25d).eval()
        self.model_25d_raw = Seg2DModule.load_from_checkpoint(best_25d_raw).eval()
        self.airway_model = Seg2DModule.load_from_checkpoint(airway).eval()
        self.parse_model = Seg2DModule.load_from_checkpoint(parse).eval()
        self.hparams = self.model_25d_raw.hparams
        self.hparams.experiment_name = '_'.join([self.model_3d.hparams.experiment_name,
                                                 '' if self.model_25d is None else self.model_25d.hparams.experiment_name,
                                                 self.model_25d_raw.hparams.experiment_name])

    def __call__(self, input_volume, spacing, tqdm_iter, minimum_return=False, atm_mode=False, act=False):
        assert input_volume.max() <= 1.0 and input_volume.min() >= 0.0
        
        if tqdm_iter is not None and not isinstance(tqdm_iter, tqdm):
            tqdm_iter = PrintInterface(tqdm_iter)

        with torch.no_grad():
            # Airway segmentation
            tqdm_iter.write("Airway and parse segmentation with ATM and Parse 22 model...")
            tqdm_iter.progress(10)
            pre_shape = input_volume.shape[2:]
            if pre_shape[1] != 512 or pre_shape[2] != 512:
                adjust_shape = (pre_shape[0], 512, 512)
                tqdm_iter.write(f"Unusual shape {pre_shape} being transformed to {adjust_shape}")
                adjusted_volume = F.interpolate(input_volume.to(self.device), adjust_shape, mode="trilinear")
                adjusted_volume = adjusted_volume.cpu()
            else:
                adjust_shape = None
                adjusted_volume = input_volume
            output_p = stack_predict(self.parse_model.to(self.device), adjusted_volume, self.batch_size, extended_2d=1, get_uncertainty=None, device=self.device, info_q=tqdm_iter).squeeze()
            self.parse_model.cpu()
            output_a = stack_predict(self.airway_model.to(self.device), adjusted_volume, self.batch_size, extended_2d=1, get_uncertainty=None, device=self.device, info_q=tqdm_iter).squeeze()
            self.airway_model.cpu()
            if adjust_shape is not None:
                a_shape = output_a.shape
                tqdm_iter.write(f"Airway output {a_shape} being transformed back to {pre_shape}")
                output_a = F.interpolate(torch.from_numpy(output_a).unsqueeze(0).unsqueeze(0).to(self.device), pre_shape, mode="nearest").squeeze().cpu().numpy()
                p_shape = output_p.shape
                tqdm_iter.write(f"Parse output {p_shape} being transformed back to {pre_shape}")
                output_p = F.interpolate(torch.from_numpy(output_p).unsqueeze(0).unsqueeze(0).to(self.device), pre_shape, mode="nearest").squeeze().cpu().numpy()
            
            if not atm_mode:
                # 3D Lung detection
                tqdm_iter.write("3D Prediction...")
                tqdm_iter.progress(20)
                self.model_3d = self.model_3d.to(self.device)
                input_volume = input_volume.to(self.device)
                left_right_label = self.model_3d(F.interpolate(input_volume, (128, 256, 256), mode="trilinear"), get_bg=True)[0]
                self.model_3d.cpu()
                input_volume = input_volume.cpu()
                left_right_label = F.interpolate(left_right_label, pre_shape, mode="nearest").squeeze().cpu().numpy()
                
                left_lung, right_lung = left_right_label[1], left_right_label[2]
                voxvol = spacing[0]*spacing[1]*spacing[2]
                left_lung_volume = round((left_lung.sum()*voxvol)/1e+6, 3)
                right_lung_volume = round((right_lung.sum()*voxvol)/1e+6, 3)
                lung_volume = left_lung_volume + right_lung_volume

                tqdm_iter.write(f"Lung volume: {lung_volume}L, left: {left_lung_volume}L, right: {right_lung_volume}L")
                
                if lung_volume < .5:
                    # Lung not detected!
                    tqdm_iter.write("ERROR: Lung doesn't seen to be present in image! Aborting.")
                    output_f = output_f_sm_consensus = atts = None
                else:
                    # Lung detected
                    if self.n is not None:
                        print("Computing attention")
                        tqdm_iter.write("Computing attention...")
                        atts = get_atts_work(self.model_3d.model.enc, pre_shape)
                    else:
                        print("Skipping attention computation")
                        tqdm_iter.write("Skipping attention computation")
                        atts = None
                    
                    tqdm_iter.progress(30)
                        
                    # 2.5D RAW
                    tqdm_iter.write("2.5D prediction...")
                    tqdm_iter.progress(40)
                    if self.n is None:
                        output_f = stack_predict(self.model_25d_raw.to(self.device), adjusted_volume, self.batch_size, extended_2d=1, get_uncertainty=self.n, device=self.device, info_q=tqdm_iter)[0]
                    else:
                        output_f, epistemic_uncertainties, mean_predictions = stack_predict(self.model_25d_raw.to(self.device), adjusted_volume, self.batch_size, extended_2d=1, get_uncertainty=self.n, device=self.device, info_q=tqdm_iter)
                    self.model_25d_raw.cpu()
                    adjusted_volume = adjusted_volume.cpu()
                    if adjust_shape is not None:
                        f_shape = output_f.shape[1:]
                        tqdm_iter.write(f"2.5D Findings output {f_shape} being transformed back to {pre_shape}")
                        output_f = F.interpolate(torch.from_numpy(output_f).unsqueeze(0).to(self.device), pre_shape, mode="nearest").squeeze().cpu().numpy()
                    
                    if self.model_25d is not None:
                        # Isometric single model 2.5D consensus
                        tqdm_iter.write("2.5D Single model isometric consensus prediction...")
                        tqdm_iter.progress(60)

                        if isinstance(spacing[0], torch.Tensor):
                            spacing = np.array([x.item() for x in spacing])
                        else:
                            spacing = np.array(spacing)
                        
                        dest_shape = (spacing*pre_shape).astype(int).tolist()
                        print(spacing, pre_shape, dest_shape)
                        input_volume.to(self.device)
                        input_iso = F.interpolate(input_volume, size=dest_shape, mode="trilinear", align_corners=False).cpu()
                        input_volume.cpu()
                        orientations = [input_iso, 
                                        input_iso.permute(0, 1, 3, 2, 4), 
                                        input_iso.permute(0, 1, 4, 2, 3)]
                        
                        output_f_sm_consensus = multi_view_consensus([self.model_25d for _ in range(3)],
                                                                      orientations=orientations,
                                                                      tqdm_iter=None,
                                                                      batch_size=self.batch_size,
                                                                      extended_2d=1,
                                                                      device=self.device,
                                                                      info_q=tqdm_iter)
                        tqdm_iter.icon()

                        output_f_sm_consensus = F.interpolate(torch.from_numpy(output_f_sm_consensus).to(self.device), pre_shape, mode="nearest").squeeze().cpu().numpy()
                    else:
                        tqdm_iter.write("Skipping 2.5D single model isometric consensus prediction due to long prediction button unchecked...")
                        tqdm_iter.progress(60)
                        tqdm_iter.icon()
                        output_f_sm_consensus = None
                    del input_volume
        
        print(f"Airway max activation {output_a.max()}")
        airway = get_connected_components((output_a > 0.5).astype(np.int32), return_largest=1)[0].astype(np.int16)  # same type as atm mask
        print(f"Final airway max activation {airway.max()}")

        print(f"Parse max activation {output_p.max()}")
        parse = get_connected_components((output_p > 0.5).astype(np.int32), return_largest=1)[0].astype(np.int16)
        print(f"Final parse max activation {parse.max()}")
        
        tqdm_iter.write("Post-processing...")
        tqdm_iter.progress(80)
        if not atm_mode:
            # Build lung and findings consensus
            lung = left_lung + right_lung
            print(f"Lung max activation: {lung.max()}")
            if output_f is not None:         
                print("Adding lung segmentation from 2.5D")           
                lung += output_f[1] + output_f[2]
                print(f"Lung max activation: {lung.max()}")
                if output_f_sm_consensus is not None:
                    print("Adding lung segmentation from SMC 2.5D")
                    lung += output_f_sm_consensus[1] + output_f_sm_consensus[2]
                    print(f"Lung max activation: {lung.max()}")
                    lung = lung/3
                else:
                    lung = lung/2
                print(f"Lung max activation after consensus: {lung.max()}")
            
            findings = np.zeros_like(lung)
            if output_f is not None:
                print("Adding findings from 2.5D")
                findings = output_f[2]            
                print(f"Findings max activation {findings.max()}")
                if output_f_sm_consensus is not None:
                    print("Adding findings from SMC 2.5D")
                    findings += output_f_sm_consensus[2]
                    print(f"Findings max activation {findings.max()}")
                    findings = findings/2
                    print(f"Findings max activation after consensus: {findings.max()}")
            
            # If return of activations requested
            if act:
                ensemble_consensus = np.zeros((3,) + pre_shape)
                ensemble_consensus[0] = (left_right_label[0] + output_f[0] + output_f_sm_consensus[0])/3  # bg
                ensemble_consensus[1] = lung - findings  # healthy
                ensemble_consensus[2] = findings  # unhealthy
            else: 
                ensemble_consensus = None

            # Binarize
            lung, findings = (lung > 0.5).astype(np.int32), (findings > 0.5).astype(np.int32)
            lung, _, _ = get_connected_components(lung, return_largest=2)
            lung = lung.astype(np.uint8)
            findings = findings.astype(np.uint8)

            # Filter by lung region
            findings = findings*lung
            if self.n is not None:
                epistemic_uncertainties = epistemic_uncertainties*(np.expand_dims(lung, axis=0))
                mean_predictions = mean_predictions*(np.expand_dims(lung, axis=0))
            else:
                epistemic_uncertainties, mean_predictions = None, None

            inverse_lung = np.ones_like(lung) - lung
            if act:
                ensemble_consensus[0] = ensemble_consensus[0] * inverse_lung
                ensemble_consensus[1] = ensemble_consensus[1] * lung
                ensemble_consensus[2] = ensemble_consensus[2] * lung

            left_right_label = left_right_label.argmax(axis=0).astype(np.uint8)
            print(f"Final lung max: {lung.max()}, findings max: {findings.max()}")
        torch.cuda.empty_cache()
        if atm_mode:
            return airway
        elif minimum_return:
            return ensemble_consensus
        else:
            return ensemble_consensus, left_right_label, lung, findings, atts, epistemic_uncertainties, mean_predictions, left_lung_volume, right_lung_volume, airway, parse


def get_atts_work(encoder, pre_shape):
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

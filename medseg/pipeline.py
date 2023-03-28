'''
Main processing pipeline.
'''
import os
import cv2 as cv
import datetime
import numpy as np
import subprocess
import torch
import site
import pandas
from typing import List
import SimpleITK as sitk
from collections import defaultdict

# V1
from medseg.utils import monitor_itksnap, multi_channel_zoom
from medseg.postprocess import post_processing
from medseg.lightning_module import MEDSegModule
from medseg.image_read import SliceDataset, read_preprocess

# V2
from medseg.seg_pipeline import SegmentationPipeline
from medseg.utils import MultiViewer


def pipeline(model_path: str, 
             runlist: List[str], 
             batch_size: int, 
             output_path: str, 
             display: bool, 
             info_q, 
             cpu: bool, 
             windows_itksnap_path: str, 
             linux_itksnap_path: str, 
             debug: bool, 
             long: bool, 
             atm_mode: bool, 
             act: bool,
             use_path_as_ID: bool):
    if debug:
        pkg_path = 'medseg'
    else:
        pkg_path = os.path.join(site.getsitepackages()[int((os.name=="nt")*1)], "medseg")
    if model_path == "best_models":
        new_pipeline = True
    else:
        new_pipeline = False
    
    assert len(runlist) > 0, "No file found on given input path."
    
    if cpu:
        gpu_available = False
    else:
        gpu_available = torch.cuda.is_available()
    if new_pipeline:
        model = SegmentationPipeline(best_3d=os.path.join(pkg_path, "poly_lung.ckpt"),
                                     best_25d=os.path.join(pkg_path, "sme2d_coedet_fiso.ckpt") if long else None,
                                     best_25d_raw=os.path.join(pkg_path, "sing_a100_up_awd_step_raw_medseg_pos.ckpt"),
                                     airway=os.path.join(pkg_path, "airway.ckpt"),
                                     parse=os.path.join(pkg_path, "parse.ckpt"),
                                     batch_size=batch_size,
                                     cpu=cpu,
                                     n=None)  # uncertainty disabled for now
    else:
        model = MEDSegModule.load_from_checkpoint(os.path.join(pkg_path, "best_coedet.ckpt")).eval()

        if gpu_available:
            model = model.cuda()

    info_q.put(("write", "Succesfully initialized network"))
    runlist_len = len(runlist)
    output_csv = defaultdict(list)
    for i, run in enumerate(runlist):
        info_q.put(("write", f"Loading and Pre-processing {run}..."))
       
        if new_pipeline:
            # Use seg_pipeline here
            data, original_shape, origin, spacing, directions, original_image = read_preprocess(run)
            dir_array = np.asarray(directions)
            if atm_mode:
                airway = model(input_volume=data.unsqueeze(0).unsqueeze(0), spacing=spacing, tqdm_iter=info_q, minimum_return=False, atm_mode=atm_mode)
            else:
                ensemble_consensus, left_right_label, lung, covid, _, _, _, left_lung_volume, right_lung_volume, airway, parse = model(input_volume=data.unsqueeze(0).unsqueeze(0), spacing=spacing, tqdm_iter=info_q, minimum_return=False, act=act)
            info_q.put(("iterbar", 90))
        else:
            raise DeprecationWarning("There is only new pipeline. Old pipeline has been deprecated in version")
        
        # Statistics
        if isinstance(run, list):
            ID = os.path.basename(os.path.dirname(run[0]))
        else:
            if use_path_as_ID:
                ID = '_'.join(run.replace(".nii", '').replace(".gz", '').split(os.sep)[-1:-4:-1][::-1])
            else:
                ID = os.path.basename(run).replace(".nii", '').replace(".gz", '')

        if not atm_mode:
            voxvol = spacing[0]*spacing[1]*spacing[2]
            airway_volume = round(airway.sum()*voxvol/1e+6, 3)
            parse_volume = round(parse.sum()*voxvol/1e+6, 3)

            try:
                lung_ocupation = round((covid.sum()/lung.sum())*100, 2)
            except:
                lung_ocupation = None
            
            try:
                left_f_v = round((covid*(left_right_label == 1)).sum()*voxvol/1e+6, 3)
            except:
                left_f_v = None
            
            try:
                right_f_v = round((covid*(left_right_label == 2)).sum()*voxvol/1e+6, 3)
            except:
                right_f_v = None

            try:
                f_v = round(covid.sum()*voxvol/1e+6, 3)
            except:
                f_v = None

            try:
                l_o = round(left_f_v*100/left_lung_volume, 2)
            except:
                l_o = None

            try:
                r_o = round(right_f_v*100/right_lung_volume, 2)
            except:
                r_o = None

            output_csv["Path"].append(run)
            output_csv["ID"].append(ID)
            output_csv["Lung Volume (L)"].append(left_lung_volume + right_lung_volume)
            output_csv["Left Lung Volume (L)"].append(left_lung_volume)
            output_csv["Right Lung Volume (L)"].append(right_lung_volume)
            output_csv["Airway Volume (L)"].append(airway_volume)
            output_csv["Vessel Volume (L)"].append(parse_volume)
            output_csv["Lung Findings Volume (L)"].append(f_v)
            output_csv["Left Lung Findings Volume (L)"].append(left_f_v)
            output_csv["Right Lung Findings Volume (L)"].append(right_f_v)
            output_csv["Occupation (%)"].append(lung_ocupation)
            output_csv["Left Occupation (%)"].append(l_o)
            output_csv["Right Occupation (%)"].append(r_o)
            output_csv["Voxel volume (mm³)"].append(voxvol)

        # Undo flips
        if len(dir_array) == 9:
            airway = np.flip(airway, np.where(dir_array[[0,4,8]][::-1]<0)[0]).copy()
            parse = np.flip(parse, np.where(dir_array[[0,4,8]][::-1]<0)[0]).copy()
            if not atm_mode:
                lung = np.flip(lung, np.where(dir_array[[0,4,8]][::-1]<0)[0]).copy()
                covid = np.flip(covid, np.where(dir_array[[0,4,8]][::-1]<0)[0]).copy()

        if not atm_mode:
            merged = lung + covid
            merged[airway==1] = 3
            merged[parse==1] = 4
        
        # Create save paths
        # output_input_path = os.path.join(output_path, ID + "_input.nii.gz")
        output_lung_path = os.path.join(output_path, ID + "_lung.nii.gz")
        output_lr_lung_path = os.path.join(output_path, ID + "_lr_lung.nii.gz")
        output_findings_path = os.path.join(output_path, ID + "_findings.nii.gz")
        output_merged_path = os.path.join(output_path, ID + "_all_segmentations.nii.gz")
        if atm_mode:
            output_airway = os.path.join(output_path, ID + ".nii.gz")
        else:
            output_airway = os.path.join(output_path, ID + "_airway.nii.gz")
        output_parse = os.path.join(output_path, ID + "_vessel.nii.gz")

        # Create images
        # Reverting spacing back for saving
        spacing = spacing[::-1]
        writer = sitk.ImageFileWriter()

        # Save airway image
        airway_image = sitk.GetImageFromArray(airway)
        airway_image.SetDirection(directions)
        airway_image.SetOrigin(origin)
        airway_image.SetSpacing(spacing)
        writer.SetFileName(output_airway)
        writer.Execute(airway_image)

        if not atm_mode:
            # Save original image removed, too much space usage for big images
            # writer.SetFileName(output_input_path)
            # writer.Execute(original_image)
            
            # Save lung image
            lung_image = sitk.GetImageFromArray(lung)
            lung_image.SetDirection(directions)
            lung_image.SetOrigin(origin)
            lung_image.SetSpacing(spacing)
            writer.SetFileName(output_lung_path)
            writer.Execute(lung_image)

            # Save parse image
            parse_image = sitk.GetImageFromArray(parse)
            parse_image.SetDirection(directions)
            parse_image.SetOrigin(origin)
            parse_image.SetSpacing(spacing)
            writer.SetFileName(output_parse)
            writer.Execute(parse_image)

            # Save lr lung image
            lr_lung_image = sitk.GetImageFromArray(left_right_label)
            lr_lung_image.SetDirection(directions)
            lr_lung_image.SetOrigin(origin)
            lr_lung_image.SetSpacing(spacing)
            writer.SetFileName(output_lr_lung_path)
            writer.Execute(lr_lung_image)

            # Save findings image
            covid_image = sitk.GetImageFromArray(covid)
            covid_image.SetDirection(directions)
            covid_image.SetOrigin(origin)
            covid_image.SetSpacing(spacing)
            writer.SetFileName(output_findings_path)
            writer.Execute(covid_image)

            # Save lung and findings image
            merged_image = sitk.GetImageFromArray(merged)
            merged_image.SetDirection(directions)
            merged_image.SetOrigin(origin)
            merged_image.SetSpacing(spacing)
            writer.SetFileName(output_merged_path)
            writer.Execute(merged_image)

        info_q.put(("iterbar", 100))
        info_q.put(("write", f"Processing finished {run}."))

        if display:
            # ITKSnap
            info_q.put(("write", "Displaying results with itksnap.\nClose itksnap windows to continue."))
            try:
                itksnap_output_path = output_airway if atm_mode else output_merged_path
                if os.name == "nt":
                    subprocess.Popen([windows_itksnap_path, "-g", run, "-s", itksnap_output_path])
                else:
                    subprocess.Popen([linux_itksnap_path, "-g", run, "-s", itksnap_output_path])
                
            except Exception as e:
                info_q.put(("write", "Error displaying results. Do you have itksnap installed?"))
                print(e)

        # Proprietary multiviewer
        if act:    
            try:
                multi_viewer = MultiViewer(np.concatenate((data.unsqueeze(0), ensemble_consensus)), left_right_label, window_name="Activations visualization (press numbers and S to navigate)", threaded=False, cube_side=256)
                multi_viewer.display(channel_select=0)
                cv.destroyAllWindows()
            except Exception as e:
                print(f"Couldn't visualize with multiviewer: {e}")
            monitor_itksnap()
            
        info_q.put(("generalbar", (100*i+100)/runlist_len))
        info_q.put(("write", f"{i+1} volumes processed out of {runlist_len}.\nResult are on the {output_path} folder."))
    uid = str(datetime.datetime.today()).replace('.', '').replace(':', '').replace('-', '').replace(' ', '')

    output_csv_path = os.path.join(output_path, f"medseg_run_statistics_{uid}.csv")
    pandas.DataFrame.from_dict(output_csv).to_csv(output_csv_path)
    info_q.put(("write", f"Sheet with pulmonary involvement statistics saved in {output_csv_path}."))
    info_q.put(None)

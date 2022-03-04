'''
Main processing pipeline.
'''
import os
import datetime
import numpy as np
import subprocess
import torch
import site
import pandas
from typing import List
import SimpleITK as sitk
from collections import defaultdict

from coedet.utils import monitor_itksnap, multi_channel_zoom
from coedet.postprocess import post_processing
from coedet.lightning_module import CoEDetModule
from coedet.image_read import SliceDataset


def pipeline(model_path: str, runlist: List[str], batch_size: int, output_path: str, display: bool, info_q, cpu: bool, 
             windows_itksnap_path: str, linux_itksnap_path: str):
    if os.name == "nt":
        model_path = os.path.join(site.getsitepackages()[1], "coedet", model_path)
    else:
        model_path = os.path.join(site.getsitepackages()[0], "coedet", model_path)
    
    assert len(runlist) > 0, "No file found on given input path."
    assert os.path.isfile(model_path), f"Couldn't find a model in {model_path}."
    
    if cpu:
        gpu_available = False
    else:
        gpu_available = torch.cuda.is_available()
    
    model = CoEDetModule.load_from_checkpoint(model_path).eval()
    if gpu_available:
        model = model.cuda()

    info_q.put(("write", "Succesfully initialized network"))
    runlist_len = len(runlist)
    output_csv = defaultdict(list)
    for i, run in enumerate(runlist):
        info_q.put(("write", f"Loading and Pre-processing {run}..."))
        slice_dataset = SliceDataset(run)
    
        directions = slice_dataset.directions
        dir_array = np.asarray(directions)
        spacing = slice_dataset.spacing
        origin = slice_dataset.origin
        original_image = slice_dataset.read_image()

        slice_dataloader = slice_dataset.get_dataloader(batch_size)
        info_q.put(("iterbar", 20))
        info_q.put(("write", "Predicting..."))
        output = []
        for s, batch in enumerate(slice_dataloader):
            info_q.put(("iterbar", 20 + s*(60/len(slice_dataset))))
            package = batch[0].squeeze().numpy().copy()
            info_q.put(("slice", package))

            if gpu_available:
                batch = batch.cuda()

            with torch.no_grad():
                output.append(model(batch).detach().cpu())
        info_q.put(("icon", ''))
            
        info_q.put(("write", "Post-processing..."))
        cpu_output = torch.stack(output, dim=0)
        N, B, C, H, W = cpu_output.shape
        cpu_output = cpu_output.reshape(N*B, C, H, W).permute(1, 0, 2, 3).numpy()
        
        original_shape = slice_dataset.original_shape
        
        output_shape = cpu_output[0].shape
        print(f"Original shape: {original_shape}")
        print(f"Network output shape: {output_shape}")
        if original_shape != output_shape:
            print("Need interpolation to original shape.")
            zoom_factor = np.array(original_shape)/np.array(output_shape)
            cpu_output = multi_channel_zoom(cpu_output, zoom_factor, order=3, threaded=False, tqdm_on=False)
        info_q.put(("iterbar", 85))
        lung, covid = post_processing(cpu_output)
        info_q.put(("iterbar", 90))

        lung_ocupation = round((covid.sum()/lung.sum())*100)
        output_csv["ID"].append(os.path.basename(run).split(".nii")[0])
        output_csv["Occupation"].append(f"{lung_ocupation} %")

        # Undo flips
        if len(dir_array) == 9:
            lung = np.flip(lung, np.where(dir_array[[0,4,8]][::-1]<0)[0]).copy()
            covid = np.flip(covid, np.where(dir_array[[0,4,8]][::-1]<0)[0]).copy()

        merged = lung + covid
        
        # Create save paths
        output_input_path = os.path.join(output_path, os.path.basename(run).replace(".nii", "_input.nii"))
        output_lung_path = os.path.join(output_path, os.path.basename(run).replace(".nii", "_lung.nii"))
        output_findings_path = os.path.join(output_path, os.path.basename(run).replace(".nii", "_findings.nii"))
        output_merged_path = os.path.join(output_path, os.path.basename(run).replace(".nii", "_lung_and_findings.nii"))

        # Create images
        writer = sitk.ImageFileWriter()

        # Save original image
        writer.SetFileName(output_input_path)
        writer.Execute(original_image)
        
        # Save lung image
        lung_image = sitk.GetImageFromArray(lung)
        lung_image.SetDirection(directions)
        lung_image.SetOrigin(origin)
        lung_image.SetSpacing(spacing)
        writer.SetFileName(output_lung_path)
        writer.Execute(lung_image)

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
            info_q.put(("write", "Displaying results with itksnap.\nClose itksnap windows to continue."))
            try:
                if os.name == "nt":
                    subprocess.run([windows_itksnap_path, "-g", output_input_path, "-s", output_merged_path])
                else:
                    subprocess.run([linux_itksnap_path, "-g", output_input_path, "-s", output_merged_path])
                monitor_itksnap()
            except Exception as e:
                info_q.put(("write", "Error displaying results. Do you have itksnap installed?"))
                print(e)
        info_q.put(("generalbar", (100*i+100)/runlist_len))
        info_q.put(("write", f"{i+1} volumes processed out of {runlist_len}.\nResult are on the {output_path} folder."))
    uid = str(datetime.datetime.today()).replace('.', '').replace(':', '').replace('-', '').replace(' ', '')

    output_csv_path = os.path.join(output_path, f"coedet_run_statistics_{uid}.csv")
    pandas.DataFrame.from_dict(output_csv).to_csv(output_csv_path)
    info_q.put(None)
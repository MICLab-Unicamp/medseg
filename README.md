# Covid EfficientDet (COEdet)
Official repository for reproducing COVID segmentation prediction using our CoEDet model.

The publication for this method, **Multitasking segmentation of lung and COVID-19 findings in CT scans using modified EfficientDet, UNet and MobileNetV3 models**, has been published at the 17th International Symposium on Medical Information Processing and Analysis (SIPAIM 2021).
http://dx.doi.org/10.1117/12.2606118

The presentation can be watched in YouTube:
https://www.youtube.com/watch?v=PlhNUD0Y4hg

## Citation
* Carmo, Diedre, et al. "Multitasking segmentation of lung and COVID-19 findings in CT scans using modified EfficientDet, UNet and MobileNetV3 models." 17th International Symposium on Medical Information Processing and Analysis. Vol. 12088. SPIE, 2021.

* @inproceedings{carmo2021multitasking,\
  title={Multitasking segmentation of lung and COVID-19 findings in CT scans using modified EfficientDet, UNet and MobileNetV3 models},\
  author={Carmo, Diedre and Campiotti, Israel and Fantini, Irene and Rodrigues, L{\'\i}via and Rittner, Let{\'\i}cia and Lotufo, Roberto},\
  booktitle={17th International Symposium on Medical Information Processing and Analysis},\
  volume={12088},\
  pages={65--74},\
  year={2021},\
  organization={SPIE}\
}

## Requirement: ITKSnap

We use itksnap for visualization. 

### Linux/Ubuntu

You can easily install itksnap in Ubuntu with.

    sudo apt install itksnap

### Windows

For windows, install using the self-install package in: http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3. In this case we assume the itksnap executable will be in "C:\\Program Files\\ITK-SNAP 3.8\\bin\\ITK-SNAP.exe". If this is different for you, change the "windows_itksnap_path" variable in the config.json file.

## Pre-built binaries

We will make those available soon.

## Requirements and running from source code

This tool was tested on Ubuntu 20.04 and Windows 10. The following instructions refer to quickly running the tool directly from code.

We recommend using a Miniconda environment. To install Miniconda for Windows or Linux follow the instructions in: https://docs.conda.io/en/latest/miniconda.html

If you are on windows. All following commands should be executed in Anaconda Prompt (installed with miniconda).

Now, install torch with CUDA GPU support following the instructions for your OS on http://www.pytorch.org/. You can also install without cuda support if you dont have a GPU.

All additional required libraries are contained on the requirements.txt file and can be installed with pip. They are:

numpy\
pillow\
scipy\
tqdm\
torchvision\
pytorch-lightning==1.3.8\
efficientnet_pytorch\
connected-components-3d\
psutil\
gputil\
SimpleITK==2.0.2\

The command for installing these libraries is: 

    pip install -r requirements.txt

## Running 

To run from code, just run the run.py file with your miniconda python. 

    python run.py

If you don't want to use a GPU, run with the --cpu flag.

    python run.py --cpu

## How to train?

This code is only intended to allow reproduction of the segmentation capabilities of our work. 
However, we provide the complete Lightning Module code (in lightning_module.py) which can be used under your workflow and data for training if you use PyTorch Lightning.

## Issues?

If you have any problems, make sure your pip is the same from your miniconda installation,
by checking if pip --version points to the miniconda directory. Also check if the python you are using is the miniconda one.

If you have any issues, feel free to create an issue on this repository.
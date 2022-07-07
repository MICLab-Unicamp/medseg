# Modified EfficientDet Segmentation (MEDSeg)
Official repository for reproducing COVID segmentation prediction using our MEDSeg model.

The publication for this method, **Multitasking segmentation of lung and COVID-19 findings in CT scans using modified EfficientDet, UNet and MobileNetV3 models**, has been published at the 17th International Symposium on Medical Information Processing and Analysis (SIPAIM 2021), and won the "SIPAIM Society Award".
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

## Requirements

This tool was tested on Ubuntu 20.04 and Windows 10. The following instructions refer to quickly running the tool installing it with Miniconda and pip.

### ITKSnap

We use itksnap for visualization. 

You can easily install itksnap in Ubuntu with.

    sudo apt install itksnap

For windows, install using the self-install package in: http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3. In this case we assume the itksnap executable will be in "C:\\Program Files\\ITK-SNAP 3.8\\bin\\ITK-SNAP.exe". If this is different for you, change the "win_itk_path" argument when running.

### Miniconda

We recommend using a Miniconda environment for installation. To install Miniconda for Windows or Linux follow the instructions in: https://docs.conda.io/en/latest/miniconda.html. If you are on windows. All following commands should be executed in Anaconda Prompt (bundled with miniconda).

### GPU Usage: PyTorch

Now, install torch with CUDA GPU support. It can be easily installed with a single command depending on your environment. 
Follow the instructions for your OS on http://www.pytorch.org/. 

If you don't want to use a GPU, you can skip this part and trust the automatic installation of dependencies.

## Installation

All additional required libraries and the tool itself will be installed with the following steps:

    git clone https://github.com/MICLab-Unicamp/coedet
    cd coedet
    pip install .
    
If you use virtual environments, it is safer to install in a new virtual environment to avoid conflicts.

## Running 

To run, just call coedet in a terminal.

    coedet

If you don't want to use a GPU, run with the --cpu flag.

    coedet_cpu

If your ITKSNap installation is not on the assumed default locations, you can change where the tool will look for it. Check the --help command for more details.

    coedet --help

## How to train?

This code is only intended to allow reproduction of the segmentation capabilities of our work. 
However, we provide the complete Lightning Module code (in lightning_module.py) which can be used under your workflow and data for training if you use PyTorch Lightning.

## Issues?

If you have any problems, make sure your pip is the same from your miniconda installation,
by checking if pip --version points to the miniconda directory.

If you have any issues, feel free to create an issue on this repository.

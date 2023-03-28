# Modified EfficientDet Segmentation (MEDSeg)
Official repository for reproducing lung, COVID-19, airway and pulmonary artery automated segmentation using our MEDSeg model.

The publication original publication for this method, **Multitasking segmentation of lung and COVID-19 findings in CT scans using modified EfficientDet, UNet and MobileNetV3 models**, has been published at the 17th International Symposium on Medical Information Processing and Analysis (SIPAIM 2021), and won the "SIPAIM Society Award".
http://dx.doi.org/10.1117/12.2606118

The presentation can be watched in YouTube:
https://www.youtube.com/watch?v=PlhNUD0Y4hg

We have also applied this model in the ATM22 Challenge (https://atm22.grand-challenge.org/). Airway segmentation is included, with a CLI argument (--atm_mode) to only segment the airway, using less memory. A short paper about this is published in arXiv **Open-source tool for Airway Segmentation in
Computed Tomography using 2.5D Modified EfficientDet: Contribution to the ATM22 Challenge**: https://arxiv.org/pdf/2209.15094.pdf

We have also trained this model to the PARSE Challenge (https://parse2022.grand-challenge.org/), (Pulmonary Artery segmentation). Pulmonary artery labels will be included in the outputs. The model achieved around 0.7 Dice in testing. An paper detailing this application will be published in the future. 

## Citation
* **COVID-19 segmentation and method in general**: Carmo, Diedre, et al. "Multitasking segmentation of lung and COVID-19 findings in CT scans using modified EfficientDet, UNet and MobileNetV3 models." 17th International Symposium on Medical Information Processing and Analysis. Vol. 12088. SPIE, 2021.

    * @inproceedings{carmo2021multitasking,\
  title={Multitasking segmentation of lung and COVID-19 findings in CT scans using modified EfficientDet, UNet and MobileNetV3 models},\
  author={Carmo, Diedre and Campiotti, Israel and Fantini, Irene and Rodrigues, L{\'\i}via and Rittner, Let{\'\i}cia and Lotufo, Roberto},\
  booktitle={17th International Symposium on Medical Information Processing and Analysis},\
  volume={12088},\
  pages={65--74},\
  year={2021},\
  organization={SPIE}\
}

* **Airway segmentation**: Carmo, Diedre, Leticia Rittner, and Roberto Lotufo. "Open-source tool for Airway Segmentation in Computed Tomography using 2.5 D Modified EfficientDet: Contribution to the ATM22 Challenge." arXiv preprint arXiv:2209.15094 (2022).

    * @article{carmo2022open,
  title={Open-source tool for Airway Segmentation in Computed Tomography using 2.5 D Modified EfficientDet: Contribution to the ATM22 Challenge},
  author={Carmo, Diedre and Rittner, Leticia and Lotufo, Roberto},
  journal={arXiv preprint arXiv:2209.15094},
  year={2022}
}


## Requirements

This tool was tested on Ubuntu 20.04 and Windows 10. The following instructions refer to quickly running the tool installing it with Miniconda and pip. Depending on the size of your input CT, you might need 32 GB of memory to run. I intend to reduce this memory requirement with future optimizations.

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

All additional required libraries and the tool itself will be installed with the following steps. First, clone the repository. Then, due to the large size of network weights, you need to go into the Release in this repository, download the [data.zip](https://github.com/MICLab-Unicamp/medseg/releases/download/v2.5/data.zip) file and extract the .ckpt files inside the medseg/medseg folder. The .ckpt files should be in the same directory level as the run.py file. Finally, go into the top level folder (with the setup.py file) and install the tool with "pip install . ". The following command lines should do all those steps:

    git clone https://github.com/MICLab-Unicamp/medseg
    cd medseg/medseg
    wget https://github.com/MICLab-Unicamp/medseg/releases/download/v2.5/data.zip
    unzip data.zip
    cd ..
    pip install .
    
The above commands require git, unzip and wget which can be installed in Ubuntu with 

    sudo apt install git wget unzip

If you don't have them on your system.

## Running 

To run, just call it in a terminal.

    medseg

If you don't want to use a GPU, run this command:

    medseg_cpu

If you don't want to use the GUI, give --input_folder and --output_folder arguments to run in a folder of exams. If your ITKSNap installation is not on the assumed default locations, you can change where the tool will look for it. Check the --help command for more details and help in general.

## How to train?

This code is only intended to allow reproduction of the segmentation capabilities of our work. 
However, we provide the complete Lightning Module code (in seg_2d_module.py) which can be used under your workflow and data for training if you use PyTorch Lightning.

## Issues?

If you have any problems, make sure your pip is the same from your miniconda installation,
by checking if pip --version points to the miniconda directory.

If you have any issues, feel free to create an issue on this repository.

### Known Issue

"Long prediction" mode is not working due to recent changes in the architecutre. However not using it should be enough for most cases, Long Prediction uses more models in the final ensemble. 

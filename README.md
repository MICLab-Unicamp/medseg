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

## Requirements and Installation

This was only tested on Ubuntu 20.04.
Python libraries required to run are the following: 

numpy\
pillow\
scipy\
tqdm\
torch\
torchvision\
pytorch-lightning==1.3.8\
efficientnet_pytorch\
connected-components-3d\
psutil\
gputil\
SimpleITK==2.0.2\

They are contained on the requirements.txt file and can be installed with a working pip. We recommend using a Miniconda environment (https://docs.conda.io/en/latest/miniconda.html).

    pip install -r requirements.txt

We recommend that you install torch with CUDA GPU support (follow the instructions on http://www.pytorch.org/), unless you plan to run in a CPU (also possible).

We use itksnap for visualization. You can easily install itksnap with.

    sudo apt install itksnap

Alternatively, you can run the install.sh script.
    
    sh install.sh

In the near future, we will make a standalone pre-compiled executable release available, for ease of use.

## Running 

To run, just run the run.sh script. Make it executable first:

     chmod +x run.sh
    
And run, no arguments are necessary since there will be a graphical user interface.

    ./run.sh

## How to train?

This code is only intended to allow reproduction of the segmentation capabilities of our work. 
However, we provide the complete Lightning Module code (in lightning_module.py) which can be used under your workflow and data for training if you use PyTorch Lightning.

## Issues?

If you have any issues, feel free to create an issue on this repository.

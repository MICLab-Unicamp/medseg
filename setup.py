import setuptools
from medseg import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

found = setuptools.find_packages()
print(f"Found these packages to add: {found}")

setuptools.setup(
    name="medseg",
    version=__version__,
    author="Diedre Carmo",
    author_email="diedre@dca.fee.unicamp.br",
    description="Modified EfficientDet published in: Multitasking segmentation of lung and COVID-19 findings in CT scans using modified EfficientDet, UNet and MobileNetV3 models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MICLab-Unicamp/medseg",
    packages=found,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['setuptools', 'numpy', 'rich', 'pillow', 'scipy', 'tqdm', 'torch', 'pandas', 'torchvision', 'pytorch-lightning', 'efficientnet_pytorch', 'connected-components-3d', 'psutil', 'gputil', 'opencv-python', 'SimpleITK==2.0.2', 'pydicom', 'matplotlib'],
    entry_points={
        'console_scripts': ["medseg = medseg.run:main", "medseg_cpu = medseg.run:main_cpu"]
    },
    include_package_data=True,
    package_data={'medseg': ["best_coedet.ckpt", "icon.png", "poly_lung.ckpt", "up_awd_step_raw_medseg_pos.ckpt", "sme2d_coedet_fiso.ckpt", "airway.ckpt"]}
)

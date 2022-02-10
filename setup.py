import os
from importlib_metadata import entry_points
import setuptools
from coedet import __version__

with open(os.path.join("coedet", "README.md"), "r") as fh:
    long_description = fh.read()

found = setuptools.find_packages()
print(f"Found these packages to add: {found}")

setuptools.setup(
    name="coedet",
    version=__version__,
    author="Diedre Carmo",
    author_email="diedre@dca.fee.unicamp.br",
    description="Modified EfficientDet published in: Multitasking segmentation of lung and COVID-19 findings in CT scans using modified EfficientDet, UNet and MobileNetV3 models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MICLab-Unicamp/coedet",
    packages=found,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['setuptools', 'numpy', 'rich', 'pillow', 'scipy', 'tqdm', 'torch', 'torchvision', 'pytorch-lightning', 'efficientnet_pytorch', 'connected-components-3d', 'psutil', 'gputil', 'SimpleITK==2.0.2'],
    entry_points={
        'console_scripts': ["coedet_gui = coedet.run:main"]
    },
    include_package_data=True,
    package_data={'coedet': ["best_coedet.ckpt", "icon.png"]}
)

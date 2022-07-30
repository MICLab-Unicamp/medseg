'''
Preprocessing and image reading functions
'''
import torch
import pydicom
import numpy as np
import SimpleITK as sitk
import multiprocessing as mp
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, InterpolationMode


def filter_dcm_list(path):
    initial_l = len(path)
    ps_dcms = [(p, pydicom.read_file(p)) for p in path]

    try:
        ps_dcms = sorted(ps_dcms, key=lambda s: s[1].SliceLocation)
    except AttributeError:
        pass

    ps_dcms_shapes = [(p, dcm, (dcm.Rows, dcm.Columns)) for p, dcm in ps_dcms]
    most_common_shape = Counter([shape for _, _, shape in ps_dcms_shapes]).most_common(1)[0][0]
    path = [p for p, _, dcm_shape in ps_dcms_shapes if dcm_shape == most_common_shape]
    path_diff = initial_l - len(path)
    if path_diff != 0:
        print(f"WARNING: {path_diff} slices removed due to misaligned shape.")

    return path

def read_preprocess(input_path):
    if isinstance(input_path, list):
        input_path = filter_dcm_list(input_path)[::-1]

    image = sitk.ReadImage(input_path)
    data = sitk.GetArrayFromImage(image)
    original_shape = data.shape
    
    directions = image.GetDirection()
    dir_array = np.asarray(directions)
    origin = image.GetOrigin()
    spacing = image.GetSpacing()[::-1]

    if len(dir_array) == 9:
        data = np.flip(data, np.where(dir_array[[0,4,8]][::-1]<0)[0]).copy()  # fix axial orientation for bed on the bottom, from lungmask

    # Pre processing
    data = np.clip(data, -1024, 600)
    data = (data - data.min()) / (data.max() - data.min())

    data = torch.from_numpy(data).float()

    return data, original_shape, origin, spacing, directions, image


class SliceDataset(Dataset):
    def __init__(self, input_path):
        '''
        Treat a volume (input_path) as a dataset of slices.

        input_path: input_path to main image
        '''
        super().__init__()
        self.input_path = input_path
        
        data, self.original_shape, self.origin, self.spacing, self.directions = read_preprocess(self.input_path)
        
        print(f"Directions: {self.directions}")
        print(f"Origin: {self.origin}")
        print(f"Spacing: {self.spacing}")
        data = data.unsqueeze(1)  # [Fatia, H, W] -> [Fatia, 1, H, W]
        
        self.data = data
        self.transform = Resize((512, 512), interpolation=InterpolationMode.BILINEAR)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        data = self.data[i]
        
        transformed_data = self.transform(data)
        
        return transformed_data 

    def get_header(self):
        return self.header

    def get_affine(self):
        return self.affine

    def read_image(self):
        return sitk.ReadImage(self.input_path)

    def get_dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=mp.cpu_count())

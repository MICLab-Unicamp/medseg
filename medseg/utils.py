'''
A collection of general utils extracted from DLPT v0.2.2.0
'''
import os
import sys
import time
import torch
import psutil
import cv2 as cv
import numpy as np
import SimpleITK as sitk
import multiprocessing as mp
from tqdm import tqdm
from torch import optim, nn
from scipy.ndimage import zoom


current_point = None


class DummyTkIntVar():
    def __init__(self, value):
        self.value = value

    def __bool__(self):
        return bool(self.value)

    def get(self):
        return self.value


def zoom_worker(x):
    channel, zoom_factor, order = x
    return zoom(channel, zoom_factor, order=order)


def check_params(hparams):
    '''
    Avoids errors while loading old models while needing new hparams.
    '''
    '''
    hparams.noschedule = getattr(hparams, "noschedule", False)
    hparams.batchfy = getattr(hparams, "batchfy", False)
    hparams.f1_average = getattr(hparams, "f1_average", 'macro')
    hparams.datasets = getattr(hparams, "datasets", 'all')
    hparams.eval_batch_size = getattr(hparams, "eval_batch_size", hparams.batch_size)
    hparams.return_path = getattr(hparams, "return_path", False)
    hparams.return_path = getattr(hparams, "patched_eval", False)
    hparams.bn = getattr(hparams, "bn", "group")
    hparams.wd = getattr(hparams, "wd", 0)
    if getattr(hparams, "noschedule", False):
        hparams.scheduling_factor = None
    '''

    return hparams


def monitor_itksnap(check_delay: int = 1):
    '''
    Blocks process checking with psutil if an itksnap instance is opened.
    Delay between checks is specified by check_delay
    '''
    itksnap_found = True
    while itksnap_found:
        process_list = [proc.name() for proc in psutil.process_iter()]
        itksnap_instances = ['itk-snap' in name.lower() for name in process_list]
        itksnap_found = any(itksnap_instances)
        if itksnap_found:
            print(' '*100, end='\r')
            time.sleep(check_delay/2)
            itk = process_list[itksnap_instances.index(True)]
            print(f"Waiting for {itk} to be closed.", end='\r')
            time.sleep(check_delay/2)


def multi_channel_zoom(full_volume, zoom_factors, order, C=None, tqdm_on=True, threaded=False):
    '''
    full_volume: Full 4D volume (numpy)
    zoom_factors: intented shape / current shape
    order: 0 - 5, higher is slower but better results, 0 is fast and bad results
    C: how many cores to spawn, defaults to number of channels in volume
    tqdm_on: verbose computation
    '''
    assert len(full_volume.shape) == 4 and isinstance(full_volume, np.ndarray)

    if C is None:
        C = full_volume.shape[0]

    if threaded:
        pool = mp.pool.ThreadPool(C)
    else:
        pool = mp.Pool(C)

    channels = [(channel, zoom_factors, order) for channel in full_volume]

    zoomed_volumes = []

    pool_iter = pool.map(zoom_worker, channels)

    if tqdm_on:
        iterator = tqdm(pool_iter, total=len(channels), desc="Computing zooms...")
    else:
        iterator = pool_iter

    for output in iterator:
        zoomed_volumes.append(output)

    return np.stack(zoomed_volumes)


class MultiViewer():
    def __init__(self, volume, mask=None, normalize=False, window_name="MultiViewer", cube_side=200, resize_factor=2, order=3,
                 threaded=False):
        if len(volume.shape) == 4 and np.argmin(volume.shape) == 3:
            print("Channel dimension has to be 0, attempting transpose")
            volume = volume.transpose(3, 0, 1, 2)
            assert np.argmin(volume.shape) == 0, "Couldn't solve wrong dimension channel. Put channel on dimension 0."

        original_min = volume.min()
        original_max = volume.max()

        if normalize:
            self.volume = (volume - volume.min()) / (volume.max() - volume.min())
        else:
            self.volume = volume

        self.volume_shape = volume.shape

        multichannel = len(self.volume_shape) > 3
        self.multichannel = multichannel

        if multichannel:
            self.C = self.volume_shape[0]
            self.last_channel = 0

        if self.volume_shape != (cube_side, cube_side, cube_side):
            zoom_factors = (cube_side/self.volume_shape[-3], cube_side/self.volume_shape[-2], cube_side/self.volume_shape[-1])
            mask_zoom = zoom_factors

            if multichannel:
                self.volume = multi_channel_zoom(self.volume, zoom_factors, order=order, threaded=threaded, tqdm_on=False)
            else:
                self.volume = zoom(self.volume, zoom_factors, order=order)

            # Reverse for proper orientation
            self.volume = np.flip(self.volume, axis=-3)

        if mask is not None:
            zoomed_mask = zoom(mask, mask_zoom, order=0).astype(np.float32)
            zoomed_mask = (zoomed_mask - zoomed_mask.min())/(zoomed_mask.max() - zoomed_mask.min())
            zoomed_mask = np.flip(zoomed_mask, axis=-3)
            self.masked_volume = np.where(zoomed_mask == 0, self.volume, zoomed_mask)
            self.original_volume = self.volume
            self.volume = self.masked_volume
            self.displaying_mask = True
            

        self.volume_shape = self.volume.shape
        assert self.volume_shape[-1:-4:-1][::-1] == (cube_side, cube_side, cube_side)

        self.current_point = (np.array(self.volume_shape[-1:-4:-1][::-1])/2).astype(np.int)
        self.window_name = window_name
        self.resize_factor = resize_factor

        get_current_window = np.concatenate((np.zeros(self.volume_shape[-3]),
                                             np.ones(self.volume_shape[-2]),
                                             2*np.ones(self.volume_shape[-1]))).astype(np.uint8)
        self.handler_param = {"get_current_window": get_current_window, "window_name": self.window_name, "volume": self.volume,
                              "point": self.current_point, "cube_size": cube_side, "previous_x": cube_side/2,
                              "previous_y": cube_side/2, "dragging": False, "display_resize": resize_factor,
                              "min": original_min, "max": original_max}

    def reset_current_point(self):
        global current_point
        current_point = None

    def display(self, channel_select=-1):
        global current_point

        self.last_channel = channel_select

        if current_point is None:
            current_point = self.current_point

        if channel_select < 0:
            axis0 = self.volume[current_point[0], :, :]
            axis1 = self.volume[:, current_point[1], :]
            axis2 = self.volume[:, :, current_point[2]]
        else:
            try:
                axis0 = self.volume[channel_select, current_point[0], :, :]
                axis1 = self.volume[channel_select, :, current_point[1], :]
                axis2 = self.volume[channel_select, :, :, current_point[2]]
            except IndexError:
                print(f"Channel {channel_select} not found. Using 0")
                self.last_channel = 0
                axis0 = self.volume[0, current_point[0], :, :]
                axis1 = self.volume[0, :, current_point[1], :]
                axis2 = self.volume[0, :, :, current_point[2]]

        self.handler_param["channel"] = channel_select
        cv.namedWindow(self.window_name)
        cv.setMouseCallback(self.window_name, mouse_handler, param=self.handler_param)
        cv.imshow(self.window_name, cv.resize(np.hstack((axis0, axis1, axis2)), (0, 0),
                                              fx=self.resize_factor, fy=self.resize_factor))
        key = cv.waitKey(0)
        if key == 27:
            return 'ESC'
        elif key == 13:
            return 'ENTER'
        elif key == 115:
            # Hide/show mask
            if self.displaying_mask:
                self.volume = self.original_volume
                self.displaying_mask = False
            else:
                self.volume = self.masked_volume
                self.displaying_mask = True
            self.handler_param["volume"] = self.volume

            return(self.display(channel_select=self.last_channel))
        elif self.multichannel:
            # Change displayed channel
            channel = key - 48

            if channel in range(self.C):
                return(self.display(channel_select=channel))
            else:
                return(self.display(channel_select=self.last_channel))



def mouse_handler(event, x, y, flags, param):
    '''
    OpenCV mouse event handler
    '''
    global current_point

    if current_point is None:
        print("Using params point")
        current_point = param["point"]

    volume, window_name = param["volume"], param["window_name"]
    x = x//param["display_resize"]
    y = y//param["display_resize"]

    if ((event == cv.EVENT_LBUTTONDOWN or param["dragging"]) and param["previous_x"] != x and param["previous_y"] != y) or event == cv.EVENT_MBUTTONDOWN:

        window = param["get_current_window"][x]
        param["dragging"] = True
        param["previous_x"] = x
        param["previous_y"] = y

        current_point[window] += (flags == 4)*1 - (flags == 12)*1

        if window == 0:
            current_point = [current_point[0], y, x]
        elif window == 1:
            current_point = [y, current_point[1], x - param["cube_size"]]
        elif window == 2:
            current_point = [y, x - param["cube_size"]*2, current_point[2]]
        param["point"] = current_point

        if x is not None and y is not None:
            if param["channel"] > -1:
                volume = volume[param["channel"]]

            axis0 = np.copy(volume[current_point[0], :, :])
            axis1 = np.copy(volume[:, current_point[1], :])
            axis2 = np.copy(volume[:, :, current_point[2]])

            axis0 = image_print(axis0, current_point[0], org=(20, 30))
            axis0 = image_print(axis0, param["channel"], org=(20, 60))
            displayed_value = volume[current_point[0], current_point[1], current_point[2]]
            original_value = displayed_value*(param["max"] - param["min"]) + param["min"]
            axis0 = image_print(axis0, displayed_value, org=(20, 90))
            axis0 = image_print(axis0, original_value, org=(20, 120))
            axis1 = image_print(axis1, current_point[1], org=(20, 30))
            axis2 = image_print(axis2, current_point[2], org=(20, 30))

            axis0 = cv.circle(axis0, (current_point[2], current_point[1]), 2, 1)
            axis1 = cv.circle(axis1, (current_point[2], current_point[0]), 2, 1)
            axis2 = cv.circle(axis2, (current_point[1], current_point[0]), 2, 1)

            axis0 = cv.line(axis0, (0, current_point[1]), (param["cube_size"] - 1, current_point[1]), 1)
            axis0 = cv.line(axis0, (current_point[2], 0), (current_point[2], param["cube_size"] - 1), 1)

            axis1 = cv.line(axis1, (current_point[2], 0), (current_point[2], param["cube_size"] - 1), 1)
            axis1 = cv.line(axis1, (0, current_point[0]), (param["cube_size"] - 1, current_point[0]), 1)

            axis2 = cv.line(axis2, (current_point[1], 0), (current_point[1], param["cube_size"] - 1), 1)
            axis2 = cv.line(axis2, (0, current_point[0]), (param["cube_size"] - 1, current_point[0]), 1)

        display = np.hstack((axis0, axis1, axis2))
        cv.imshow(window_name, cv.resize(display, (0, 0), fx=param["display_resize"], fy=param["display_resize"]))
    elif event == cv.EVENT_LBUTTONUP:
        param["dragging"] = False


def image_print(img, st, org=(20, 20), scale=1, color=1, font=cv.FONT_HERSHEY_SIMPLEX):
    '''
    Simplifies printing text on images
    '''
    if st is None:
        return img
    else:
        return cv.putText(img, str(st), org, font, scale, color)



def get_optimizer(name, params, lr, wd=0):
    if name == "RAdam":
        return optim.RAdam(params, lr=lr, weight_decay=wd)
    elif name == "Adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    elif name == "AdamW":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == "SGD":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError("Invalid optimizer name")


class DICELoss(torch.nn.Module):
    '''
    Calculates DICE Loss
    Use per channel for multiple targets.
    '''
    def __init__(self, volumetric=False, negative_loss=False, per_channel=False, check_bounds=True):
        self.name = "DICE Loss"
        super(DICELoss, self).__init__()
        self.volumetric = volumetric
        self.negative_loss = negative_loss
        self.per_channel = per_channel
        self.check_bounds = check_bounds

    def __call__(self, probs, targets):
        '''
        probs: output of last convolution, sigmoided or not (use apply_sigmoid=True if not)
        targets: binary target mask
        '''
        p_min = probs.min()
        p_max = probs.max()
        bounded = p_max <= 1.0 and p_min >= 0.0
        if self.check_bounds:
            if not bounded:
                raise ValueError(f"FATAL ERROR: DICE loss input not bounded between 1 and 0! {p_min} {p_max}")
        else:
            if not bounded:
                print(f"WARNING: DICE loss input not bounded between 1 and 0! {p_min} {p_max}")


        score = 0

        if self.per_channel:
            assert len(targets.shape) >= 4, ("less than 4 dimensions makes no sense with multi channel in a batch of 2D or 3D"
                                             "volumes")
            nchannels = targets.shape[1]
            if self.volumetric:
                score = torch.stack([vol_dice(probs[:, c], targets[:, c]) for c in range(nchannels)]).mean()
            else:
                score = torch.stack([batch_dice(probs[:, c], targets[:, c]) for c in range(nchannels)]).mean()
        else:
            if self.volumetric:
                score = vol_dice(probs, targets)
            else:
                score = batch_dice(probs, targets)

        if self.negative_loss:
            loss = -score
        else:
            loss = 1 - score

        return loss


def vol_dice(inpt, target, smooth=1.0):
    '''
    Calculate DICE of volume
    '''
    # q = inpt.size(0)
    assert len(inpt) != 0, " trying to compute DICE of nothing"

    iflat = inpt.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    eps = 0
    if smooth == 0.0:
        eps = sys.float_info.epsilon

    iflat_sum = iflat.sum()
    tflat_sum = tflat.sum()

    if iflat_sum.item() == 0.0 and tflat_sum.item() == 0.0:
        # print("DICE Metric got black mask and prediction!")
        dice = torch.tensor(1.0, requires_grad=True, device=inpt.device)
    else:
        dice = (2. * intersection + smooth) / (iflat_sum + tflat_sum + smooth + eps)

    value = dice.item()
    assert value >= 0.0 or value <= 1.0, " DICE not between 0 and 1! something is wrong"

    return dice


def batch_dice(inpt, target, smooth=1.0):
    '''
    Calculate DICE of a batch of two binary masks
    Returns mean dice of all slices
    '''
    q = inpt.size(0)
    assert len(inpt) != 0, " trying to compute DICE of nothing"

    iflat = inpt.contiguous().view(q, -1)
    tflat = target.contiguous().view(q, -1)
    intersection = (iflat * tflat).sum(dim=1)

    eps = 0
    if smooth == 0.0:
        eps = sys.float_info.epsilon

    iflat_sum = iflat.sum(dim=1)
    tflat_sum = tflat.sum(dim=1)

    dice = (2. * intersection + smooth) / (iflat_sum + tflat_sum + smooth + eps)

    dice = dice.mean()
    value = dice.item()
    assert value >= 0.0 or value <= 1.0, " DICE not between 0 and 1! something is wrong"

    return dice


class DICEMetric(nn.Module):
    '''
    Calculates DICE Metric
    '''
    def __init__(self, apply_sigmoid=False, mask_ths=0.5, skip_ths=False, per_channel_metric=False, check_bounds=True):
        self.name = "DICE"
        self.lower_is_best = False
        super(DICEMetric, self).__init__()
        self.apply_sigmoid = apply_sigmoid
        self.mask_ths = mask_ths
        self.skip_ths = skip_ths
        self.per_channel_metric = per_channel_metric
        self.check_bounds = check_bounds

    def __call__(self, probs, target):
        '''
        Returns only DICE metric, as volumetric dice
        probs: output of last convolution, sigmoided or not (use apply_sigmoid=True if not)
        targets: float binary target mask
        '''
        probs = probs.type(torch.float32)
        target = target.type(torch.float32)

        if self.apply_sigmoid:
            probs = probs.sigmoid()

        p_min = probs.min()
        if self.check_bounds:
            assert p_min >= 0.0, f"FATAL ERROR: DICE metric input not positive! {p_min}"
        else:
            if p_min < 0:
                print(f"WARNING: Negative probabilities entering into DICE! {p_min}")

        if self.skip_ths:
            mask = probs
        else:
            mask = (probs > self.mask_ths).float()

        if self.per_channel_metric:
            assert len(target.shape) >= 4, ("less than 4 dimensions makes no sense with multi channel in a batch of 2D or 3D"
                                            "volumes")
            nchannels = target.shape[1]
            return [vol_dice(mask[:, c], target[:, c], smooth=0.0).item() for c in range(nchannels)]
        else:
            return vol_dice(mask, target, smooth=0.0).item()


def itk_snap_spawner(nparray: np.ndarray, title: str = "ITKSnap", itksnap_path: str = "/usr/bin/itksnap",
                     block: bool = True):
    '''
    Displays a three dimensional numpy array using SimpleITK and itksnap.
    Assumes itksnap is installed on /usr/bin/itksnap.
    Blocks process until all itksnap instances openend on the computer are closed. 
    '''
    assert os.path.isfile(itksnap_path), f"Couldn't find itksnap on {itksnap_path}"
    assert len(nparray.shape) in [3, 4], "Array not three dimensional"

    if len(nparray.shape) == 4 and np.array(nparray.shape).argmin() == 0:
        adjusted_nparray = nparray.transpose(1, 2, 3, 0)
    else:
        adjusted_nparray = nparray

    image_viewer = sitk.ImageViewer()
    image_viewer.SetTitle(title)
    image_viewer.SetApplication(itksnap_path)
    image_viewer.Execute(sitk.GetImageFromArray(adjusted_nparray))
    if block:
        monitor_itksnap()


class CoUNet3D_metrics():
    def __init__(self, classes=["P", "L"]):
        self.dice = DICEMetric(per_channel_metric=True)
        self.classes = classes

    def __call__(self, preds, tgt):
        dices = self.dice(preds, tgt)
        report = {}

        for i, c in enumerate(self.classes):
            report[f"{c}_dice"] = dices[i]

        return report
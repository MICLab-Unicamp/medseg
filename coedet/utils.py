import psutil
import time
import numpy as np
import multiprocessing as mp
from scipy.ndimage import zoom
from tqdm import tqdm


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

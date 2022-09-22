import numpy as np
from operator import itemgetter
import cc3d


def get_connected_components(volume, return_largest=2, verbose=False):
    '''
    volume: input volume
    return_largest: how many of the largest labels to return. If 0, nothing is changed in input volume
    verbose: prints label_count

    returns:
        filtered_volume, label_count, labeled_volume
    '''
    labels_out = cc3d.connected_components(volume.astype(np.int32))
    label_count = np.unique(labels_out, return_counts=True)[1]

    # Indicate which was the original label and sort by count
    label_count = [(label, count) for label, count in enumerate(label_count)]
    label_count.sort(key=itemgetter(1), reverse=True)
    label_count.pop(0)  # remove largest which should be background

    if verbose:
        print(f"Label count: {label_count}")

    filtered = None
    if return_largest > 0:
        for i in range(return_largest):
            try:
                id_max = label_count[i][0]
                if filtered is None:
                    filtered = (labels_out == id_max)
                else:
                    filtered += (labels_out == id_max)
            except IndexError:
                # We want more components that what is in the image, stop
                break

        if filtered is None:
            print("WARNING: Couldn find largest connected labels of lung.")
        else:
            volume = filtered * volume
            labels_out = filtered * labels_out

    return volume, label_count, labels_out


def post_processing(output, tqdm_iter=None):
    if tqdm_iter is not None:
        tqdm_iter.write("Full processing enabled, might take a while...")
    if tqdm_iter is not None:
        tqdm_iter.write("Unpacking outputs...")
    lung, covid = (output[0] > 0.5).astype(np.int32), (output[1] > 0.5).astype(np.int32)
    if tqdm_iter is not None:
        tqdm_iter.write("Calculating lung connected components...")
    lung, lung_lc, lung_labeled = get_connected_components(lung, return_largest=2)

    if tqdm_iter is not None:
        tqdm_iter.write("Extracting first and second largest components...")
    first_component = lung_labeled == lung_lc[0][0]
    try:
        second_component = lung_labeled == lung_lc[1][0]
    except IndexError:
        tqdm_iter.write("WARNING: Was not able to get a second component from lung segmentation.")
        second_component = np.zeros_like(first_component)

        
    lung = first_component + second_component
    covid = covid*lung

    return lung.astype(np.uint8), covid.astype(np.uint8)

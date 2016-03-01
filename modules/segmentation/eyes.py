import numpy as np
from modules.tools.misc import BBox
from modules.tools.morphology import gather_statistics

def similarity_matrix(stats, column):
    out = np.empty((stats.shape[0],stats.shape[0]))
    for i,vali in enumerate(stats[column].values):
        for j,valj in enumerate(stats[column].values):
            out[i,j] = float('inf') if i == j else np.abs(vali - valj)

    return out

def eyes_statistics(stack_statistics, min_area=1000, min_sphericity=1.0):
    eyes_stats = stack_statistics[(stack_statistics['area'] > min_area) & \
                                  (stack_statistics['sphericity'] > min_sphericity)]

    print eyes_stats
    sm = similarity_matrix(eyes_stats, 'area')
    if len(sm):
        i,j = np.unravel_index(sm.argmin(), sm.shape)
    else:
        raise ValueError("Similarity matrix of eyes is empty.")

    return eyes_stats.iloc[[i,j]]

def eyes_zrange(eyes_stats, extra_offset_ratio=0.5):
    if eyes_stats == None:
        return np.empty((0))

    start = eyes_stats['bb_z'].min()
    stop = start + eyes_stats['bb_depth'].max()
    extra_offset = int((stop - start) * extra_offset_ratio)

    return np.arange(start - extra_offset, stop + extra_offset + 1).astype(int)

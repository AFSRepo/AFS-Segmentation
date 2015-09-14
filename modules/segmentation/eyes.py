import numpy as np
from modules.tools.misc import BBox
from modules.tools.morphology import gather_statistics

def eyes_statistics(stack_statistics, min_area=1000, min_sphericity=0.9):
    eyes_stats = stack_statistics[(stack_statistics['area'] > min_area) & \
                                  (stack_statistics['sphericity'] > min_sphericity)]
    filtered_eyes_stats = eyes_stats.sort(['sphericity'], ascending=False).head(2)

    return filtered_eyes_stats

def eyes_zrange(eyes_stats, extra_offset_ratio=0.5):
    if eyes_stats == None:
        return np.empty((0))

    start = eyes_stats['bb_z'].min()
    stop = start + eyes_stats['bb_depth'].max()
    extra_offset = int((stop - start) * extra_offset_ratio)

    return np.arange(start - extra_offset, stop + extra_offset + 1).astype(int)

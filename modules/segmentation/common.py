import numpy as np
from modules.tools.morphology import object_counter, gather_statistics

def split_fish(stack_labels):
    #get bounding boxes of abdom and head parts
    objects_stats = object_counter(stack_labels)

    print objects_stats

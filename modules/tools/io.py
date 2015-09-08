import os
import numpy as np
import fabio as fb
import re

def create_raw_stack(dirpath, prefix):
    if os.path.exists(dirpath):
        files = [f for f in os.listdir(dirpath) if prefix in f]
        files.sort()

        _shape = fb.open(os.path.join(dirpath, files[0])).data.shape

        stack_data = np.zeros((len(files), _shape[1], _shape[0]))

        for i in np.arange(stack_data.shape[0]):
            stack_data[i] = fb.open(os.path.join(dirpath, files[i])).data
            if i % 100 == 0:
                print 'Converted slices %d' % i

        return stack_data
    else:
        print 'No dir: %s' % dirpath
        return None

import time
import numpy as np

class BBox(object):
    def __init__(self, *args, **kwargs):
        _input = args[0]
        if isinstance(_input, tuple):
            self.x = _input[2].start
            self.y = _input[1].start
            self.z = _input[0].start

            self.width  = _input[2].stop - _input[2].start + 1
            self.height = _input[1].stop - _input[1].start + 1
            self.depth  = _input[0].stop - _input[0].start + 1

        elif isinstance(args[0], dict):
            self.x = _input['bb_x']
            self.y = _input['bb_y']
            self.z = _input['bb_z']

            self.width = _input['bb_width']
            self.height = _input['bb_height']
            self.depth = _input['bb_depth']

    def create_tuple(self, offset=0):
        z_start, z_end = self.z if (self.z - offset) < 0 else (self.z - offset), \
                          self.z + self.depth + offset
        y_start, y_end = self.y if (self.y - offset) < 0 else (self.y - offset), \
                          self.y + self.height + offset
        x_start, x_end = self.x if (self.x - offset) < 0 else (self.x - offset), \
                          self.x + self.width + offset

        return np.index_exp[z_start:z_end, y_start:y_end, x_start:x_end]

    def __str__(self):
        return 'Origin: (%d, %d, %d) Shape:(%d, %d, %d)' % \
                    (self.x, self.y, self.z, self.width, self.height, self.depth)

class Timer(object):
    def __init__(self):
        self.start = time.time()

    def elapsed(self, title):
        self.stop = time.time()
        print "Elapsed time (%s):%f sec" % (title, self.stop - self.start)

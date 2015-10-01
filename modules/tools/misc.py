import time
import numpy as np

class BBox(object):
    def __init__(self, *args, **kwargs):
        _input = args[0]
        self.is2DBox = False

        if  (len(_input) != 3) or ('bb_x' in _input) or ('bb_width' in _input):
            self.is2DBox = True

        if isinstance(_input, tuple):
            self.y = _input[1].start
            self.z = _input[0].start

            self.height = _input[1].stop - _input[1].start + 1
            self.depth  = _input[0].stop - _input[0].start + 1

            if not self.is2DBox:
                self.x = _input[2].start
                self.width  = _input[2].stop - _input[2].start + 1

        elif isinstance(_input, dict):
            self.y = _input['bb_y']
            self.z = _input['bb_z']

            self.height = _input['bb_height']
            self.depth = _input['bb_depth']

            if not self.is2DBox:
                self.x = _input['bb_x']
                self.width = _input['bb_width']

    def create_tuple(self, offset=0):
        z_start, z_end = self.z if (self.z - offset) < 0 else (self.z - offset), \
                          self.z + self.depth + offset
        y_start, y_end = self.y if (self.y - offset) < 0 else (self.y - offset), \
                          self.y + self.height + offset

        if not self.is2DBox:
            x_start, x_end = self.x if (self.x - offset) < 0 else (self.x - offset), \
                              self.x + self.width + offset


        return np.index_exp[z_start:z_end, y_start:y_end] if self.is2DBox \
                else np.index_exp[z_start:z_end, y_start:y_end, x_start:x_end]

    def __str__(self):
        if not self.is2DBox:
            return 'Origin: (%d, %d, %d) Shape:(%d, %d, %d)' % \
                        (self.x, self.y, self.z, self.width, self.height, self.depth)
        else:
            return 'Origin: (%d, %d) Shape:(%d, %d)' % \
                        (self.y, self.z, self.height, self.depth)

class Timer(object):
    def __init__(self):
        self.start = time.time()

    def elapsed(self, title):
        self.stop = time.time()
        print "Elapsed time (%s):%f sec" % (title, self.stop - self.start)

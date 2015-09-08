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

    def create_tuple(self):
        return np.index_exp[self.z:(self.z + self.depth),\
                            self.y:(self.y + self.height),\
                            self.x:(self.x + self.width)]

    def __str__(self):
        return 'Origin: (%d, %d, %d) Shape:(%d, %d, %d)' % \
                    (self.x, self.y, self.z, self.width, self.height, self.depth)

class Timer(object):
    def __init__(self):
        self.start = time.time()

    def elapsed(self, title):
        self.stop = time.time()
        print "Elapsed time (%s):%f sec" % (title, self.stop - self.start)

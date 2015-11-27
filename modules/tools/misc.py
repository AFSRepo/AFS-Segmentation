import time
import numpy as np

class BBox(object):
    def __init__(self, *args, **kwargs):
        _input = args[0]
        self.is2DBox = False

        if isinstance(_input, dict) and (('bb_x' not in _input) or ('bb_width' not in _input)):
            self.is2DBox = True

        if isinstance(_input, tuple) and len(_input) < 3:
            self.is2DBox = True

        if isinstance(_input, tuple):
            self.y = _input[1].start
            self.z = _input[0].start

            self.height = _input[1].stop - _input[1].start
            self.depth  = _input[0].stop - _input[0].start

            if not self.is2DBox:
                self.x = _input[2].start
                self.width  = _input[2].stop - _input[2].start

        elif isinstance(_input, dict):
            self.y = _input['bb_y']
            self.z = _input['bb_z']

            self.height = _input['bb_height']
            self.depth = _input['bb_depth']

            if not self.is2DBox:
                self.x = _input['bb_x']
                self.width = _input['bb_width']

        self.y = self._force_evenness(self.y)
        self.z = self._force_evenness(self.z)

        self.height = self._force_evenness(self.height)
        self.depth = self._force_evenness(self.depth)

        if not self.is2DBox:
            self.x = self._force_evenness(self.x)
            self.width = self._force_evenness(self.width)

    def _force_evenness(self, value):
        return (value - 1) if value % 2 != 0 else value

    def create_tuple(self, offset=0, max_ranges=None):
        z_start, z_end = None, None
        y_start, y_end = None, None
        x_start, x_end = None, None

        if max_ranges:
            z_max, y_max, x_max = max_ranges

            z_start, z_end = self.z if (self.z - offset) < 0 else (self.z - offset), \
                              z_max if (self.z + self.depth + offset) > z_max else \
                                        (self.z + self.depth + offset)
            y_start, y_end = self.y if (self.y - offset) < 0 else (self.y - offset), \
                              y_max if (self.y + self.height + offset) > y_max else \
                                        (self.y + self.height + offset)

            if not self.is2DBox:
                x_start, x_end = self.x if (self.x - offset) < 0 else (self.x - offset), \
                                  x_max if (self.x + self.width + offset) > x_max else \
                                            (self.x + self.width + offset)
        else:
            z_start, z_end = self.z if (self.z - offset) < 0 else (self.z - offset), \
                              self.z + self.depth + offset
            y_start, y_end = self.y if (self.y - offset) < 0 else (self.y - offset), \
                              self.y + self.height + offset

            if not self.is2DBox:
                x_start, x_end = self.x if (self.x - offset) < 0 else (self.x - offset), \
                                  self.x + self.width + offset

        z_start, z_end = self._force_evenness(z_start), self._force_evenness(z_end)
        y_start, y_end = self._force_evenness(y_start), self._force_evenness(y_end)

        if not self.is2DBox:
            x_start, x_end = self._force_evenness(x_start), self._force_evenness(x_end)

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

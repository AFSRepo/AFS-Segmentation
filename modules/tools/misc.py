import os
import timeit
import numpy as np

if os.name == 'posix':
    import psutil
else:
    import wmi

class BBox(object):
    def __init__(self, _input, force_evenness=True):
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

        if force_evenness:
            self.y = self._force_evenness(self.y)
            self.z = self._force_evenness(self.z)

            self.height = self._force_evenness(self.height)
            self.depth = self._force_evenness(self.depth)

            if not self.is2DBox:
                self.x = self._force_evenness(self.x)
                self.width = self._force_evenness(self.width)

    def _force_evenness(self, value):
        return (value - 1) if value % 2 != 0 else value

    def create_tuple(self, offset=0, max_ranges=None, force_positiveness=True):
        z_start, z_end = None, None
        y_start, y_end = None, None
        x_start, x_end = None, None

        frac_offset = offset/100.

        max_dim = max([self.depth, self.height]) if self.is2DBox else max([self.depth, self.height, self.width])

        offset_dim = max_dim * frac_offset

        z_max, y_max, x_max = None, None, None

        if max_ranges:
            if not self.is2DBox:
                z_max, y_max, x_max = max_ranges
            else:
                z_max, y_max = max_ranges

            z_start, z_end = self.z if (self.z - offset_dim) < 0 else (self.z - offset_dim), \
                              z_max if (self.z + self.depth + offset_dim) > z_max else \
                                        (self.z + self.depth + offset_dim)
            y_start, y_end = self.y if (self.y - offset_dim) < 0 else (self.y - offset_dim), \
                              y_max if (self.y + self.height + offset_dim) > y_max else \
                                        (self.y + self.height + offset_dim)

            if not self.is2DBox:
                x_start, x_end = self.x if (self.x - offset_dim) < 0 else (self.x - offset_dim), \
                                  x_max if (self.x + self.width + offset_dim) > x_max else \
                                            (self.x + self.width + offset_dim)
        else:
            if force_positiveness:
                z_start, z_end = self.z if (self.z - offset_dim) < 0 else (self.z - offset_dim), \
                                  self.z + self.depth + offset_dim
                y_start, y_end = self.y if (self.y - offset_dim) < 0 else (self.y - offset_dim), \
                                  self.y + self.height + offset_dim

                if not self.is2DBox:
                    x_start, x_end = self.x if (self.x - offset_dim) < 0 else (self.x - offset_dim), \
                                      self.x + self.width + offset_dim
            else:
                z_start, z_end = self.z - offset_dim, self.z + self.depth + offset_dim
                y_start, y_end = self.y - offset_dim, self.y + self.height + offset_dim

                if not self.is2DBox:
                    x_start, x_end = self.x - offset_dim, self.x + self.width + offset_dim

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
        self.start = timeit.default_timer()

    def elapsed(self, title):
        self.stop = timeit.default_timer()
        diff = self.stop - self.start
        print "Elapsed time (%s): %f sec ~= %f min" % (title, diff, diff / 60.)

#http://stackoverflow.com/questions/5478351/python-time-measure-function
def timing(f):
    def wrap(*args, **kwargs):
        print '%s has started execution.' % f.func_name
        time1 = timeit.default_timer()
        ret = f(*args, **kwargs)
        time2 = timeit.default_timer()
        residual = time2 - time1
        print '%s function took %0.3f ms ~= %0.2f sec ~= %0.2f min.' % (f.func_name, residual * 1000., residual, residual / 60.)
        return ret
    return wrap

def print_available_ram():
    total, available = -1, -1

    if os.name == 'posix':
        stats = psutil.virtual_memory()
        total, available = stats[0] / 1024. / 1024., stats[4] / 1024. / 1024.
    else:
        pc = wmi.WMI()
        total, available = float(pc.Win32_ComputerSystem()[0].TotalPhysicalMemory) / 1024. / 1024., float(pc.Win32_OperatingSystem()[0].FreePhysicalMemory) / 1024.

    fraction = available/total

    print "########## Avaliable RAM (%0.1f / %0.1f MB ~= %0.1f%%)" % (available, total, (fraction * 100.))

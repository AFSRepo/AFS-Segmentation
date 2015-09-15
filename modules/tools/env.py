import os
import csv
from .io import parse_filename, create_filename_with_shape, create_filename_with_bits

class DataEnvironment(object):
    def __init__(self, project_path, input_filepath):
        self.envs = dict()
        self.envs['project_path'] = project_path

        if not os.path.exists(self.envs['project_path']):
            os.makedirs(self.envs['project_path'])

        self.envs['input_data_path'] = input_filepath
        self.envs['target_filepath'] = ''

        self.envs['filename_ext'] = os.path.basename(input_filepath)
        self.envs['filename_no_ext'] = os.path.splitext(os.path.basename(input_filepath))[0]

        self.envs['project_temp_path'] = os.path.join(self.envs['project_path'], 'temp')

        if not os.path.exists(self.envs['project_temp_path']):
            os.makedirs(self.envs['project_temp_path'])

        self.oneWarpName = '%s1Warp.nii.gz'
        self.oneInverseWarp = '%s1InverseWarp.nii.gz'
        self.genericAffine = '%s0GenericAffine.mat'
        self.warped = '%sWarped.nii.gz'

        self.ANTSPATH = "C:\\Users\\Administrator\\Documents\\ANTs"

    def save(self):
        w = csv.writer(open(self._get_cache_path(), "w"))
        for key, val in self.envs.items():
            w.writerow([key, val])

    def load(self):
        if os.path.exists(self._get_cache_path()):
            for key, val in csv.reader(open(self._get_cache_path())):
                self.envs[key] = val

    def is_entry_exists(self, key_val):
        if key_val in self.envs:
            return True

        return False

    def _get_cache_path(self):
        return os.path.join(self.envs['project_temp_path'], \
                            "env_%s.csv" % self.envs['filename_no_ext'])

    def get_working_path(self):
        return self.envs['project_temp_path']

    def get_input_path(self):
        return self.envs['input_data_path']

    def get_target_path(self):
        return self.envs['target_filepath']

    def get_target_labels_path(self):
        self.envs['target_labels_filepath'] = \
            self._gen_new_filepath(self.envs['project_temp_path'],
                                   self.envs['target_filepath'],
                                   prefix='labels',
                                   bits=8)

        return self.envs['target_labels_filepath']

    def get_input_labels_path(self):
        self.envs['input_labels_filepath'] = \
            self._gen_new_filepath(self.envs['project_temp_path'],
                                   self.envs['input_data_path'],
                                   prefix='labels',
                                   bits=8)

        return self.envs['input_labels_filepath']

    def set_target_data_path(self, target_filepath):
        self.envs['target_filepath'] = target_filepath

    def _get_new_filepath(self, root_path, shape=(), prefix='', filename='', ext='.raw'):
        filename = self.envs['input_data_path']
        new_filename = create_filename_with_shape(filename, shape, prefix, ext)
        return os.path.join(root_path, new_filename)

    def _gen_new_filepath(self, root_path, filepath='', shape=(), prefix='', ext='', bits=0):
        _name, _bits, _shape, _ext = parse_filename(filepath)

        if len(shape) != 0:
            _shape = shape

        if len(prefix) != 0:
            _name = _name + '_' + prefix

        if len(ext) != 0:
            _ext = ext

        if bits != 0:
            _bits = bits

        if _ext == '.raw':
            _shape = tuple(reversed(_shape))

        new_filename = '%s_%dbit_%s%s' % (_name, _bits, 'x'.join(str(v) for v in _shape), _ext)

        return os.path.join(root_path, new_filename)

    def _get_ants_output_names(self, fixed_image_name, moving_image_name, phase_name):
        final_name = '%s_%sto%s' % (phase_name, moving_image_name, fixed_image_name)

        out_names = dict()
        out_names['out_name'] = final_name
        out_names['warp'] = self.oneWarpName % final_name
        out_names['iwarp'] = self.oneInverseWarp % final_name
        out_names['gen_affine'] = self.genericAffine % final_name
        out_names['warped'] = self.warped % final_name

        return out_names

    def get_input_labels_from_data_path(self):
        tail, head = os.path.split(self.get_input_path())

        name, _, shape, ext = parse_filename(head)

        new_filename = '%s_labels_%dbit_%s%s' % (name, 8, 'x'.join(str(v) for v in shape), ext)

        self.envs['input_data_labels_path'] = os.path.join(tail, new_filename)

        return self.envs['input_data_labels_path']

    def get_extracted_volume_path(self, data_shape):
        self.envs['extracted_input_data_path'] = \
            self._get_new_filepath(self.envs['project_temp_path'], \
                                        shape=data_shape, prefix='extracted')

        return self.envs['extracted_input_data_path']

    def get_extracted_volume_niigz_path(self, data_shape):
        self.envs['extracted_input_data_path_niigz'] = \
            self._get_new_filepath(self.envs['project_temp_path'], \
                                        shape=data_shape, prefix='extracted', ext='.nii.gz')

        return self.envs['extracted_input_data_path_niigz']

    def get_new_volume_niigz_path(self, data_shape, new_prefix):
        self.envs['%s_input_data_path_niigz' % new_prefix] = \
            self._get_new_filepath(self.envs['project_temp_path'], \
                                            shape=data_shape, prefix=new_prefix, ext='.nii.gz')

        return self.envs['%s_input_data_path_niigz' % new_prefix]

    def get_new_volume_path(self, data_shape, new_prefix):
        self.envs['%s_input_data_path' % new_prefix] = \
            self._get_new_filepath(self.envs['project_temp_path'], \
                                                shape=data_shape, prefix=new_prefix, ext='.raw')

        return self.envs['%s_input_data_path' % new_prefix]

    def get_head_abdomen_volume_paths(self, head_shape, abdomen_shape):
        self.envs['extracted_data_parts_paths'] = { \
            'head': self._get_new_filepath(self.envs['project_temp_path'], \
                                           shape=head_shape, \
                                           prefix='extracted_head_data_part'), \
            'abdomen': self._get_new_filepath(self.envs['project_temp_path'], \
                                              shape=abdomen_shape, \
                                              prefix='extracted_abdomen_data_part') \
        }

        return self.envs['extracted_data_parts_paths']


    def get_aligned_data_paths(self, phase_name, produce_paths=True):
        if len(self.envs['target_filepath']) != 0:
            moving_image_name, _, _, _ = parse_filename(self.envs['input_data_path'])
            fixed_image_name, _, _, _ = parse_filename(self.envs['target_filepath'])

            out_names = self._get_ants_output_names(fixed_image_name, moving_image_name, phase_name)

            if produce_paths:
                for key, value in out_names.items():
                    out_names[key] = os.path.join(self.envs['project_temp_path'], value)

            self.envs['%s_aligned_input_data_path' % phase_name] = out_names

            return self.envs['%s_aligned_input_data_path' % phase_name]
        else:
            print 'No target file!'
            return None

    def get_statistic_path(self, prefix):
        name, _, _, _ = parse_filename(self.envs['input_data_path'])

        self.envs['input_data_%s_statistics' % prefix] = \
            os.path.join(self.envs['project_temp_path'], \
                            '%s_statistics_%s.csv' % (prefix, name))

        return self.envs['input_data_%s_statistics' % prefix]

    def test_paths(self):
        self.set_target_data_path("C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish204\\fish204_32bit_621x621x1800.raw")
        print self.get_input_path()
        print self.get_input_labels_from_data_path()
        print self.get_extracted_volume_path((320,320,1000))
        print self.get_extracted_volume_niigz_path((320,320,1000))
        print self.get_head_abdomen_volume_paths((300,300,300), (500,500,500))
        print self.get_aligned_data_paths("FISH_SEPARATION")
        print self.get_statistic_path("eyes")

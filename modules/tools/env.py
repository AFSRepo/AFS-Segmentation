import os
import csv
import pickle
from .io import parse_filename, create_filename_with_shape, create_filename_with_bits

class NoneDict(dict):
    def __getitem__(self, key):
        return dict.get(self, key, '')

class DataEnvironment(object):
    def __init__(self, project_path, input_filepath):
        self.envs = NoneDict()
        self.envs['project_path'] = project_path

        if not os.path.exists(self.envs['project_path']):
            os.makedirs(self.envs['project_path'])

        self.envs['input_data_path'] = input_filepath
        self.envs['target_data_path'] = None
        self.envs['input_data_labels_path'] = None
        self.envs['input_data_spine_labels_path'] = None

        self.envs['filename_ext'] = os.path.basename(input_filepath)
        self.envs['filename_no_ext'] = os.path.splitext(os.path.basename(input_filepath))[0]

        self.envs['project_temp_path'] = os.path.join(self.envs['project_path'], 'temp')

        if not os.path.exists(self.envs['project_temp_path']):
            os.makedirs(self.envs['project_temp_path'])

        self.oneWarpName = '%s1Warp.nii.gz'
        self.oneInverseWarp = '%s1InverseWarp.nii.gz'
        self.genericAffine = '%s0GenericAffine.mat'
        self.warped = '%sWarped.nii.gz'

        self.regName = '%sOutputRegResult.nii.gz'
        self.nonRigidTransName = '%sOutputRegNonRigidTransform.nii.gz'
        self.affineMatrix = '%sAffineMatrix.txt'
        self.invAffineMatrix = '%sInvAffineMatrix.txt'

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
        if self.get_target_path():
            return os.path.join(self.envs['project_temp_path'], \
                                "env_%s_to_%s.csv" % (self.envs['filename_no_ext'], os.path.splitext(os.path.basename(self.envs['target_data_path']))[0]))
        else:
            return os.path.join(self.envs['project_temp_path'], \
                                "env_%s.csv" % self.envs['filename_no_ext'])

    def get_working_path(self):
        return self.envs['project_temp_path']

    def get_input_path(self):
        return self.envs['input_data_path']

    def set_target_data_path(self, filepath):
        self.envs['target_data_path'] = filepath

    def get_target_path(self):
        return self.envs['target_data_path']

    def set_effective_volume_bbox(self, bbox):
        self.envs['effective_volume_bbox'] = pickle.dumps(bbox)

    def get_effective_volume_bbox(self):
        return pickle.loads(self.envs['effective_volume_bbox']) if self.envs['effective_volume_bbox'] else None

    def set_input_labels_path(self, filepath):
        self.envs['input_data_labels_path'] = filepath

    def get_input_labels_path(self):
        return self.envs['input_data_labels_path']

    def set_input_spine_labels_path(self, filepath):
        self.envs['input_data_spine_labels_path'] = filepath

    def get_input_spine_labels_path(self):
        return self.envs['input_data_spine_labels_path']

    def _get_ants_output_names(self, fixed_image_name, fixed_image_size, moving_image_name, moving_image_size, phase_name):
        final_name = '%s_%s_%sTO%s_%s' % (phase_name, moving_image_name, moving_image_size, fixed_image_name, fixed_image_size)

        out_names = dict()
        out_names['out_name'] = final_name
        out_names['warp'] = self.oneWarpName % final_name
        out_names['iwarp'] = self.oneInverseWarp % final_name
        out_names['gen_affine'] = self.genericAffine % final_name
        out_names['warped'] = self.warped % final_name

        return out_names

    def _get_nifty_output_names(self, fixed_image_name, fixed_image_size, moving_image_name, moving_image_size, phase_name):
        final_name = '%s_%s_%sTO%s_%s' % (phase_name, moving_image_name, moving_image_size, fixed_image_name, fixed_image_size)

        out_names = dict()
        out_names['out_name'] = final_name
        out_names['reg_result'] = self.regName % final_name
        out_names['non_rigid_trans'] = self.nonRigidTransName % final_name
        out_names['affine_mat'] = self.affineMatrix % final_name
        out_names['inv_affine_mat'] = self.invAffineMatrix % final_name

        return out_names

    def _gen_new_filepath(self, root_path, filepath, shape=None, prefix=None, ext=None, bits=None):
        _name, _bits, _shape, _ext = parse_filename(filepath)

        if shape:
            _shape = shape

        if prefix:
            _name = _name + '_' + prefix

        if ext:
            _ext = ext

        if bits:
            _bits = bits

        if _ext == '.raw':
            _shape = tuple(reversed(_shape))

        new_filename = '%s_%dbit_%s%s' % (_name, _bits, 'x'.join(str(v) for v in _shape), _ext)

        return os.path.join(root_path, new_filename)

    def get_new_volume_niigz_path(self, data_shape, new_prefix, bits=32):
        self.envs['%s_input_data_path_niigz' % new_prefix] = \
            self._gen_new_filepath(self.envs['project_temp_path'], \
                self.envs['input_data_path'], shape=data_shape, \
                    prefix=new_prefix, ext='.nii.gz', bits=bits)

        return self.envs['%s_input_data_path_niigz' % new_prefix]

    def get_new_volume_path(self, data_shape, new_prefix, bits=32):
        self.envs['%s_input_data_path' % new_prefix] = \
            self._gen_new_filepath(self.envs['project_temp_path'], \
                self.envs['input_data_path'], shape=data_shape, \
                    prefix=new_prefix, ext='.raw', bits=bits)

        return self.envs['%s_input_data_path' % new_prefix]

    def get_new_volume_labels_niigz_path(self, data_shape, new_prefix):
        self.envs['%s_input_data_labels_path_niigz' % new_prefix] = \
            self._gen_new_filepath(self.envs['project_temp_path'], \
                self.envs['input_data_labels_path'], shape=data_shape, \
                    prefix=new_prefix, ext='.nii.gz')

        return self.envs['%s_input_data_labels_path_niigz' % new_prefix]

    def get_new_volume_labels_path(self, data_shape, new_prefix):
        self.envs['%s_input_data_labels_path' % new_prefix] = \
            self._gen_new_filepath(self.envs['project_temp_path'], \
                self.envs['input_data_labels_path'], shape=data_shape, \
                    prefix=new_prefix, ext='.raw')

        return self.envs['%s_input_data_labels_path' % new_prefix]


    def get_head_abdomen_volume_paths(self, head_shape, abdomen_shape):
        self.envs['extracted_data_parts_paths'] = { \
            'head': self._gen_new_filepath(self.envs['project_temp_path'], \
                            self.envs['input_data_path'], shape=head_shape, \
                                prefix='extracted_head_data_part', ext='.raw'),

            'abdomen': self._gen_new_filepath(self.envs['project_temp_path'], \
                            self.envs['input_data_path'], shape=abdomen_shape, \
                                prefix='extracted_abdomen_data_part', ext='.raw')
        }

        return self.envs['extracted_data_parts_paths']


    def get_aligned_data_paths(self, phase_name, produce_paths=True):
        if self.envs['target_data_path']:
            moving_image_name, _, moving_image_size, _ = parse_filename(self.envs['input_data_path'])
            fixed_image_name, _, fixed_image_size, _ = parse_filename(self.envs['target_data_path'])

            moving_image_size_str = 'x'.join(str(v) for v in moving_image_size)
            fixed_image_size_str = 'x'.join(str(v) for v in fixed_image_size)

            out_names = self._get_ants_output_names(fixed_image_name, fixed_image_size_str, \
                            moving_image_name, moving_image_size_str, phase_name)

            if produce_paths:
                for key, value in out_names.items():
                    out_names[key] = os.path.join(self.envs['project_temp_path'], value)

            self.envs['%s_ants_input_data_path' % phase_name] = out_names

            return self.envs['%s_ants_input_data_path' % phase_name]
        else:
            print 'Error (get_aligned_data_paths): No target file!'
            return None

    def get_aligned_data_paths_nifty(self, phase_name, produce_paths=True):
        if self.envs['target_data_path']:
            moving_image_name, _, moving_image_size, _ = parse_filename(self.envs['input_data_path'])
            fixed_image_name, _, fixed_image_size, _ = parse_filename(self.envs['target_data_path'])

            moving_image_size_str = 'x'.join(str(v) for v in moving_image_size)
            fixed_image_size_str = 'x'.join(str(v) for v in fixed_image_size)

            out_names = self._get_nifty_output_names(fixed_image_name, fixed_image_size_str, \
                            moving_image_name, moving_image_size_str, phase_name)

            if produce_paths:
                for key, value in out_names.items():
                    out_names[key] = os.path.join(self.envs['project_temp_path'], value)

            self.envs['%s_nifty_input_data_path' % phase_name] = out_names

            return self.envs['%s_nifty_input_data_path' % phase_name]
        else:
            print 'Error (get_aligned_data_paths_nifty): No target file!'
            return None

    def get_statistic_path(self, prefix):
        name, _, _, _ = parse_filename(self.envs['input_data_path'])

        self.envs['input_data_%s_statistics' % prefix] = \
            os.path.join(self.envs['project_temp_path'], \
                            '%s_statistics_%s.csv' % (prefix, name))

        return self.envs['input_data_%s_statistics' % prefix]

    def test_paths(self):
        self.set_target_data_path("C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish204\\fish204_32bit_621x621x1800.raw")
        print self.get_working_path()
        print self.get_input_path()
        print self.get_target_path()
        print self.get_new_volume_niigz_path((320,320,1000), 'extracted_zoomed_0p5')
        print self.get_new_volume_path((1000,320,320), 'extracted_zoomed_0p5')
        print self.get_new_volume_labels_niigz_path((320,320,1000), 'extracted')
        print self.get_new_volume_labels_path((1000,320,320), 'extracted')
        print self.get_head_abdomen_volume_paths((300,300,300), (500,500,500))
        print self.get_aligned_data_paths("FISH_SEPARATION")
        print self.get_statistic_path("eyes")

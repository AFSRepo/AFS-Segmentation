import os
import re
import numpy as np
import fabio as fb
import nibabel as nib

INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'MedakaRawData'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'AFS-output'))
LSDF_DIR = '/mnt/LSDF' if os.name == 'posix' else "Z:\\"
ANTS_SCRIPTS_PATH_FMT = os.path.abspath(os.path.join(os.path.dirname(__file__), \
                                        os.path.pardir, \
                                        os.path.pardir, \
                                        'scripts'))
ORGAN_LABEL_TEMPLATE = '%s_organ_%s_label'
ORGAN_DATA_TEMPLATE = '%s_organ_%s'

def open_data(filepath):
    _, glob_ext = os.path.splitext(os.path.basename(filepath))
    data = None

    if glob_ext == '.raw':
        name, bits, size, ext = parse_filename(filepath)
        data_type = np.float32 if bits == 32 else np.uint8
        data = np.memmap(filepath, dtype=data_type, shape=tuple(reversed(size)))
    elif glob_ext == '.nii.gz' or glob_ext == '.nii' or glob_ext == '.gz':
        print filepath
        data = nib.load(filepath).get_data()
    else:
        print 'Incorrent file format, or filename.'

    return data

def create_raw_stack(dirpath, prefix):
    if os.path.exists(dirpath):
        files = [f for f in os.listdir(dirpath) if prefix in f]
        files.sort()

        _shape = fb.open(os.path.join(dirpath, files[0])).data.shape

        stack_data = np.zeros((len(files), _shape[1], _shape[0]), dtype=np.float32)

        for i in np.arange(stack_data.shape[0]):
            stack_data[i] = fb.open(os.path.join(dirpath, files[i])).data.astype(np.float32)

            if i % 100 == 0:
                print 'Converted slices %d' % i

        return stack_data
    else:
        print 'No dir: %s' % dirpath
        return None

def save_as_nifti(data_stack, output_filepath):
    nifti_stack_data = nib.Nifti1Image(data_stack, np.eye(4))
    nib.save(nifti_stack_data, output_filepath)

def get_filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

def parse_filename(filepath):
    basename, ext = os.path.splitext(os.path.basename(filepath))

    comps = basename.split('_')
    size = tuple([int(v) for v in comps[-1:][0].split('x')])
    bits = int(re.findall('\d+', comps[-2:-1][0])[0])
    name = '_'.join(comps[:-2])

    return name, bits, size, ext

def create_filename_with_shape(filepath, shape, prefix='', ext='.raw'):
    name, bits, _, _ = parse_filename(filepath)

    if ext == '.raw':
        shape = tuple(reversed(shape))

    if len(prefix) != 0:
        name = name + '_' + prefix

    return '%s_%dbit_%s%s' % (name, bits, 'x'.join(str(v) for v in shape), ext)

def create_filename_with_bits(filepath, prefix='', bits=32):
    name, _, size, ext = parse_filename(filepath)

    if ext == '.raw':
        shape = tuple(reversed(shape))

    if len(prefix) != 0:
        name = name + '_' + prefix

    return '%s_%dbit_%s%s' % (name, bits, 'x'.join(str(v) for v in shape), ext)

def check_files(working_dir, key_word):
    for f in os.listdir(working_dir):
        if os.path.exists(os.path.join(working_dir, f)) and key_word in f:
            return True

    return False

def get_path_by_name(fish_number, input_dir, isFindLabels=False, label_prefix=None):
    print 'fish_number = %d, input_dir = %s, isFindLabels = %d, label_prefix = %s' % (fish_number, input_dir, isFindLabels, str(label_prefix))
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    files = os.listdir(input_dir)
    files = [os.path.join(input_dir, f) for f in files]
    print '&&&&&&input_dir = %s' % input_dir

    file_name_lens = {}
    if OUTPUT_DIR in input_dir:
        file_name_lens = {'data': 2, 'labels': 3, 'organ_labels': 4}
    else:
        file_name_lens = {'data': 1, 'labels': 2, 'organ_labels': 3}

    print 'file_name_lens = %s' % str(file_name_lens)
    output_filepath = None
    for f in files:
        name, _, _, _ = parse_filename(f)
        name = name.split('_')
        if isFindLabels:
            if label_prefix is None:
                if len(name) == file_name_lens['labels']:
                    output_filepath = f
            else:
                if (len(name) == file_name_lens['organ_labels']) and (label_prefix in f):
                    output_filepath = f
        else:
            if len(name) == file_name_lens['data']:
                output_filepath = f

    print output_filepath
    return output_filepath

def __get_path_by_name(fish_number, input_dir, isFindLabels=False, label_prefix=None):
    print 'fish_number = %d, input_dir = %s, isFindLabels = %d, label_prefix = %s' % (fish_number, input_dir, isFindLabels, str(label_prefix))
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    #print 'LIST DIR:'
    #print os.listdir(input_dir)

    for fname in os.listdir(input_dir):
        path = os.path.join(input_dir, fname)
        if os.path.isfile(path):
            if isFindLabels:
                if label_prefix is not None:
                    #if all(v in path for v in [str(fish_number), 'labels', label_prefix]):
                    if '_'.join([str(fish_number), 'labels', label_prefix]) in path:
                        print 'Filepath label: %s' % path
                        return path
                elif '_'.join([str(fish_number), 'labels']) in path:
                    return path
                # lababel_name = '_'.join([v for v in ['labels', label_prefix] if v is not None])
                # print lababel_name
                # if all(v in path for v in [str(fish_number), lababel_name]):
                #     print 'Filepath label: %s' % path
                #     return path
            else:
                if str(fish_number) in path and 'label' not in path:
                    print 'Filepath data: %s' % path
                    return path
        else:
            continue

    return None

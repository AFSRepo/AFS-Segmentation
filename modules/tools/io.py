import os
import re
import numpy as np
import fabio as fb
import nibabel as nib

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

def get_path_by_name(fish_number, input_dir):
    for fname in os.listdir(input_dir):
        path = os.path.join(input_dir, fname)
        if os.path.isfile(path):
            if str(fish_number) in path:
                return path
            else:
                continue

    return None

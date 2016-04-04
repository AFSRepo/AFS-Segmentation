import sys
import os
import numpy as np
from modules.tools.env import DataEnvironment
from multiprocessing import Process, Pool
from modules.tools.io import create_raw_stack, open_data, create_filename_with_shape, parse_filename, get_path_by_name
from modules.tools.processing import binarizator, align_fish_by_eyes_tail
from modules.tools.misc import Timer
from modules.tools.morphology import object_counter, gather_statistics, extract_largest_area_data
from modules.segmentation.eyes import eyes_statistics, eyes_zrange
from modules.segmentation.spine import run_spine_segmentation
from modules.segmentation.common import split_fish, align_fish
from modules.segmentation.common import crop_align_data, brain_segmentation, \
brain_segmentation_nifty, brain_segmentation_ants
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.interpolation import zoom, rotate
import pandas as pd
import pdb
import shutil

INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'MedakaRawData'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'AFS-Segmentation'))
LSDF_DIR = '/mnt/lsdf' if os.name == 'posix' else "Z:\\"

def convert_fish(fish_number):
    print '----Converting fish #%d' % fish_number
    dir_path = os.path.join(LSDF_DIR, 'grif', 'ANKA_data', '2014', 'XRegioMay2014', 'tomography', 'Phenotyping', 'Recon', 'fish%d', 'Complete', 'Corr')
    dir_path = dir_path % fish_number
    out_path = os.path.join(LSDF_DIR, 'grif', 'Phenotype_medaka', 'Originals', '%s')

    raw_data_stack = create_raw_stack(dir_path, "fish%d_proj_" % fish_number)

    print '----Raw stack creating fish #%d' % fish_number
    raw_data_stack.tofile(out_path % ("fish%d_32bit_%dx%dx%d.raw" % \
            (fish_number, raw_data_stack.shape[2], raw_data_stack.shape[1], raw_data_stack.shape[0])))
    del raw_data_stack

def convert_fish_in_parallel(fish_num_array, core=4):
    t = Timer()

    p = Pool(core)
    p.map(convert_fish, fish_num_array)

    t.elapsed('Fish converting')

def zoom_chunk_fishes(args):
    for arg in args:
        zoom_rotate(arg)

def zoom_in_parallel(fish_num_array, input_dir, output_dir, core=2):
    t = Timer()

    args = []
    for fish_num in fish_num_array:
        args.append(tuple([get_path_by_name(fish_num, input_dir), output_dir]))

    processes = [Process(target=zoom_rotate, args=(ip,op,)) for ip,op in args]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    t.elapsed('Zooming global')

def align_fishes(fish_num_array, input_dir, output_dir):
    for fish_num in fish_num_array:
        t = Timer()

        print 'Aligning started fish%d...' % fish_num

        input_path = get_path_by_name(fish_num, input_dir)

        print "Input: %s" % input_path

        input_data = open_data(input_path)
        aligned_data = align_fish_by_eyes_tail(input_data)

        name, bits, size, ext = parse_filename(input_path)
        output_file = create_filename_with_shape(input_path, aligned_data.shape, prefix="aligned")

        output_path = os.path.join(output_dir, output_file)
        print 'Output: %s' % output_path

        aligned_data.astype('float%d' % bits).tofile(output_path)

        del input_data, aligned_data

        t.elapsed('Aligning')

def zoom_fishes(fish_num_array, input_dir, output_dir):
    for fish_num in fish_num_array:

        t = Timer()

        print 'Zooming started fish%d...' % fish_num

        input_path = get_path_by_name(fish_num, input_dir)

        print "Input: %s" % input_path

        input_data = open_data(input_path)
        zoomed_data = zoom(input_data, 0.5, order=3)

        name, bits, size, ext = parse_filename(input_path)
        output_file = create_filename_with_shape(input_path, zoomed_data.shape)

        output_path = os.path.join(output_dir, output_file)

        print 'Output: %s' % output_path

        zoomed_data.astype('float%d' % bits).tofile(output_path)

        t.elapsed('Zooming')

def zoom_rotate(input_path, output_path, rotate_angle=0, rot_axis='z', in_folder=False):
    t = Timer()

    print "Input: %s" % input_path
    print "Output: %s" % output_path

    input_data = open_data(input_path)

    print 'Zooming started...'
    zoomed_data = zoom(input_data, 0.5, order=3)

    if rot_axis == 'z':
        axes = (2, 1)

    rotated_data = None
    prefix = ""
    if rotate_angle != 0:
        print 'Rotation started...'
        rotated_data = rotate(zoomed_data, rotate_angle, axes=axes, order=3, reshape=False)
        prefix = 'rotated'
    else:
        rotated_data = zoomed_data

    name, bits, size, ext = parse_filename(input_path)
    output_file = create_filename_with_shape(input_path, rotated_data.shape, prefix=prefix)

    if in_folder:
        output_path = os.path.join(output_path, name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    output_path = os.path.join(output_path, output_file)
    print 'Output will be: %s' % output_path

    rotated_data.tofile(output_path)

    t.elapsed('Zoom and rotation: %s' % input_path)

class FishDataEnv:
    def __init__(self, target_project_path, target_input_path, \
            target_input_labels_path, moving_project_path, moving_input_path, \
            moving_fish_num):
        self.fish_num = moving_fish_num

        self.moving_project_path = moving_project_path
        self.moving_input_path = moving_input_path

        self.target_project_path = target_project_path
        self.target_input_path = target_input_path

        self.moving_data_env = DataEnvironment(moving_project_path, moving_input_path)
        self.fixed_data_env = DataEnvironment(target_project_path, target_input_path)

        self.fixed_data_env.set_input_labels_path(target_input_labels_path)
        self.fixed_data_env.set_target_data_path(moving_input_path)

        self.moving_data_env.set_target_data_path(target_input_path)

    def __str__(self):
        text = ''
        text += 'Moving data project path: %s\n' % self.moving_project_path
        text += 'Moving data input path: %s\n' % self.moving_input_path
        text += 'Fixed data project path: %s\n' % self.target_project_path
        text += 'Fixed data input path: %s' % self.target_input_path
        text += 'Fixed data target data path: %s' % self.fixed_data_env.envs['target_data_path']

        return text

def _build_fish_env(reference_fish_num, target_fish_num):
    print os.path.join(OUTPUT_DIR, 'fish%d' % reference_fish_num)
    print get_path_by_name(reference_fish_num, os.path.join(INPUT_DIR, 'fish%d' % reference_fish_num))
    print get_path_by_name(reference_fish_num, os.path.join(INPUT_DIR, 'fish%d' % reference_fish_num), isFindLabels=True)
    print os.path.join(OUTPUT_DIR, 'fish%d' % target_fish_num)
    print get_path_by_name(target_fish_num, os.path.join(INPUT_DIR, 'fish%d' % target_fish_num))
    print target_fish_num
    return FishDataEnv(os.path.join(OUTPUT_DIR, 'fish%d' % reference_fish_num),\
                       get_path_by_name(reference_fish_num, os.path.join(INPUT_DIR, 'fish%d' % reference_fish_num)),\
                       get_path_by_name(reference_fish_num, os.path.join(INPUT_DIR, 'fish%d' % reference_fish_num), isFindLabels=True),\
                       os.path.join(OUTPUT_DIR, 'fish%d' % target_fish_num),\
                       get_path_by_name(reference_fish_num, os.path.join(INPUT_DIR, 'fish%d' % target_fish_num)),\
                       target_fish_num)

def _build_fish_data_paths():
    data = []
    data.append(*_build_fish_env(202, 204))
    return data

def clean_version_run_brain_segmentation_unix():
    fishes_envs = _build_fish_data_paths()

    print 'PEW PEW PEW FIST LAUNCH OF AUTO-BRAIN SEGMENTATION!'
    for fish_env in fishes_envs:
        print '############################# Fish %d ###################################' % fish_env.fish_num
        print fish_env

        #brain_segmentation_nifty(fish_env.fixed_data_env, fish_env.moving_data_env)
        brain_segmentation_ants(fish_env.fixed_data_env, fish_env.moving_data_env)

def scaling_aligning():
    fish_num_array = np.array([200, 204, 215, 223, 226, 228, 230, 231, 233, 238, 243])

    input_dir = os.path.join(LSDF_DIR, 'grif', 'Phenotype_medaka', 'Misc', 'Originals')
    output_zoom_dir = os.path.join(LSDF_DIR, 'grif', 'Phenotype_medaka', 'Misc', 'Originals_scaled')
    output_align_dir = os.path.join(LSDF_DIR, 'grif', 'Phenotype_medaka', 'Misc', 'Originals_aligned')

    zoom_fishes(fish_num_array, input_dir, output_zoom_dir)

    fish_num_array = np.array([230, 231, 233, 238, 243])
    align_fishes(fish_num_array, output_zoom_dir, output_align_dir)


if __name__ == "__main__":
    #run_spine_segmentation("C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish204\\fish204_aligned_32bit_60x207x1220.raw")
    #clean_version_run_brain_segmentation_unix()
    print _build_fish_env(202, 204)

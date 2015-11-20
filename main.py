import sys
import os
import numpy as np
from modules.tools.env import DataEnvironment
from modules.tools.io import create_raw_stack, open_data, create_filename_with_shape, parse_filename
from modules.tools.processing import binarizator
from modules.tools.misc import Timer
from modules.tools.morphology import object_counter, gather_statistics, extract_largest_area_data
from modules.segmentation.eyes import eyes_statistics, eyes_zrange
from modules.segmentation.common import split_fish, align_fish, crop_align_data, brain_segmentation, brain_segmentation_nifty
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.interpolation import zoom, rotate
import pandas as pd
import pdb
#import subproces
import shutil

base_save_path = "C:\\Users\\Administrator\\Documents\\afs_test_out\\%s"

def convert_fish(fish_number):
        dir_path = "Z:\\grif\\ANKA_data\\2014\\XRegioMay2014\\tomography\\Phenotyping\\Recon\\fish%d\\Complete\\Corr" % fish_number

        out_path = "Z:\\grif\\Phenotype_medaka\\Originals\\%s"
        raw_data_stack = create_raw_stack(dir_path, "fish%d_proj_" % fish_number)
        raw_data_stack.tofile(out_path % ("fish%d_32bit_%dx%dx%d.raw" % \
            (fish_number, raw_data_stack.shape[2], raw_data_stack.shape[1], raw_data_stack.shape[0])))
        del raw_data_stack

def rotate_data_fish202():
    output_path = "C:\\Users\\Administrator\\Documents\\AFS_results\\fish%d"

    input_path = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish202\\fish202_32bit_640x640x1996.raw"
    fish_data = np.memmap(input_path, dtype=np.float32, shape=(1996, 640, 640))

    t = Timer()
    a_z = -114.0
    a_y = -2.0
    rotated_data = rotate(fish_data, a_z, axes=(2, 1), order=3, reshape=False)
    rotated_data = rotate(rotated_data, a_y, axes=(0, 2), order=3, reshape=False)
    t.elapsed('Rotation')

    rotated_data.tofile((output_path % 202) + '\\fish202_rotated_order3_32bit_640x640x1996.raw')


#FIX!!!!!!!
def zoom_rotate(input_path, rotate_angle, output_path, rot_axis='z'):
    t = Timer()
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

    if not os.path.exists(output_path % name):
        os.makedirs(output_path % name)

    print 'Output will be: %s' % ((output_path % name) + '\\' + output_file)

    rotated_data.tofile((output_path % name) + '\\' + output_file)

    t.elapsed('Zoom and rotation: %s' % input_path)

def inverse_rotate_zoom_data_fish202():
    output_path = "C:\\Users\\Administrator\\Documents\\AFS_results\\fish%d"
    input_path = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish202\\fish202_labels_8bit_640x640x1996.raw"
    fish_data = np.memmap(input_path, dtype=np.uint8, shape=(1996, 640, 640))

    t = Timer()
    a_y = 2.0
    a_z = 114.0
    rotated_data = rotate(fish_data, a_y, axes=(0, 2), order=0, reshape=False)
    rotated_data = rotate(rotated_data, a_z, axes=(2, 1), order=0, reshape=False)
    t.elapsed('Inverse Rotation')

    rotated_data.tofile((output_path % 202) + '\\fish202_inverse_rotated_labels_8bit_640x640x1996.raw')

    t = Timer()
    zoomed_data = zoom(rotated_data, 2.0, order=0)
    t.elapsed('Scaling')

    output_zoom = ((output_path % 202) + "\\fish202_scaled_rotated_label_8bit_%dx%dx%d.raw") % (zoomed_data.shape[2], zoomed_data.shape[1], zoomed_data.shape[0])
    zoomed_data.tofile(output_zoom)

def convert_data():
    convert_fish(209)
    convert_fish(214)
    convert_fish(215)
    convert_fish(221)
    convert_fish(223)
    convert_fish(224)
    convert_fish(226)
    convert_fish(228)
    convert_fish(230)
    convert_fish(231)
    convert_fish(233)
    convert_fish(235)
    convert_fish(236)
    convert_fish(237)
    convert_fish(238)
    convert_fish(239)
    convert_fish(243)
    convert_fish(244)
    convert_fish(245)

def split_body_test():
    input_path = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish202\\fish202_labels_8bit_640x640x1996.raw"
    fish_data_labels = np.memmap(input_path, dtype=np.uint8, shape=(1996, 640, 640))

    split_fish(fish_data_labels)

def align_fish_test():
    align_fish("","")

def run_segmentation():
    target_project_path = "C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\fish202"
    target_input_path = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish202\\fish202_aligned_32bit_640x640x1996.raw"
    target_input_labels_path = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish202\\fish202_aligned_labels_8bit_640x640x1996.raw"
    target_input_spine_labels_path = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish202\\fish202_aligned_spine_labels_8bit_640x640x1996.raw"

    moving_project_path = "C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\fish204"
    moving_input_path = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish204\\fish204_rotated_32bit_631x631x1992.raw"

    fixed_data_env = DataEnvironment(target_project_path, target_input_path)
    moving_data_env = DataEnvironment(moving_project_path, moving_input_path)

    fixed_data_env.set_input_labels_path(target_input_labels_path)
    fixed_data_env.set_input_spine_labels_path(target_input_spine_labels_path)
    fixed_data_env.set_target_data_path(moving_input_path)

    moving_data_env.set_target_data_path(target_input_path)

    crop_align_data(fixed_data_env, moving_data_env)

def run_brain_segmentation():
    target_project_path = "C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\fish202"
    target_input_path = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish202\\fish202_aligned_32bit_640x640x1996.raw"
    target_input_labels_path = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish202\\fish202_aligned_labels_8bit_640x640x1996.raw"
    target_input_spine_labels_path = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish202\\fish202_aligned_spine_labels_8bit_640x640x1996.raw"

    # moving_project_paths = ["C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\fish204",\
    #                         "C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\fish200",\
    #                         "C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\fish215"]
    # moving_input_paths = ["C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish204\\fish204_rotated_32bit_631x631x1992.raw",\
    #                       "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish200\\fish200_rotated_32bit_573x573x2470.raw",\
    #                       "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish215\\fish215_32bit_640x640x2478.raw"]
    # fish_num = ["204", "200", "215"]

    # moving_project_paths = ["C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\fish200",\
    #                         "C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\fish215"]
    # moving_input_paths = ["C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish200\\fish200_rotated_32bit_573x573x2470.raw",\
    #                       "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish215\\fish215_32bit_640x640x2478.raw"]
    # fish_num = ["200", "215"]

    moving_project_paths = ["C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\fish204"]
    moving_input_paths = ["C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish204\\fish204_rotated_32bit_631x631x1992.raw"]
    fish_num = ["204"]

    print 'PEW PEW PEW FIST LAUNCH OF AUTO-BRAIN SEGMENTATION!'

    for proj_path, input_path, fn in zip(moving_project_paths, moving_input_paths, fish_num):
        print '########################################## Fish %s ##########################################' % fn
        moving_data_env = DataEnvironment(proj_path, input_path)
        fixed_data_env = DataEnvironment(target_project_path, target_input_path)

        fixed_data_env.set_input_labels_path(target_input_labels_path)
        fixed_data_env.set_input_spine_labels_path(target_input_spine_labels_path)
        fixed_data_env.set_target_data_path(input_path)

        moving_data_env.set_target_data_path(target_input_path)

        print 'Moving data project path: %s' % proj_path
        print 'Moving data input path: %s' % proj_path

        print 'Fixed data project path: %s' % target_project_path
        print 'Fixed data input path: %s' % target_input_path
        print 'Fixed data target data path: %s' % fixed_data_env.envs['target_data_path']

        brain_segmentation(fixed_data_env, moving_data_env)

def run_brain_segmentation_unix():
    target_project_path = "/home/rshkarin/ANKA_work/AFS-playground/Segmentation/fish202"
    target_input_path = "/home/rshkarin/ANKA_work/AFS-playground/ProcessedMedaka/fish202/fish202_aligned_32bit_640x640x1996.raw"
    target_input_labels_path = "/home/rshkarin/ANKA_work/AFS-playground/ProcessedMedaka/fish202/fish202_aligned_labels_8bit_640x640x1996.raw"
    target_input_spine_labels_path = "/home/rshkarin/ANKA_work/AFS-playground/ProcessedMedaka/fish202/fish202_aligned_spine_labels_8bit_640x640x1996.raw"

    # moving_project_paths = ["C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\fish204",\
    #                         "C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\fish200",\
    #                         "C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\fish215"]
    # moving_input_paths = ["C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish204\\fish204_rotated_32bit_631x631x1992.raw",\
    #                       "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish200\\fish200_rotated_32bit_573x573x2470.raw",\
    #                       "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish215\\fish215_32bit_640x640x2478.raw"]
    # fish_num = ["204", "200", "215"]

    # moving_project_paths = ["C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\fish200",\
    #                         "C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\fish215"]
    # moving_input_paths = ["C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish200\\fish200_rotated_32bit_573x573x2470.raw",\
    #                       "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish215\\fish215_32bit_640x640x2478.raw"]
    # fish_num = ["200", "215"]

    moving_project_paths = ["/home/rshkarin/ANKA_work/AFS-playground/Segmentation/fish204"]
    moving_input_paths = ["/home/rshkarin/ANKA_work/AFS-playground/ProcessedMedaka/fish204/fish204_rotated_32bit_631x631x1992.raw"]
    fish_num = ["204"]

    print 'PEW PEW PEW FIST LAUNCH OF AUTO-BRAIN SEGMENTATION!'

    for proj_path, input_path, fn in zip(moving_project_paths, moving_input_paths, fish_num):
        print '########################################## Fish %s ##########################################' % fn
        moving_data_env = DataEnvironment(proj_path, input_path)
        fixed_data_env = DataEnvironment(target_project_path, target_input_path)

        fixed_data_env.set_input_labels_path(target_input_labels_path)
        fixed_data_env.set_input_spine_labels_path(target_input_spine_labels_path)
        fixed_data_env.set_target_data_path(input_path)

        moving_data_env.set_target_data_path(target_input_path)

        print 'Moving data project path: %s' % proj_path
        print 'Moving data input path: %s' % proj_path

        print 'Fixed data project path: %s' % target_project_path
        print 'Fixed data input path: %s' % target_input_path
        print 'Fixed data target data path: %s' % fixed_data_env.envs['target_data_path']

        brain_segmentation_nifty(fixed_data_env, moving_data_env)

if __name__ == "__main__":
    run_brain_segmentation_unix()
    # run_brain_segmentation()
    # convert_fish(200)
    # convert_fish(215)
    #
    # output_path = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\%s"
    #
    # print "Copying Z:\\grif\\Phenotype_medaka\\Originals\\fish200_32bit_1145x1145x4939.raw"
    # shutil.copy("Z:\\grif\\Phenotype_medaka\\Originals\\fish200_32bit_1145x1145x4939.raw", "C:\\Users\\Administrator\\Documents\\ProcessedMedaka")
    #
    # print "Copying Z:\\grif\\Phenotype_medaka\\Originals\\fish215_32bit_1280x1280x4955.raw"
    # shutil.copy("Z:\\grif\\Phenotype_medaka\\Originals\\fish215_32bit_1280x1280x4955.raw", "C:\\Users\\Administrator\\Documents\\ProcessedMedaka")

    # zoom_rotate("Z:\\grif\\Phenotype_medaka\\Originals\\fish200_32bit_1145x1145x4939.raw", -90, output_path, rot_axis='z')
    # zoom_rotate("Z:\\grif\\Phenotype_medaka\\Originals\\fish215_32bit_1280x1280x4955.raw", 0, output_path, rot_axis='z' )

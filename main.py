import sys
import os
import numpy as np
from modules.tools.env import DataEnvironment
from modules.tools.io import create_raw_stack, open_data
from modules.tools.processing import binarizator
from modules.tools.misc import Timer
from modules.tools.morphology import object_counter, gather_statistics, extract_largest_area_data
from modules.segmentation.eyes import eyes_statistics, eyes_zrange
from modules.segmentation.common import split_fish, align_fish, crop_align_data
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.interpolation import zoom, rotate
import pandas as pd
import pdb
#import subproces

base_save_path = "C:\\Users\\Administrator\\Documents\\afs_test_out\\%s"

def convert_fish(fish_number):
        dir_path = "Z:\\grif\\ANKA_data\\2014\\XRegioMay2014\\tomography\\Phenotyping\\Recon\\fish%d\\Complete\\Corr" % fish_number

        out_path = "Z:\\grif\\Phenotype_medaka\\Originals\\%s"
        raw_data_stack = create_raw_stack(dir_path, "fish%d_proj_" % fish_number)
        raw_data_stack.tofile(out_path % ("fish%d_32bit_%dx%dx%d.raw" % \
            (fish_number, raw_data_stack.shape[2], raw_data_stack.shape[1], \
                raw_data_stack.shape[0])))
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

if __name__ == "__main__":
    target_project_path = "C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\fish202"
    target_input_path = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish202\\fish202_aligned_32bit_640x640x1996.raw"
    target_input_labels_path = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish202\\fish202_aligned_labels_8bit_640x640x1996.raw"

    moving_project_path = "C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\fish204"
    moving_input_path = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish204\\fish204_rotated_32bit_631x631x1992.raw"

    fixed_data_env = DataEnvironment(target_project_path, target_input_path)
    moving_data_env = DataEnvironment(moving_project_path, moving_input_path)

    fixed_data_env.set_input_labels_path(target_input_labels_path)
    fixed_data_env.set_target_data_path(moving_input_path)
    
    moving_data_env.set_target_data_path(target_input_path)

    crop_align_data(fixed_data_env, moving_data_env)

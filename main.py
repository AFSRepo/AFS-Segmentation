import sys
import os
import numpy as np
from modules.tools.env import DataEnvironment
from multiprocessing import Process, Pool
from modules.tools.io import create_raw_stack, open_data, create_filename_with_shape, parse_filename, get_path_by_name
from modules.tools.io import INPUT_DIR, OUTPUT_DIR, LSDF_DIR
from modules.tools.processing import binarizator, align_fish_by_eyes_tail
from modules.tools.processing import convert_fish, convert_fish_in_parallel, zoom_chunk_fishes, zoom_in_parallel
from modules.tools.processing import align_fishes, zoom_fishes, zoom_rotate, downsample_data
from modules.tools.processing import scaling_aligning, produce_aligned_fish, get_aligned_fish, get_fish_folder
from modules.tools.processing import get_fish_project_folder, get_fish_path
from modules.tools.misc import Timer
from modules.tools.morphology import object_counter, gather_statistics, extract_largest_area_data
from modules.segmentation.eyes import eyes_statistics, eyes_zrange
from modules.segmentation.spine import run_spine_segmentation
from modules.segmentation.common import split_fish, align_fish
from modules.segmentation.common import crop_align_data, brain_segmentation, \
brain_segmentation_nifty, brain_segmentation_ants, full_body_registration_ants
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.interpolation import zoom, rotate
import pandas as pd
import pdb
import shutil

#TODO:
# 1. Sparate and pick up files by zoom level somehow.

class FishDataEnv:
    def __init__(self, reference_project_path, reference_input_path, \
            reference_input_labels_path, target_project_path, target_input_path, \
            reference_fish_num, target_fish_num):
        self.target_fish_num = target_fish_num
        self.reference_fish_num = reference_fish_num

        self.target_project_path = target_project_path
        self.target_input_path = target_input_path

        self.reference_project_path = reference_project_path
        self.reference_input_path = reference_input_path

        self.target_data_env = DataEnvironment(target_project_path, target_input_path, target_fish_num)
        self.reference_data_env = DataEnvironment(reference_project_path, reference_input_path, reference_fish_num)

        self.reference_data_env.set_input_labels_path(reference_input_labels_path)
        self.reference_data_env.set_target_data_path(target_input_path)

        self.target_data_env.set_target_data_path(reference_input_path)

    def __str__(self):
        text = ''
        text += 'Target data project path: %s\n' % self.target_project_path
        text += 'Target data input path: %s\n' % self.target_input_path
        text += 'Reference data project path: %s\n' % self.reference_project_path
        text += 'Reference data input path: %s\n' % self.reference_input_path
        text += 'Reference data target data path: %s\n' % self.reference_data_env.envs['target_data_path']

        return text

def _build_fish_env(reference_fish_num, target_fish_num, zoom_level=2):
    return FishDataEnv(get_fish_project_folder(reference_fish_num),\
                       get_fish_path(reference_fish_num, zoom_level=zoom_level),\
                       get_fish_path(reference_fish_num, zoom_level=zoom_level, isLabel=True),\
                       get_fish_project_folder(target_fish_num),\
                       get_fish_path(target_fish_num, zoom_level=zoom_level),\
                       reference_fish_num, \
                       target_fish_num)

def _build_fish_data_paths():
    data = []
    data.append(_build_fish_env(202, 243, zoom_level=2))
    data.append(_build_fish_env(202, 204, zoom_level=2))
    data.append(_build_fish_env(233, 238, zoom_level=2))
    data.append(_build_fish_env(233, 230, zoom_level=2))
    data.append(_build_fish_env(233, 231, zoom_level=2))
    data.append(_build_fish_env(200, 215, zoom_level=2))
    data.append(_build_fish_env(233, 223, zoom_level=2))
    data.append(_build_fish_env(233, 226, zoom_level=2))
    return data

def clean_version_run_brain_segmentation_unix(useAnts=True):
    fishes_envs = _build_fish_data_paths()

    print 'PEW PEW PEW FIST LAUNCH OF AUTO-BRAIN SEGMENTATION!'
    for fish_env in fishes_envs:
        print '############################# Fish %d -> Fish %d ###################################' % \
                                (fish_env.reference_fish_num, fish_env.target_fish_num)

        if useAnts:
            #full_body_registration_ants(fish_env.reference_data_env, fish_env.target_data_env)
            brain_segmentation_ants(fish_env.reference_data_env, fish_env.target_data_env)
        else:
            brain_segmentation_nifty(fish_env.reference_data_env, fish_env.target_data_env)

if __name__ == "__main__":
    #run_spine_segmentation("C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish204\\fish204_aligned_32bit_60x207x1220.raw")
    clean_version_run_brain_segmentation_unix()
    #input_aligned_data, input_aligned_data_label = get_aligned_fish(202, zoom_level=8, min_zoom_level=4)
    #print input_aligned_data.dtype
    #print input_aligned_data_label.dtype
    #output = get_aligned_fish(204, zoom_level=8, min_zoom_level=4)
    #print str(output)

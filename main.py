import sys
import os
import numpy as np
import nibabel as nib
from modules.tools.io import create_raw_stack
from modules.tools.processing import binarizator
from modules.tools.misc import Timer
from modules.tools.morphology import object_counter, gather_statistics, extract_largest_area_data
from modules.segmentation.eyes import eyes_statistics, eyes_zrange
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.interpolation import zoom, rotate
import pandas as pd
import pdb
#import subproces

base_save_path = "C:\\Users\\Administrator\\Documents\\afs_test_out\\%s"

def extract_net_volume(input_path, output_path, filepath, size_zyx, fish_num):
    #filepath = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish202\\fish202_aligned_32bit_640x640x1996.raw"
    #filepath_scaled = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish202\\fish202_aligned_32bit_320x320x998.raw"
    #stack_data = np.memmap(filepath_scaled, dtype=np.float32, shape=(998,320,320))
    in_path = os.path.join((input_path % fish_num), filepath)

    out_path = (output_path % fish_num)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    timer_total = Timer()

    timer = Timer()
    print 'Reading...'
    stack_data = np.memmap(in_path, dtype=np.float32, shape=size_zyx)
    timer.elapsed('Reading')

    timer = Timer()
    print 'Binarizing...'
    binarized_stack, bbox, eyes_stats = binarizator(stack_data)
    binarized_stack.tofile(os.path.join(out_path, "binarized_stack.raw"))
    eyes_stats.to_csv(os.path.join(out_path, "eyes_stats.csv"))
    timer.elapsed('Binarizing')

    timer = Timer()
    print 'Object counting...'
    binary_stack_stats = object_counter(binarized_stack)
    binary_stack_stats.to_csv(os.path.join(out_path, "binary_stack_stats.csv"))
    timer.elapsed('Object counting')

    timer = Timer()
    print 'Big volume extraction...'
    largest_volume_region = extract_largest_area_data(stack_data, binary_stack_stats)
    timer.elapsed('Big volume extraction')

    timer = Timer()
    print 'Saving...'
    vol_shape = largest_volume_region.shape
    filepath_scaled_extracted = os.path.join(out_path, ("fish%d_aligned_extracted_32bit_%dx%dx%d.raw" % (fish_num, vol_shape[2], vol_shape[1], vol_shape[0])))
    largest_volume_region.tofile(filepath_scaled_extracted)
    timer.elapsed('Saving')

    timer_total.elapsed('Total')

    return largest_volume_region

def save_nifti():
    filepath = "fish202_aligned_extracted_32bit_367x387x1834.raw"
    ext_stack_data = np.memmap(base_save_path % filepath, dtype=np.float32, shape=(367,387,1834), order='F')
    nifti_stack_data = nib.Nifti1Image(ext_stack_data, np.eye(4))
    nib.save(nifti_stack_data, base_save_path % ("fish202_aligned_extracted_32bit_%dx%dx%d.nii.gz" % (1834,387,367)))

#def register_volumes():
#    os.environ["ANTSPATH"] = "C:\\Users\\Administrator\\Documents\\ANTs"
#    subprocess.call(["bash","antsRegistrationSyN.sh"])

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

def main():
    output_path = "C:\\Users\\Administrator\\Documents\\AFS_results\\fish%d"
    input_path = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish%d"

    fish202 = extract_net_volume(input_path, output_path, "fish202_aligned_32bit_1280x1280x3992.raw", (3992, 1280, 1280), 202)
    fish204 = extract_net_volume(input_path, output_path, "fish204_32bit_1261x1261x3983.raw", (3983, 1261, 1261), 204)

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


if __name__ == "__main__":
    #rotate_data_fish202()
    #inverse_rotate_zoom_data_fish202()

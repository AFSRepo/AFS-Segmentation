import os
import sys
import subprocess as subpr
import numpy as np
from modules.segmentation.eyes import eyes_statistics, eyes_zrange
from modules.tools.morphology import object_counter, gather_statistics, extract_largest_area_data
from modules.tools.io import get_filename, open_data, save_as_nifti, check_files, parse_filename
from modules.tools.processing import binarizator
from modules.tools.env import DataEnvironment
from modules.tools.misc import Timer
from scipy.ndimage.interpolation import zoom, rotate

def produce_cropped_data(data_env):
    phase_name = 'extracted_input_data_path_niigz'
    phase_name_zoomed = 'zoomed_0p5_extracted_input_data_path_niigz'

    data_env.load()

    ext_volume_niigz_path = None
    zoomed_ext_volume_niigz_path = None

    ext_volume_labels_niigz_path = None
    zoomed_ext_volume_labels_niigz_path = None

    if not data_env.is_entry_exists(phase_name):
        t = Timer()

        if data_env.get_input_path():
            input_data = open_data(data_env.get_input_path())

            ext_volume, bbox = extract_effective_volume(input_data, bb_side_offset=50)
            zoomed_0p5_ext_volume = zoom(ext_volume, 0.5, order=3)

            ext_volume_path = data_env.get_new_volume_path(ext_volume.shape, 'extracted')
            ext_volume_niigz_path = data_env.get_new_volume_niigz_path(ext_volume.shape, 'extracted')

            zoomed_ext_volume_path = data_env.get_new_volume_path(zoomed_0p5_ext_volume.shape, 'zoomed_0p5_extracted')
            zoomed_ext_volume_niigz_path = data_env.get_new_volume_niigz_path(zoomed_0p5_ext_volume.shape, 'zoomed_0p5_extracted')

            save_as_nifti(ext_volume, ext_volume_niigz_path)
            save_as_nifti(zoomed_0p5_ext_volume, zoomed_ext_volume_niigz_path)

            ext_volume.tofile(ext_volume_path)
            zoomed_0p5_ext_volume.tofile(zoomed_ext_volume_path)

            if data_env.get_input_labels_path():
                input_data_labels = open_data(data_env.get_input_labels_path())

                ext_volume_labels = input_data_labels[bbox]
                zoomed_0p5_ext_volume_labels = zoom(ext_volume_labels, 0.5, order=0)

                ext_volume_labels_path = data_env.get_new_volume_labels_path(ext_volume_labels.shape, 'extracted')
                ext_volume_labels_niigz_path = data_env.get_new_volume_labels_niigz_path(ext_volume_labels.shape, 'extracted')

                zoomed_ext_volume_labels_path = data_env.get_new_volume_labels_path(zoomed_0p5_ext_volume_labels.shape, 'zoomed_0p5_extracted')
                zoomed_ext_volume_labels_niigz_path = data_env.get_new_volume_labels_niigz_path(zoomed_0p5_ext_volume_labels.shape, 'zoomed_0p5_extracted')

                save_as_nifti(ext_volume_labels, ext_volume_labels_niigz_path)
                save_as_nifti(zoomed_0p5_ext_volume_labels, zoomed_ext_volume_labels_niigz_path)

                ext_volume_labels.tofile(ext_volume_labels_path)
                zoomed_0p5_ext_volume_labels.tofile(zoomed_ext_volume_labels_path)

        else:
            print 'There\'s no input data'

        data_env.save()

        t.elapsed('Data cropping is finished')
    else:
        print 'Files of \'%s\' phase are already in working directory: %s' % (phase_name, data_env.get_working_path())

    return {'scaled_0p5_extracted': zoomed_ext_volume_niigz_path, 'extracted': ext_volume_niigz_path, \
            'scaled_0p5_extracted_labels': zoomed_ext_volume_labels_niigz_path, 'extracted_labels': ext_volume_labels_niigz_path}

def crop_align_data(fixed_data_env, moving_data_env):
    fixed_data_env.load()
    moving_data_env.load()

    # Crop the raw data
    fixed_data_results = produce_cropped_data(fixed_data_env)
    moving_data_results = produce_cropped_data(moving_data_env)

    fixed_data_env.save()
    moving_data_env.save()

    #generate_stats(fixed_data_env)
    #generate_stats(moving_data_env)

    # Pre-alignment fish1 to fish_aligned
    ants_prefix_prealign = 'pre_alignment'
    ants_prealign_paths = moving_data_env.get_aligned_data_paths(ants_prefix_prealign)
    ants_prealign_names = moving_data_env.get_aligned_data_paths(ants_prefix_prealign, produce_paths=False)

    working_env_prealign = moving_data_env
    fixed_image_path_prealign = fixed_data_env.envs['zoomed_0p5_extracted_input_data_path_niigz']
    fixed_image_path_prealign_raw = fixed_data_env.envs['zoomed_0p5_extracted_input_data_path']
    moving_image_path_prealign = moving_data_env.envs['zoomed_0p5_extracted_input_data_path_niigz']
    output_name_prealign = ants_prealign_names['out_name']
    warped_path_prealign = ants_prealign_paths['warped']
    iwarped_path_prealign = ants_prealign_paths['iwarp']

    if not os.path.exists(warped_path_prealign):
        align_fish(working_env_prealign, fixed_image_path_prealign, moving_image_path_prealign, \
                    output_name_prealign, warped_path_prealign, iwarped_path_prealign, reg_prefix=ants_prefix_prealign)


        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Data is already prealigned"

    # Registration of fish_aligned to fish1
    ants_prefix_sep = 'parts_separation'
    ants_separation_paths = fixed_data_env.get_aligned_data_paths(ants_prefix_sep)
    working_env_sep = fixed_data_env
    fixed_image_path_sep = warped_path_prealign
    moving_image_path_sep = fixed_image_path_prealign
    output_name_sep = ants_separation_paths['out_name']
    warped_path_sep = ants_separation_paths['warped']
    iwarped_path_sep = ants_separation_paths['iwarp']

    if not os.path.exists(warped_path_sep):
        align_fish(working_env_sep, fixed_image_path_sep, moving_image_path_sep, \
                   output_name_sep, warped_path_sep, iwarped_path_sep, \
                   reg_prefix=ants_prefix_sep, use_syn=True)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Data is already registered for separation"

    # Transforming labels of fish_aligned to fish1
    wokring_env_tr = moving_data_env
    ref_image_path_tr = ants_prealign_paths['warped']
    transformation_path_tr = ants_separation_paths['gen_affine']
    labels_image_path_tr = fixed_data_env.envs['zoomed_0p5_extracted_input_data_labels_path_niigz']
    test_data = open_data(fixed_image_path_prealign_raw)

    __, __, new_size, __ = parse_filename(fixed_image_path_prealign_raw)

    transformation_output_tr = moving_data_env.get_new_volume_niigz_path(new_size, 'zoomed_0p5_extracted_labels', bits=8)
    reg_prefix_tr = 'label_deforming'

    if not os.path.exists(transformation_output_tr):
        apply_transform_fish(wokring_env_tr, ref_image_path_tr, transformation_path_tr, labels_image_path_tr, transformation_output_tr, reg_prefix=reg_prefix_tr)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Data is already transformed"

    #Separate head and tail of fixed image
    print "Fish separation (Fixed image)..."
    aligned_data_fixed = open_data(fixed_image_path_prealign)
    aligned_data_labels_fixed = open_data(labels_image_path_tr)

    separation_pos_fixed, abdomen_label_fixed_full, head_label_fixed_full = find_separation_pos(aligned_data_labels_fixed)

    abdomen_data_part_fixed, head_data_part_fixed = split_fish_by_pos(aligned_data_fixed, separation_pos_fixed, overlap=20)
    abdomen_label_fixed, _ = split_fish_by_pos(abdomen_label_fixed_full, separation_pos_fixed, overlap=20)
    _, head_label_fixed = split_fish_by_pos(head_label_fixed_full, separation_pos_fixed, overlap=20)

    abdomen_data_part_fixed_niigz_path = fixed_data_env.get_new_volume_niigz_path(abdomen_data_part_fixed.shape, 'zoomed_0p5_abdomen')
    if not os.path.exists(abdomen_data_part_fixed_niigz_path):
        save_as_nifti(abdomen_data_part_fixed, abdomen_data_part_fixed_niigz_path)

    head_data_part_fixed_niigz_path = fixed_data_env.get_new_volume_niigz_path(head_data_part_fixed.shape, 'zoomed_0p5_head')
    if not os.path.exists(head_data_part_fixed_niigz_path):
        save_as_nifti(head_data_part_fixed, head_data_part_fixed_niigz_path)

    abdomen_data_part_labels_fixed_niigz_path = fixed_data_env.get_new_volume_niigz_path(abdomen_label_fixed.shape, 'zoomed_0p5_abdomen_labels')
    if not os.path.exists(abdomen_data_part_labels_fixed_niigz_path):
        save_as_nifti(abdomen_label_fixed, abdomen_data_part_labels_fixed_niigz_path)

    head_data_part_labels_fixed_niigz_path = fixed_data_env.get_new_volume_niigz_path(head_label_fixed.shape, 'zoomed_0p5_head_labels')
    if not os.path.exists(head_data_part_labels_fixed_niigz_path):
        save_as_nifti(head_label_fixed, head_data_part_labels_fixed_niigz_path)

    print abdomen_data_part_labels_fixed_niigz_path
    print head_data_part_labels_fixed_niigz_path

    fixed_data_env.save()

    #Separate head and tail of moving image
    print "Fish separation (Moving image)..."
    aligned_data_moving = open_data(ants_prealign_paths['warped'])
    aligned_data_labels_moving = open_data(transformation_output_tr)

    separation_pos_moving, abdomen_label_moving_full, head_label_moving_full = find_separation_pos(aligned_data_labels_moving)

    abdomen_data_part_moving, head_data_part_moving = split_fish_by_pos(aligned_data_moving, separation_pos_moving, overlap=20)
    abdomen_label_moving, _ = split_fish_by_pos(abdomen_label_moving_full, separation_pos_moving, overlap=20)
    _, head_label_moving = split_fish_by_pos(head_label_moving_full, separation_pos_moving, overlap=20)

    abdomen_data_part_moving_niigz_path = moving_data_env.get_new_volume_niigz_path(abdomen_data_part_moving.shape, 'zoomed_0p5_abdomen')
    if not os.path.exists(abdomen_data_part_moving_niigz_path):
        save_as_nifti(abdomen_data_part_moving, abdomen_data_part_moving_niigz_path)

    head_data_part_moving_niigz_path = moving_data_env.get_new_volume_niigz_path(head_data_part_moving.shape, 'zoomed_0p5_head')
    if not os.path.exists(head_data_part_moving_niigz_path):
        save_as_nifti(head_data_part_moving, head_data_part_moving_niigz_path)

    abdomen_data_part_labels_moving_niigz_path = moving_data_env.get_new_volume_niigz_path(abdomen_label_moving.shape, 'zoomed_0p5_abdomen_labels')
    if not os.path.exists(abdomen_data_part_labels_moving_niigz_path):
        save_as_nifti(abdomen_label_moving, abdomen_data_part_labels_moving_niigz_path)

    head_data_part_labels_moving_niigz_path = moving_data_env.get_new_volume_niigz_path(head_label_moving.shape, 'zoomed_0p5_head_labels')
    if not os.path.exists(head_data_part_labels_moving_niigz_path):
        save_as_nifti(head_label_moving, head_data_part_labels_moving_niigz_path)

    print abdomen_data_part_labels_moving_niigz_path
    print head_data_part_labels_moving_niigz_path

    moving_data_env.save()

    #Register fixed head to moving one
    ants_prefix_head_reg = 'head_registration'
    ants_head_reg_paths = fixed_data_env.get_aligned_data_paths(ants_prefix_head_reg)
    working_env_head_reg = fixed_data_env
    fixed_image_path_head_reg = head_data_part_moving_niigz_path
    moving_image_path_head_reg = head_data_part_fixed_niigz_path
    output_name_head_reg = ants_head_reg_paths['out_name']
    warped_path_head_reg = ants_head_reg_paths['warped']
    iwarped_path_head_reg = ants_head_reg_paths['iwarp']

    if not os.path.exists(warped_path_head_reg):
        align_fish(working_env_head_reg, fixed_image_path_head_reg, moving_image_path_head_reg, \
                   output_name_head_reg, warped_path_head_reg, iwarped_path_head_reg, \
                   reg_prefix=ants_prefix_head_reg, use_syn=True)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Head of the fixed data is already registered to the head of moving one"

    # Transforming labels of head of fixed fish to the head of moving one
    wokring_env_htr = moving_data_env
    ref_image_path_htr = head_data_part_moving_niigz_path
    transformation_path_htr = ants_head_reg_paths['gen_affine']
    labels_image_path_htr = head_data_part_labels_fixed_niigz_path
    test_data_htr = open_data(ref_image_path_htr)
    transformation_output_htr = moving_data_env.get_new_volume_niigz_path(test_data_htr.shape, 'zoomed_0p5_head_brain_labels', bits=8)
    reg_prefix_htr = 'head_label_deforming'

    if not os.path.exists(transformation_output_htr):
        apply_transform_fish(wokring_env_tr, ref_image_path_htr, transformation_path_htr, labels_image_path_htr, transformation_output_htr, reg_prefix=reg_prefix_htr)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Head data is already transformed"

    # Extract moving brain volume
    head_brain_label_moving = open_data(transformation_output_htr)
    head_brain_data_moving = open_data(ref_image_path_htr)
    brain_data_volume_moving, _ = extract_largest_volume_by_label(head_brain_data_moving, head_brain_label_moving, bb_side_offset=10)
    brain_data_volume_moving_niigz_path = moving_data_env.get_new_volume_niigz_path(brain_data_volume_moving.shape, 'zoomed_0p5_head_extracted_brain')

    print brain_data_volume_moving_niigz_path

    if not os.path.exists(brain_data_volume_moving_niigz_path):
        save_as_nifti(brain_data_volume_moving, brain_data_volume_moving_niigz_path)

    # Extract fixed brain volume
    head_brain_label_fixed = open_data(labels_image_path_htr)
    head_brain_data_fixed = open_data(moving_image_path_head_reg)
    brain_data_volume_fixed, brain_fixed_bbox = extract_largest_volume_by_label(head_brain_data_fixed, head_brain_label_fixed, bb_side_offset=10)
    brain_data_labels_volume_fixed = head_brain_label_fixed[brain_fixed_bbox]

    brain_data_volume_fixed_niigz_path = fixed_data_env.get_new_volume_niigz_path(brain_data_volume_fixed.shape, 'zoomed_0p5_head_extracted_brain')
    brain_data_labels_volume_fixed_niigz_path = fixed_data_env.get_new_volume_niigz_path(brain_data_labels_volume_fixed.shape, 'zoomed_0p5_head_extracted_brain_labels')

    print brain_data_volume_fixed_niigz_path
    print brain_data_labels_volume_fixed_niigz_path

    if not os.path.exists(brain_data_volume_fixed_niigz_path):
        save_as_nifti(brain_data_volume_fixed, brain_data_volume_fixed_niigz_path)

    if not os.path.exists(brain_data_labels_volume_fixed_niigz_path):
        save_as_nifti(brain_data_labels_volume_fixed, brain_data_labels_volume_fixed_niigz_path)

    # Register the fixed brain to the moving one
    ants_prefix_head_brain_reg = 'head_brain_registration'
    ants_head_brain_reg_paths = fixed_data_env.get_aligned_data_paths(ants_prefix_head_brain_reg)
    working_env_head_brain_reg = fixed_data_env
    fixed_image_path_head_brain_reg = brain_data_volume_moving_niigz_path
    moving_image_path_head_brain_reg = brain_data_volume_fixed_niigz_path
    output_name_head_brain_reg = ants_head_brain_reg_paths['out_name']
    warped_path_head_brain_reg = ants_head_brain_reg_paths['warped']
    iwarped_path_head_brain_reg = ants_head_brain_reg_paths['iwarp']

    if not os.path.exists(warped_path_head_brain_reg):
        align_fish(working_env_head_brain_reg, fixed_image_path_head_brain_reg, moving_image_path_head_brain_reg, \
                   output_name_head_brain_reg, warped_path_head_brain_reg, iwarped_path_head_brain_reg, \
                   reg_prefix=ants_prefix_head_brain_reg, use_syn=True, small_volume=True)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "The brain of the head of the fixed data is already registered to the brain of the head of moving one"

    # Transforming labels of the brain of head of fixed fish to the brain of the head of moving one
    wokring_env_brain_tr = moving_data_env
    ref_image_path_brain_tr = brain_data_volume_moving_niigz_path
    transformation_path_brain_tr = ants_head_brain_reg_paths['gen_affine']
    labels_image_path_brain_tr = brain_data_labels_volume_fixed_niigz_path
    test_data_brain_tr = open_data(ref_image_path_brain_tr)
    transformation_output_brain_tr = moving_data_env.get_new_volume_niigz_path(test_data_brain_tr.shape, 'zoomed_0p5_head_extracted_brain_labels', bits=8)
    reg_prefix_brain_tr = 'head_brain_label_deforming'

    if not os.path.exists(transformation_output_brain_tr):
        apply_transform_fish(wokring_env_brain_tr, ref_image_path_brain_tr, transformation_path_brain_tr,\
                             labels_image_path_brain_tr, transformation_output_brain_tr, reg_prefix=reg_prefix_brain_tr)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "The brain of the fixed head data is already transformed"


def extract_largest_volume_by_label(stack_data, stack_labels, bb_side_offset=0):
    stack_stats, _ = object_counter(stack_labels)
    largest_volume_region, bbox = extract_largest_area_data(stack_data, stack_stats, bb_side_offset)

    return largest_volume_region, bbox

def extract_effective_volume(stack_data, eyes_stats=None, bb_side_offset=0):
    timer_total = Timer()

    timer = Timer()
    print 'Binarizing...'
    binarized_stack, bbox, eyes_stats = binarizator(stack_data)
    timer.elapsed('Binarizing')

    timer = Timer()
    print 'Object counting...'
    binary_stack_stats = object_counter(binarized_stack)
    timer.elapsed('Object counting')

    timer = Timer()
    print 'Big volume extraction...'
    largest_volume_region, bbox = extract_largest_area_data(stack_data, binary_stack_stats, bb_side_offset)
    timer.elapsed('Big volume extraction')

    timer_total.elapsed('Total')

    return largest_volume_region, bbox

def generate_stats(data_env):
    data_env.load()

    glob_stats = 'input_data_global_statistics'
    eyes_stats = 'input_data_eyes_statistics'

    if not data_env.is_entry_exists(glob_stats):
        input_data = open_data(data_env.envs['extracted_input_data_path'])

        print 'Global statistics...'
        t = Timer()

        stack_statistics, _ = gather_statistics(input_data)
        global_stats_path = data_env.get_statistic_path('global')
        stack_statistics.to_csv(global_stats_path)

        t.elapsed('Gathering statistics')

        data_env.save()
    else:
        print "Global statistics is already gathered: %s" % data_env.envs[glob_stats]

    if not data_env.is_entry_exists(eyes_stats):
        input_data = open_data(data_env.envs['extracted_input_data_path'])

        print 'Filtering eyes\' statistics...'
        t = Timer()

        eyes_stats = eyes_statistics(stack_statistics)
        eyes_stats_path = data_env.get_statistic_path('eyes')
        eyes_stats.to_csv(eyes_stats_path)

        t.elapsed('Eyes statistics filtered')

        data_env.save()
    else:
        print "Eyes statistics is already gathered: %s" % data_env.envs[eyes_stats]

def align_fish(working_env, fixed_image_path, moving_image_path, output_name, warped_path, iwarped_path, reg_prefix=None, use_syn=False, small_volume=False):

    working_path = working_env.get_working_path()
    os.environ["ANTSPATH"] = working_env.ANTSPATH

    args_fmt = {'out_name': output_name, 'warped_path': warped_path, 'iwarped_path': iwarped_path, 'fixedImagePath': fixed_image_path, 'movingImagePath': moving_image_path}

    app = None

    if not use_syn:
        app = 'antsRegistration --dimensionality 3 --float 1 --output [{out_name},{warped_path},{iwarped_path}] --interpolation BSpline --use-histogram-matching 0 --initial-moving-transform [{fixedImagePath},{movingImagePath},1] --transform Rigid[0.01] --metric MI[{fixedImagePath},{movingImagePath},1,32,Regular,0.25] --convergence [1000x500x250x10,1e-8,10] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox'.format(**args_fmt)
    else:
        if not small_volume:
            app = 'antsRegistration --dimensionality 3 --float 1 --output [{out_name},{warped_path},{iwarped_path}] --interpolation BSpline --use-histogram-matching 0 --initial-moving-transform [{fixedImagePath},{movingImagePath},1] --transform Rigid[0.1] --metric MI[{fixedImagePath},{movingImagePath},1,32,Regular,0.25] --convergence [1000x500x250x100,1e-6,10] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox --transform Affine[0.1] --metric MI[{fixedImagePath},{movingImagePath},1,32,Regular,0.25] --convergence [1000x500x250x100,1e-6,10] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox --transform SyN[0.1,3,0] --metric CC[{fixedImagePath},{movingImagePath},1,4] --convergence [200x100x50x5,1e-6,10] --shrink-factors 6x4x2x1 --smoothing-sigmas 3x2x1x0vox'.format(**args_fmt)
        else:
            app = 'antsRegistration --dimensionality 3 --float 1 --output [{out_name},{warped_path},{iwarped_path}] --interpolation BSpline --use-histogram-matching 0 --initial-moving-transform [{fixedImagePath},{movingImagePath},1] --transform Rigid[0.1] --metric MI[{fixedImagePath},{movingImagePath},1,32,Regular,0.25] --convergence [1000x500x250x100,1e-6,10] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox --transform Affine[0.1] --metric MI[{fixedImagePath},{movingImagePath},1,32,Regular,0.25] --convergence [1000x500x250x100,1e-6,10] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox --transform SyN[0.1,3,0] --metric CC[{fixedImagePath},{movingImagePath},1,10] --metric MI[{fixedImagePath},{movingImagePath},1,32,Regular,0.25] --convergence [200x150x20x5,1e-6,10] --shrink-factors 6x4x2x1 --smoothing-sigmas 3x2x1x0vox'.format(**args_fmt)

    process = subpr.Popen(app, cwd=working_path)
    streamdata = process.communicate()[0]
    rc = process.returncode
    print "antsRegistration = %d" % rc

# Warping from "Fixed" to "Moving" space
def apply_transform_fish(wokring_env, ref_image_path, transformation_path, labels_image_path, transformation_output, reg_prefix=None):
    working_path = wokring_env.get_working_path()
    os.environ["ANTSPATH"] = wokring_env.ANTSPATH

    args_fmt = {'refImage': ref_image_path, 'affineTransformation': transformation_path, 'labelImage': labels_image_path, 'newSegmentationImage': transformation_output}
    app3 = 'antsApplyTransforms -d 3 -r {refImage} -t {affineTransformation} -n NearestNeighbor -i {labelImage} -o {newSegmentationImage}'.format(**args_fmt)

    process = subpr.Popen(app3, cwd=working_path)
    streamdata = process.communicate()[0]
    rc = process.returncode
    print "antsApplyTransforms = %d" % rc

def find_separation_pos(stack_labels, scale_factor=1):
    objects_stats, labels = object_counter(stack_labels)
    objects_stats = objects_stats.sort(['area'], ascending=False)

    abdomen_part_z = objects_stats.loc[0, 'bb_z'] + objects_stats.loc[0, 'bb_depth']

    abdomen_label = (labels == objects_stats.loc[0, 'label']).astype(np.uint8)
    head_label = ((labels != 0) & (labels != objects_stats.loc[0, 'label'])).astype(np.uint8)

    return int(abdomen_part_z / scale_factor), abdomen_label, head_label

def split_fish_by_pos(stack_data, separation_pos, overlap=1):
    abdomen_data_part = stack_data[:separation_pos + overlap,:,:]
    head_data_part = stack_data[(separation_pos - overlap + 1):,:,:]

    return abdomen_data_part, head_data_part

def split_fish(stack_data, stack_labels):
    #get bounding boxes of abdom and head parts
    objects_stats = object_counter(stack_labels)
    objects_stats = objects_stats.sort(['area'], ascending=False)

    abdomen_part_z = objects_stats.loc[0, 'bb_z'] + objects_stats.loc[0, 'bb_depth']

    abdomen_data_part = stack_data[:abdomen_part_z,:,:]
    head_data_part = stack_data[(abdomen_part_z + 1):,:,:]

    return abdomen_data_part, head_data_part

def flip_fish(stack_data, eyes_stats, is_tail_fisrt=True):
    data_shape = stack_data.shape

    eyes_z_pos = eyes_stats['com_z'].mean()

    if (data_shape[0]/2.0 > eyes_z_pos):
        return stack_data[::-1,:,:] if is_tail_fisrt else stack_data
    else:
        return stack_data if not is_tail_fisrt else stack_data[::-1,:,:]

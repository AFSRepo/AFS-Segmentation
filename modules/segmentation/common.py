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

#ANTs - 0 , NiftyReg - 1
REG_TOOL = 1

ANTS_SCRIPTS_PATH_FMT = "~/ANKA_work/ANTs_Scripts/"

def produce_cropped_data(data_env):
    phase_name = 'extracted_input_data_path_niigz'
    phase_name_zoomed = 'zoomed_0p5_extracted_input_data_path_niigz'

    data_env.load()

    ext_volume_niigz_path = None
    zoomed_ext_volume_niigz_path = None

    ext_volume_labels_niigz_path = None
    zoomed_ext_volume_labels_niigz_path = None

    ext_volume_spine_labels_niigz_path = None
    zoomed_ext_volume_spine_labels_niigz_path = None

    bbox_ext_volume = None

    if not data_env.is_entry_exists(phase_name):
        t = Timer()

        if data_env.get_input_path():
            input_data = open_data(data_env.get_input_path())

            ext_volume, bbox_ext_volume = extract_effective_volume(input_data, bb_side_offset=10)
            print "bbox_ext_volume "
            print bbox_ext_volume
            print ext_volume.shape

            zoom_factor = 0.5

            zoomed_0p5_ext_volume = zoom(ext_volume, zoom_factor, order=3)
            zoomed_bbox_ext_volume  = _zoom_bbox(bbox_ext_volume, zoom_factor)

            data_env.set_effective_volume_bbox(bbox_ext_volume)
            data_env.set_zoomed_effective_volume_bbox(zoomed_bbox_ext_volume)

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

                ext_volume_labels = input_data_labels[bbox_ext_volume]
                zoomed_0p5_ext_volume_labels = zoom(ext_volume_labels, zoom_factor, order=0)

                ext_volume_labels_path = data_env.get_new_volume_labels_path(ext_volume_labels.shape, 'extracted')
                ext_volume_labels_niigz_path = data_env.get_new_volume_labels_niigz_path(ext_volume_labels.shape, 'extracted')

                zoomed_ext_volume_labels_path = data_env.get_new_volume_labels_path(zoomed_0p5_ext_volume_labels.shape, 'zoomed_0p5_extracted')
                zoomed_ext_volume_labels_niigz_path = data_env.get_new_volume_labels_niigz_path(zoomed_0p5_ext_volume_labels.shape, 'zoomed_0p5_extracted')

                save_as_nifti(ext_volume_labels, ext_volume_labels_niigz_path)
                save_as_nifti(zoomed_0p5_ext_volume_labels, zoomed_ext_volume_labels_niigz_path)

                ext_volume_labels.tofile(ext_volume_labels_path)
                zoomed_0p5_ext_volume_labels.tofile(zoomed_ext_volume_labels_path)

                print "Abdomen and brain labels are written and zoomed"

            if data_env.get_input_spine_labels_path():
                input_data_spine_labels = open_data(data_env.get_input_spine_labels_path())

                ext_volume_spine_labels = input_data_spine_labels[bbox_ext_volume]
                zoomed_0p5_ext_volume_spine_labels = zoom(ext_volume_spine_labels, zoom_factor, order=0)

                ext_volume_spine_labels_path = data_env.get_new_volume_labels_path(ext_volume_spine_labels.shape, 'extracted_spine')
                ext_volume_spine_labels_niigz_path = data_env.get_new_volume_labels_niigz_path(ext_volume_spine_labels.shape, 'extracted_spine')

                zoomed_ext_volume_spine_labels_path = data_env.get_new_volume_labels_path(zoomed_0p5_ext_volume_spine_labels.shape, 'zoomed_0p5_extracted_spine')
                zoomed_ext_volume_spine_labels_niigz_path = data_env.get_new_volume_labels_niigz_path(zoomed_0p5_ext_volume_spine_labels.shape, 'zoomed_0p5_extracted_spine')

                save_as_nifti(ext_volume_spine_labels, ext_volume_spine_labels_niigz_path)
                save_as_nifti(zoomed_0p5_ext_volume_spine_labels, zoomed_ext_volume_spine_labels_niigz_path)

                ext_volume_spine_labels.tofile(ext_volume_spine_labels_path)
                zoomed_0p5_ext_volume_spine_labels.tofile(zoomed_ext_volume_spine_labels_path)

                print "Spine labels are written and zoomed"

        else:
            print 'There\'s no input data'

        data_env.save()

        t.elapsed('Data cropping is finished')
    else:
        print 'Files of \'%s\' phase are already in working directory: %s' % (phase_name, data_env.get_working_path())

    return {'scaled_0p5_extracted': zoomed_ext_volume_niigz_path, 'extracted': ext_volume_niigz_path, \
            'scaled_0p5_extracted_labels': zoomed_ext_volume_labels_niigz_path, 'extracted_labels': ext_volume_labels_niigz_path, \
            'scaled_0p5_extracted_spine_labels': zoomed_ext_volume_spine_labels_niigz_path, 'extracted_spine_labels': ext_volume_spine_labels_niigz_path, \
            'bbox_extracted': bbox_ext_volume}

def crop_align_data(fixed_data_env, moving_data_env):
    use_full_size = True

    fixed_data_env.load()
    moving_data_env.load()

    # Crop the raw data
    print "--Extracting net volumes"
    fixed_data_results = produce_cropped_data(fixed_data_env)
    moving_data_results = produce_cropped_data(moving_data_env)

    fixed_data_env.save()
    moving_data_env.save()

    #generate_stats(fixed_data_env)
    #generate_stats(moving_data_env)

    print "--Pre-alignment of the unknown fish to the known one"
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

    print "--Pre-alignment of the unknown fish to the known one (Full size)"
    # Pre-alignment fish1 to fish_aligned
    ants_prefix_prealign_full = None
    ants_prealign_paths_full = None
    ants_prealign_names_full = None

    working_env_prealign_full = None
    fixed_image_path_prealign_full = None
    fixed_image_path_prealign_raw_full = None
    moving_image_path_prealign_full = None
    output_name_prealign_full = None
    warped_path_prealign_full = None
    iwarped_path_prealign_full = None

    if use_full_size:
        ants_prefix_prealign_full = 'pre_alignment_full'
        ants_prealign_paths_full = moving_data_env.get_aligned_data_paths(ants_prefix_prealign_full)
        ants_prealign_names_full = moving_data_env.get_aligned_data_paths(ants_prefix_prealign_full, produce_paths=False)

        working_env_prealign_full = moving_data_env
        fixed_image_path_prealign_full = fixed_data_env.envs['extracted_input_data_path_niigz']
        fixed_image_path_prealign_raw_full = fixed_data_env.envs['extracted_input_data_path']
        moving_image_path_prealign_full = moving_data_env.envs['extracted_input_data_path_niigz']
        output_name_prealign_full = ants_prealign_names_full['out_name']
        warped_path_prealign_full = ants_prealign_paths_full['warped']
        iwarped_path_prealign_full = ants_prealign_paths_full['iwarp']

        if not os.path.exists(warped_path_prealign_full):
            align_fish(working_env_prealign_full, fixed_image_path_prealign_full, moving_image_path_prealign_full, \
                        output_name_prealign_full, warped_path_prealign_full, iwarped_path_prealign_full, reg_prefix=ants_prefix_prealign_full, \
                        rigid_case=2)


            fixed_data_env.save()
            moving_data_env.save()
        else:
            print "Data is already prealigned (Full size)"

    print  "--Registration of the known fish to the unknown one"
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

    print  "--Registration of the known fish to the unknown one (Full size)"
    # Registration of fish_aligned to fish1
    ants_prefix_sep_full = None
    ants_separation_paths_full = None
    working_env_sep_full = None
    fixed_image_path_sep_full = None
    moving_image_path_sep_full = None
    output_name_sep_full = None
    warped_path_sep_full = None
    iwarped_path_sep_full = None

    if use_full_size:
        ants_prefix_sep_full = 'parts_separation_full'
        ants_separation_paths_full = fixed_data_env.get_aligned_data_paths(ants_prefix_sep_full)
        working_env_sep_full = fixed_data_env
        fixed_image_path_sep_full = warped_path_prealign_full
        moving_image_path_sep_full = fixed_image_path_prealign_full
        output_name_sep_full = ants_separation_paths_full['out_name']
        warped_path_sep_full = ants_separation_paths_full['warped']
        iwarped_path_sep_full = ants_separation_paths_full['iwarp']

        if not os.path.exists(warped_path_sep_full):
            align_fish(working_env_sep_full, fixed_image_path_sep_full, moving_image_path_sep_full, \
                       output_name_sep_full, warped_path_sep_full, iwarped_path_sep_full, \
                       reg_prefix=ants_prefix_sep_full, use_syn=True, syn_big_data_case=2)

            fixed_data_env.save()
            moving_data_env.save()
        else:
            print "Data is already registered for separation (Full size)"

    print "--Transforming brain and abdomen labels of the known fish to the unknown's one"
    # Transforming labels of fish_aligned to fish1
    wokring_env_tr = moving_data_env
    ref_image_path_tr = ants_prealign_paths['warped']
    transformation_path_tr = ants_separation_paths['gen_affine']
    labels_image_path_tr = fixed_data_env.envs['zoomed_0p5_extracted_input_data_labels_path_niigz']

    __, __, new_size, __ = parse_filename(fixed_image_path_prealign_raw)

    transformation_output_tr = moving_data_env.get_new_volume_niigz_path(new_size, 'zoomed_0p5_extracted_labels', bits=8)
    reg_prefix_tr = 'label_deforming'

    if not os.path.exists(transformation_output_tr):
        apply_transform_fish(wokring_env_tr, ref_image_path_tr, transformation_path_tr, labels_image_path_tr, transformation_output_tr, reg_prefix=reg_prefix_tr)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Abdomen and brain data is already transformed"

    print "--Transforming brain and abdomen labels of the known fish to the unknown's one (Full size)"
    # Transforming labels of fish_aligned to fish1
    wokring_env_tr_full = None
    ref_image_path_tr_full = None
    transformation_path_tr_full = None
    labels_image_path_tr_full = None
    new_size_full = None
    transformation_output_tr_full = None
    reg_prefix_tr_full = None

    if use_full_size:
        wokring_env_tr_full = moving_data_env
        ref_image_path_tr_full = ants_prealign_paths_full['warped']
        transformation_path_tr_full = ants_separation_paths_full['gen_affine']
        labels_image_path_tr_full = fixed_data_env.envs['extracted_input_data_labels_path_niigz']

        __, __, new_size_full, __ = parse_filename(fixed_image_path_prealign_raw_full)

        transformation_output_tr_full = moving_data_env.get_new_volume_niigz_path(new_size_full, 'extracted_labels', bits=8)
        reg_prefix_tr_full = 'label_deforming_full'

        if not os.path.exists(transformation_output_tr_full):
            apply_transform_fish(wokring_env_tr_full, ref_image_path_tr_full, transformation_path_tr_full, \
                                    labels_image_path_tr_full, transformation_output_tr_full, reg_prefix=reg_prefix_tr_full)

            fixed_data_env.save()
            moving_data_env.save()
        else:
            print "Abdomen and brain data is already transformed (Full size)"

    print "--Transforming spin labels of the known fish to the unknown's one"
    # Transforming spin labels of the known fish to the unknown's one
    wokring_env_spine_tr = moving_data_env
    ref_image_path_spine_tr = ants_prealign_paths['warped']
    transformation_path_spine_tr = ants_separation_paths['gen_affine']
    labels_image_path_spine_tr = fixed_data_env.envs['zoomed_0p5_extracted_spine_input_data_labels_path_niigz']

    __, __, new_size_spine, __ = parse_filename(fixed_image_path_prealign_raw)

    transformation_output_spine_tr = moving_data_env.get_new_volume_niigz_path(new_size_spine, 'zoomed_0p5_extracted_spine_labels', bits=8)
    reg_prefix_spine_tr = 'spine_label_deforming'

    if not os.path.exists(transformation_output_spine_tr):
        apply_transform_fish(wokring_env_spine_tr, ref_image_path_spine_tr, transformation_path_spine_tr, labels_image_path_spine_tr, transformation_output_spine_tr, reg_prefix=reg_prefix_spine_tr)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Spine data is already transformed"

    print "--Transforming spin labels of the known fish to the unknown's one (Full size)"
    # Transforming spin labels of the known fish to the unknown's one
    wokring_env_spine_tr_full = None
    ref_image_path_spine_tr_full = None
    transformation_path_spine_tr_full = None
    labels_image_path_spine_tr_full = None

    new_size_spine_full = None

    transformation_output_spine_tr_full = None
    reg_prefix_spine_tr_full = None

    if use_full_size:
        wokring_env_spine_tr_full = moving_data_env
        ref_image_path_spine_tr_full = ants_prealign_paths_full['warped']
        transformation_path_spine_tr_full = ants_separation_paths_full['gen_affine']
        labels_image_path_spine_tr_full = fixed_data_env.envs['extracted_spine_input_data_labels_path_niigz']

        __, __, new_size_spine_full, __ = parse_filename(fixed_image_path_prealign_raw_full)

        transformation_output_spine_tr_full = moving_data_env.get_new_volume_niigz_path(new_size_spine_full, 'extracted_spine_labels', bits=8)
        reg_prefix_spine_tr_full = 'spine_label_deforming_full'

        if not os.path.exists(transformation_output_spine_tr_full):
            apply_transform_fish(wokring_env_spine_tr_full, ref_image_path_spine_tr_full, transformation_path_spine_tr_full, \
                                    labels_image_path_spine_tr_full, transformation_output_spine_tr_full, reg_prefix=reg_prefix_spine_tr_full)

            fixed_data_env.save()
            moving_data_env.save()
        else:
            print "Spine data is already transformed (Full size)"

    #Separate head and tail of fixed image
    print "--Fish separation (Fixed image)..."
    abdomen_data_part_fixed_niigz_path = None
    head_data_part_fixed_niigz_path = None
    abdomen_data_part_labels_fixed_niigz_path = None
    head_data_part_labels_fixed_niigz_path = None

    if not os.path.exists(fixed_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['zoomed_0p5_head_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']):

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
    else:
        abdomen_data_part_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']
        head_data_part_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_head_input_data_path_niigz']
        abdomen_data_part_labels_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']
        head_data_part_labels_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']

    print "--Fish separation (Fixed image) (Full size)..."
    aligned_data_fixed_full = None
    aligned_data_labels_fixed_full = None
    separation_pos_fixed_full, abdomen_label_fixed_full_full, head_label_fixed_full_full = None, None, None
    abdomen_data_part_fixed_full, head_data_part_fixed_full = None, None
    abdomen_label_fixed_full = None
    head_label_fixed_full = None
    abdomen_data_part_fixed_niigz_path_full = None
    head_data_part_fixed_niigz_path_full = None
    abdomen_data_part_labels_fixed_niigz_path_full = None
    head_data_part_labels_fixed_niigz_path_full = None

    if not os.path.exists(fixed_data_env.envs['abdomen_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['head_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['abdomen_labels_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['head_labels_input_data_path_niigz']):

        if use_full_size:
            aligned_data_fixed_full = open_data(fixed_image_path_prealign_full)
            aligned_data_labels_fixed_full = open_data(labels_image_path_tr_full)

            separation_pos_fixed_full, abdomen_label_fixed_full_full, head_label_fixed_full_full = find_separation_pos(aligned_data_labels_fixed_full)
            abdomen_data_part_fixed_full, head_data_part_fixed_full = split_fish_by_pos(aligned_data_fixed_full, separation_pos_fixed_full, overlap=20)
            abdomen_label_fixed_full, _ = split_fish_by_pos(abdomen_label_fixed_full_full, separation_pos_fixed_full, overlap=20)
            _, head_label_fixed_full = split_fish_by_pos(head_label_fixed_full_full, separation_pos_fixed_full, overlap=20)

            abdomen_data_part_fixed_niigz_path_full = fixed_data_env.get_new_volume_niigz_path(abdomen_data_part_fixed_full.shape, 'abdomen')
            if not os.path.exists(abdomen_data_part_fixed_niigz_path_full):
                save_as_nifti(abdomen_data_part_fixed_full, abdomen_data_part_fixed_niigz_path_full)

            head_data_part_fixed_niigz_path_full = fixed_data_env.get_new_volume_niigz_path(head_data_part_fixed_full.shape, 'head')
            if not os.path.exists(head_data_part_fixed_niigz_path_full):
                save_as_nifti(head_data_part_fixed_full, head_data_part_fixed_niigz_path_full)

            abdomen_data_part_labels_fixed_niigz_path_full = fixed_data_env.get_new_volume_niigz_path(abdomen_label_fixed_full.shape, 'abdomen_labels')
            if not os.path.exists(abdomen_data_part_labels_fixed_niigz_path_full):
                save_as_nifti(abdomen_label_fixed_full, abdomen_data_part_labels_fixed_niigz_path_full)

            head_data_part_labels_fixed_niigz_path_full = fixed_data_env.get_new_volume_niigz_path(head_label_fixed_full.shape, 'head_labels')
            if not os.path.exists(head_data_part_labels_fixed_niigz_path_full):
                save_as_nifti(head_label_fixed_full, head_data_part_labels_fixed_niigz_path_full)

            print abdomen_data_part_labels_fixed_niigz_path_full
            print head_data_part_labels_fixed_niigz_path_full

        fixed_data_env.save()
    else:
        abdomen_data_part_fixed_niigz_path_full = fixed_data_env.envs['abdomen_input_data_path_niigz']
        head_data_part_fixed_niigz_path_full = fixed_data_env.envs['head_input_data_path_niigz']
        abdomen_data_part_labels_fixed_niigz_path_full = fixed_data_env.envs['abdomen_labels_input_data_path_niigz']
        head_data_part_labels_fixed_niigz_path_full = fixed_data_env.envs['head_labels_input_data_path_niigz']

    print "--Fish's spine separation (Fixed image)..."
    #Separate spine of tail region of fixed image from head
    spine_data_part_fixed_niigz_path = None
    spine_data_part_labels_fixed_niigz_path = None

    if not os.path.exists(fixed_data_env.envs['zoomed_0p5_spine_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['zoomed_0p5_spine_labels_input_data_path_niigz']):

        aligned_data_spine_fixed = open_data(fixed_image_path_prealign)
        aligned_data_spine_labels_fixed = open_data(labels_image_path_spine_tr)

        spine_data_part_fixed, _ = split_fish_by_pos(aligned_data_spine_fixed, separation_pos_fixed, overlap=20)
        spine_label_fixed, _ = split_fish_by_pos(aligned_data_spine_labels_fixed, separation_pos_fixed, overlap=20)

        spine_data_part_fixed_niigz_path = fixed_data_env.get_new_volume_niigz_path(spine_data_part_fixed.shape, 'zoomed_0p5_spine')
        if not os.path.exists(spine_data_part_fixed_niigz_path):
            save_as_nifti(spine_data_part_fixed, spine_data_part_fixed_niigz_path)

        spine_data_part_labels_fixed_niigz_path = fixed_data_env.get_new_volume_niigz_path(spine_label_fixed.shape, 'zoomed_0p5_spine_labels')
        if not os.path.exists(spine_data_part_labels_fixed_niigz_path):
            save_as_nifti(spine_label_fixed, spine_data_part_labels_fixed_niigz_path)

        print spine_data_part_fixed_niigz_path
        print spine_data_part_labels_fixed_niigz_path

        fixed_data_env.save()
    else:
        spine_data_part_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_spine_input_data_path_niigz']
        spine_data_part_labels_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_spine_labels_input_data_path_niigz']

    print "--Fish's spine separation (Fixed image) (Full size)..."
    #Separate spine of tail region of fixed image from head
    aligned_data_spine_fixed_full = None
    aligned_data_spine_labels_fixed_full = None
    spine_data_part_fixed_full = None
    spine_label_fixed_full = None
    spine_data_part_fixed_niigz_path_full = None
    spine_data_part_labels_fixed_niigz_path_full = None

    if not os.path.exists(fixed_data_env.envs['spine_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['spine_labels_input_data_path_niigz']):

        if use_full_size:
            aligned_data_spine_fixed_full = open_data(fixed_image_path_prealign_full)
            aligned_data_spine_labels_fixed_full = open_data(labels_image_path_spine_tr_full)

            spine_data_part_fixed_full, _ = split_fish_by_pos(aligned_data_spine_fixed_full, separation_pos_fixed_full, overlap=20)
            spine_label_fixed_full, _ = split_fish_by_pos(aligned_data_spine_labels_fixed_full, separation_pos_fixed_full, overlap=20)

            spine_data_part_fixed_niigz_path_full = fixed_data_env.get_new_volume_niigz_path(spine_data_part_fixed_full.shape, 'spine')
            if not os.path.exists(spine_data_part_fixed_niigz_path_full):
                save_as_nifti(spine_data_part_fixed_full, spine_data_part_fixed_niigz_path_full)

            spine_data_part_labels_fixed_niigz_path_full = fixed_data_env.get_new_volume_niigz_path(spine_label_fixed_full.shape, 'spine_labels')
            if not os.path.exists(spine_data_part_labels_fixed_niigz_path_full):
                save_as_nifti(spine_label_fixed_full, spine_data_part_labels_fixed_niigz_path_full)

            print spine_data_part_fixed_niigz_path_full
            print spine_data_part_labels_fixed_niigz_path_full

        fixed_data_env.save()
    else:
        spine_data_part_fixed_niigz_path_full = fixed_data_env.envs['spine_input_data_path_niigz']
        spine_data_part_labels_fixed_niigz_path_full = fixed_data_env.envs['spine_labels_input_data_path_niigz']

    #Separate head and tail of moving image
    print "--Fish separation (Moving image)..."
    abdomen_data_part_moving_niigz_path = None
    abdomen_data_part_labels_moving_niigz_path = None
    head_data_part_moving_niigz_path = None
    head_data_part_labels_moving_niigz_path = None

    if not os.path.exists(moving_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['zoomed_0p5_head_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']):

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
    else:
        abdomen_data_part_moving_niigz_path = moving_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']
        abdomen_data_part_labels_moving_niigz_path = moving_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']
        head_data_part_moving_niigz_path = moving_data_env.envs['zoomed_0p5_head_input_data_path_niigz']
        head_data_part_labels_moving_niigz_path = moving_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']

    print "--Fish separation (Moving image) (Full size)..."
    aligned_data_moving_full = None
    aligned_data_labels_moving_full = None
    separation_pos_moving_full, abdomen_label_moving_full_full, head_label_moving_full_full = None, None, None
    abdomen_data_part_moving_full, head_data_part_moving_full = None, None
    abdomen_label_moving_full = None
    head_label_moving_full = None
    abdomen_data_part_moving_niigz_path_full = None
    head_data_part_moving_niigz_path_full = None
    abdomen_data_part_labels_moving_niigz_path_full = None
    head_data_part_labels_moving_niigz_path_full = None

    if not os.path.exists(moving_data_env.envs['abdomen_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['head_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['abdomen_labels_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['head_labels_input_data_path_niigz']):

        if use_full_size:
            aligned_data_moving_full = open_data(ants_prealign_paths_full['warped'])
            aligned_data_labels_moving_full = open_data(transformation_output_tr_full)

            separation_pos_moving_full, abdomen_label_moving_full_full, head_label_moving_full_full = find_separation_pos(aligned_data_labels_moving_full)

            abdomen_data_part_moving_full, head_data_part_moving_full = split_fish_by_pos(aligned_data_moving_full, separation_pos_moving_full, overlap=20)
            abdomen_label_moving_full, _ = split_fish_by_pos(abdomen_label_moving_full_full, separation_pos_moving_full, overlap=20)
            _, head_label_moving_full = split_fish_by_pos(head_label_moving_full_full, separation_pos_moving_full, overlap=20)

            abdomen_data_part_moving_niigz_path_full = moving_data_env.get_new_volume_niigz_path(abdomen_data_part_moving_full.shape, 'abdomen')
            if not os.path.exists(abdomen_data_part_moving_niigz_path_full):
                save_as_nifti(abdomen_data_part_moving_full, abdomen_data_part_moving_niigz_path_full)

            head_data_part_moving_niigz_path_full = moving_data_env.get_new_volume_niigz_path(head_data_part_moving_full.shape, 'head')
            if not os.path.exists(head_data_part_moving_niigz_path_full):
                save_as_nifti(head_data_part_moving_full, head_data_part_moving_niigz_path_full)

            abdomen_data_part_labels_moving_niigz_path_full = moving_data_env.get_new_volume_niigz_path(abdomen_label_moving_full.shape, 'abdomen_labels')
            if not os.path.exists(abdomen_data_part_labels_moving_niigz_path_full):
                save_as_nifti(abdomen_label_moving_full, abdomen_data_part_labels_moving_niigz_path_full)

            head_data_part_labels_moving_niigz_path_full = moving_data_env.get_new_volume_niigz_path(head_label_moving_full.shape, 'head_labels')
            if not os.path.exists(head_data_part_labels_moving_niigz_path_full):
                save_as_nifti(head_label_moving_full, head_data_part_labels_moving_niigz_path_full)

            print abdomen_data_part_labels_moving_niigz_path_full
            print head_data_part_labels_moving_niigz_path_full

        moving_data_env.save()
    else:
        abdomen_data_part_moving_niigz_path_full = moving_data_env.envs['abdomen_input_data_path_niigz']
        head_data_part_moving_niigz_path_full = moving_data_env.envs['head_input_data_path_niigz']
        abdomen_data_part_labels_moving_niigz_path_full = moving_data_env.envs['abdomen_labels_input_data_path_niigz']
        head_data_part_labels_moving_niigz_path_full = moving_data_env.envs['head_labels_input_data_path_niigz']

    spine_data_part_moving_niigz_path = None
    spine_data_part_labels_moving_niigz_path = None

    if not os.path.exists(moving_data_env.envs['zoomed_0p5_spine_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['zoomed_0p5_spine_labels_input_data_path_niigz']):

        print "--Fish's spine separation (Moving image)..."
        aligned_data_spine_moving = open_data(ants_prealign_paths['warped'])
        aligned_data_spine_labels_moving = open_data(transformation_output_spine_tr)

        spine_data_part_moving, _ = split_fish_by_pos(aligned_data_spine_moving, separation_pos_moving, overlap=20)
        spine_label_moving, _ = split_fish_by_pos(aligned_data_spine_labels_moving, separation_pos_moving, overlap=20)

        spine_data_part_moving_niigz_path = moving_data_env.get_new_volume_niigz_path(spine_data_part_moving.shape, 'zoomed_0p5_spine')
        if not os.path.exists(spine_data_part_moving_niigz_path):
            save_as_nifti(spine_data_part_moving, spine_data_part_moving_niigz_path)

        spine_data_part_labels_moving_niigz_path = moving_data_env.get_new_volume_niigz_path(spine_label_moving.shape, 'zoomed_0p5_spine_labels')
        if not os.path.exists(spine_data_part_labels_moving_niigz_path):
            save_as_nifti(spine_label_moving, spine_data_part_labels_moving_niigz_path)

        print spine_data_part_moving_niigz_path
        print spine_data_part_labels_moving_niigz_path

        moving_data_env.save()
    else:
        spine_data_part_moving_niigz_path = moving_data_env.envs['zoomed_0p5_spine_input_data_path_niigz']
        spine_data_part_labels_moving_niigz_path = moving_data_env.envs['zoomed_0p5_spine_labels_input_data_path_niigz']

    print "--Fish's spine separation (Moving image) (Full size)..."
    aligned_data_spine_moving_full = None
    aligned_data_spine_labels_moving_full = None
    spine_data_part_moving_full = None
    spine_label_moving_full = None
    spine_data_part_moving_niigz_path_full = None
    spine_data_part_labels_moving_niigz_path_full = None

    if not os.path.exists(moving_data_env.envs['spine_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['spine_labels_input_data_path_niigz']):

        if use_full_size:
            aligned_data_spine_moving_full = open_data(ants_prealign_paths_full['warped'])
            aligned_data_spine_labels_moving_full = open_data(transformation_output_spine_tr_full)

            spine_data_part_moving_full, _ = split_fish_by_pos(aligned_data_spine_moving_full, separation_pos_moving_full, overlap=20)
            spine_label_moving_full, _ = split_fish_by_pos(aligned_data_spine_labels_moving_full, separation_pos_moving_full, overlap=20)

            spine_data_part_moving_niigz_path_full = moving_data_env.get_new_volume_niigz_path(spine_data_part_moving_full.shape, 'spine')
            if not os.path.exists(spine_data_part_moving_niigz_path_full):
                save_as_nifti(spine_data_part_moving_full, spine_data_part_moving_niigz_path_full)

            spine_data_part_labels_moving_niigz_path_full = moving_data_env.get_new_volume_niigz_path(spine_label_moving_full.shape, 'spine_labels')
            if not os.path.exists(spine_data_part_labels_moving_niigz_path_full):
                save_as_nifti(spine_label_moving_full, spine_data_part_labels_moving_niigz_path_full)

            print spine_data_part_moving_niigz_path_full
            print spine_data_part_labels_moving_niigz_path_full

        moving_data_env.save()
    else:
        spine_data_part_moving_niigz_path_full = moving_data_env.envs['spine_input_data_path_niigz']
        spine_data_part_labels_moving_niigz_path_full = moving_data_env.envs['spine_labels_input_data_path_niigz']

    print "--Register known fish's head to the unknown's one..."
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

    print "--Transfrom labels of known fish's head into the unknown's one..."
    # Transforming labels of head of fixed fish to the head of moving one
    wokring_env_htr = moving_data_env
    ref_image_path_htr = head_data_part_moving_niigz_path
    print ref_image_path_htr
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

    print "--Extract the unknown fish's brain using transformed head labels..."
    # Extract moving brain volume
    head_brain_label_moving = open_data(transformation_output_htr)
    head_brain_data_moving = open_data(ref_image_path_htr)
    brain_data_volume_moving, _ = extract_largest_volume_by_label(head_brain_data_moving, head_brain_label_moving, bb_side_offset=10)
    brain_data_volume_moving_niigz_path = moving_data_env.get_new_volume_niigz_path(brain_data_volume_moving.shape, 'zoomed_0p5_head_extracted_brain')

    print brain_data_volume_moving_niigz_path

    if not os.path.exists(brain_data_volume_moving_niigz_path):
        save_as_nifti(brain_data_volume_moving, brain_data_volume_moving_niigz_path)

    print "--Extract the known fish's head using labels..."
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

    print "--Register the known fish's brain to the unknown's one..."
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


    print "--Transform the known fish's brain labels into the unknown's one..."
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

    print "--Register known fish's abdomen to the unknown's one..."
    #Register fixed abdomen to moving one
    ants_prefix_abdomen_reg = 'abdomen_registration'
    ants_abdomen_reg_paths = fixed_data_env.get_aligned_data_paths(ants_prefix_abdomen_reg)
    working_env_abdomen_reg = fixed_data_env
    fixed_image_path_abdomen_reg = abdomen_data_part_moving_niigz_path
    moving_image_path_abdomen_reg = abdomen_data_part_fixed_niigz_path
    output_name_abdomen_reg = ants_abdomen_reg_paths['out_name']
    warped_path_abdomen_reg = ants_abdomen_reg_paths['warped']
    iwarped_path_abdomen_reg = ants_abdomen_reg_paths['iwarp']

    if not os.path.exists(warped_path_abdomen_reg):
        align_fish(working_env_abdomen_reg, fixed_image_path_abdomen_reg, moving_image_path_abdomen_reg, \
                   output_name_abdomen_reg, warped_path_abdomen_reg, iwarped_path_abdomen_reg, \
                   reg_prefix=ants_prefix_abdomen_reg, use_syn=True)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Abdomen of the fixed data is already registered to the abdomen of moving one"

    print "--Register known fish's abdomen to the unknown's one (Full size)..."
    #Register fixed abdomen to moving one
    ants_prefix_abdomen_reg_full = None
    ants_abdomen_reg_paths_full = None
    working_env_abdomen_reg_full = None
    fixed_image_path_abdomen_reg_full = None
    moving_image_path_abdomen_reg_full = None
    output_name_abdomen_reg_full = None
    warped_path_abdomen_reg_full = None
    iwarped_path_abdomen_reg_full = None

    if use_full_size:
        ants_prefix_abdomen_reg_full = 'abdomen_registration_full'
        ants_abdomen_reg_paths_full = fixed_data_env.get_aligned_data_paths(ants_prefix_abdomen_reg_full)
        working_env_abdomen_reg_full = fixed_data_env
        fixed_image_path_abdomen_reg_full = abdomen_data_part_moving_niigz_path_full
        moving_image_path_abdomen_reg_full = abdomen_data_part_fixed_niigz_path_full
        output_name_abdomen_reg_full = ants_abdomen_reg_paths_full['out_name']
        warped_path_abdomen_reg_full = ants_abdomen_reg_paths_full['warped']
        iwarped_path_abdomen_reg_full = ants_abdomen_reg_paths_full['iwarp']

        if not os.path.exists(warped_path_abdomen_reg_full):
            align_fish(working_env_abdomen_reg_full, fixed_image_path_abdomen_reg_full, moving_image_path_abdomen_reg_full, \
                       output_name_abdomen_reg_full, warped_path_abdomen_reg_full, iwarped_path_abdomen_reg_full, \
                       reg_prefix=ants_prefix_abdomen_reg_full, use_syn=True, syn_big_data_case=2)

            fixed_data_env.save()
            moving_data_env.save()
        else:
            print "Abdomen of the fixed data is already registered to the abdomen of moving one (Full size)"

    print "--Transfrom labels of known fish's abdomen into the unknown's one..."
    # Transforming labels of abdomen of fixed fish to the abdomen of moving one
    wokring_env_abdomen_tr = moving_data_env
    ref_image_path_abdomen_tr = abdomen_data_part_moving_niigz_path
    transformation_path_abdomen_tr = ants_abdomen_reg_paths['gen_affine']
    labels_image_path_abdomen_tr = abdomen_data_part_labels_fixed_niigz_path
    test_data_abdomen_tr = open_data(ref_image_path_abdomen_tr)
    transformation_output_abdomen_tr = moving_data_env.get_new_volume_niigz_path(test_data_abdomen_tr.shape, 'zoomed_0p5_abdomen_labels', bits=8)
    reg_prefix_abdomen_tr= 'abdomen_label_deforming'

    if not os.path.exists(transformation_output_abdomen_tr):
        apply_transform_fish(wokring_env_abdomen_tr, ref_image_path_abdomen_tr, transformation_path_abdomen_tr, \
                             labels_image_path_abdomen_tr, transformation_output_abdomen_tr, reg_prefix=reg_prefix_abdomen_tr)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Abdomen data is already transformed"

    print "--Transfrom labels of known fish's abdomen into the unknown's one (Full size)..."
    # Transforming labels of abdomen of fixed fish to the abdomen of moving one
    wokring_env_abdomen_tr_full = None
    ref_image_path_abdomen_tr_full = None
    transformation_path_abdomen_tr_full = None
    labels_image_path_abdomen_tr_full = None
    test_data_abdomen_tr_full = None
    transformation_output_abdomen_tr_full = None
    reg_prefix_abdomen_tr_full = None

    if use_full_size:
        wokring_env_abdomen_tr_full = moving_data_env
        ref_image_path_abdomen_tr_full = abdomen_data_part_moving_niigz_path_full
        transformation_path_abdomen_tr_full = ants_abdomen_reg_paths_full['gen_affine']
        labels_image_path_abdomen_tr_full = abdomen_data_part_labels_fixed_niigz_path_full
        test_data_abdomen_tr_full = open_data(ref_image_path_abdomen_tr_full)
        transformation_output_abdomen_tr_full = moving_data_env.get_new_volume_niigz_path(test_data_abdomen_tr_full.shape, 'abdomen_labels', bits=8)
        print "****transformation_output_abdomen_tr_full = %s" % transformation_output_abdomen_tr_full
        reg_prefix_abdomen_tr_full = 'abdomen_label_deforming_full'

        if not os.path.exists(transformation_output_abdomen_tr_full):
            apply_transform_fish(wokring_env_abdomen_tr_full, ref_image_path_abdomen_tr_full, transformation_path_abdomen_tr_full, \
                                 labels_image_path_abdomen_tr_full, transformation_output_abdomen_tr_full, reg_prefix=reg_prefix_abdomen_tr_full)

            fixed_data_env.save()
            moving_data_env.save()
        else:
            print "Abdomen data is already transformed (Full size)"

    print "--Transfrom labels of known fish's spine of abdomen into the unknown's one..."
    # Transforming labels of abdomen of fixed fish to the abdomen of moving one
    wokring_env_abdomen_spine_tr = moving_data_env
    ref_image_path_abdomen_spine_tr = abdomen_data_part_moving_niigz_path
    transformation_path_abdomen_spine_tr = ants_abdomen_reg_paths['gen_affine']
    labels_image_path_abdomen_spine_tr = spine_data_part_labels_fixed_niigz_path
    test_data_abdomen_spine_tr = open_data(ref_image_path_abdomen_spine_tr)
    transformation_output_abdomen_spine_tr = moving_data_env.get_new_volume_niigz_path(test_data_abdomen_spine_tr.shape, 'zoomed_0p5_abdomen_spine_labels', bits=8)
    reg_prefix_abdomen_spine_tr= 'abdomen_spine_label_deforming'

    if not os.path.exists(transformation_output_abdomen_spine_tr):
        apply_transform_fish(wokring_env_abdomen_spine_tr, ref_image_path_abdomen_spine_tr, transformation_path_abdomen_spine_tr, \
                             labels_image_path_abdomen_spine_tr, transformation_output_abdomen_spine_tr, reg_prefix=reg_prefix_abdomen_spine_tr)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Abdomen spine data is already transformed"

    print "--Transfrom labels of known fish's spine of abdomen into the unknown's one (Full size)..."
    # Transforming labels of abdomen of fixed fish to the abdomen of moving one
    wokring_env_abdomen_spine_tr_full = None
    ref_image_path_abdomen_spine_tr_full = None
    transformation_path_abdomen_spine_tr_full = None
    labels_image_path_abdomen_spine_tr_full = None
    test_data_abdomen_spine_tr_full = None
    transformation_output_abdomen_spine_tr_full = None
    reg_prefix_abdomen_spine_tr_full = None

    if use_full_size:
        wokring_env_abdomen_spine_tr_full = moving_data_env
        ref_image_path_abdomen_spine_tr_full = abdomen_data_part_moving_niigz_path_full
        transformation_path_abdomen_spine_tr_full = ants_abdomen_reg_paths_full['gen_affine']
        labels_image_path_abdomen_spine_tr_full = spine_data_part_labels_fixed_niigz_path_full
        test_data_abdomen_spine_tr_full = open_data(ref_image_path_abdomen_spine_tr_full)
        transformation_output_abdomen_spine_tr_full = moving_data_env.get_new_volume_niigz_path(test_data_abdomen_spine_tr_full.shape, 'abdomen_spine_labels', bits=8)
        reg_prefix_abdomen_spine_tr_full = 'abdomen_spine_label_deforming_full'

        if not os.path.exists(transformation_output_abdomen_spine_tr_full):
            apply_transform_fish(wokring_env_abdomen_spine_tr_full, ref_image_path_abdomen_spine_tr_full, transformation_path_abdomen_spine_tr_full, \
                                 labels_image_path_abdomen_spine_tr_full, transformation_output_abdomen_spine_tr_full, reg_prefix=reg_prefix_abdomen_spine_tr_full)

            fixed_data_env.save()
            moving_data_env.save()
        else:
            print "Abdomen spine data is already transformed (Full size)"

    print "--Extract the unknown fish's abdomen using transformed abdomen labels..."
    # Extract moving abdomen volume
    abdomen_label_moving = open_data(transformation_output_abdomen_tr)
    abdomen_data_moving = open_data(ref_image_path_abdomen_tr)
    abdomen_volume_moving, _ = extract_largest_volume_by_label(abdomen_data_moving, abdomen_label_moving, bb_side_offset=10)
    abdomen_volume_moving_niigz_path = moving_data_env.get_new_volume_niigz_path(abdomen_volume_moving.shape, 'zoomed_0p5_abdomen_extracted')

    print abdomen_volume_moving_niigz_path

    if not os.path.exists(abdomen_volume_moving_niigz_path):
        save_as_nifti(abdomen_volume_moving, abdomen_volume_moving_niigz_path)

    print "--Extract the unknown fish's abdomen using transformed abdomen labels (Full size)..."
    # Extract moving abdomen volume
    abdomen_label_moving_full = None
    abdomen_data_moving_full = None
    abdomen_volume_moving_full = None
    abdomen_volume_moving_niigz_path_full = None

    if use_full_size:
        abdomen_label_moving_full = open_data(transformation_output_abdomen_tr_full)
        abdomen_data_moving_full = open_data(ref_image_path_abdomen_tr_full)
        abdomen_volume_moving_full, _ = extract_largest_volume_by_label(abdomen_data_moving_full, abdomen_label_moving_full, bb_side_offset=10)
        abdomen_volume_moving_niigz_path_full = moving_data_env.get_new_volume_niigz_path(abdomen_volume_moving_full.shape, 'abdomen_extracted')

        print abdomen_volume_moving_niigz_path_full

        if not os.path.exists(abdomen_volume_moving_niigz_path_full):
            save_as_nifti(abdomen_volume_moving_full, abdomen_volume_moving_niigz_path_full)

    print "--Extract the known fish's abdomen using labels..."
    # Extract fixed abdomen volume
    abdomen_label_fixed = open_data(labels_image_path_abdomen_tr)
    abdomen_data_fixed = open_data(moving_image_path_abdomen_reg)
    abdomen_volume_fixed, abdomen_fixed_bbox = extract_largest_volume_by_label(abdomen_data_fixed, abdomen_label_fixed, bb_side_offset=10)
    abdomen_labels_volume_fixed = abdomen_label_fixed[abdomen_fixed_bbox]

    abdomen_volume_fixed_niigz_path = fixed_data_env.get_new_volume_niigz_path(abdomen_volume_fixed.shape, 'zoomed_0p5_abdomen_extracted')
    abdomen_labels_volume_fixed_niigz_path = fixed_data_env.get_new_volume_niigz_path(abdomen_labels_volume_fixed.shape, 'zoomed_0p5_abdomen_extracted_labels')

    print abdomen_volume_fixed_niigz_path
    print abdomen_labels_volume_fixed_niigz_path

    if not os.path.exists(abdomen_volume_fixed_niigz_path):
        save_as_nifti(abdomen_volume_fixed, abdomen_volume_fixed_niigz_path)

    if not os.path.exists(abdomen_labels_volume_fixed_niigz_path):
        save_as_nifti(abdomen_labels_volume_fixed, abdomen_labels_volume_fixed_niigz_path)

    print "--Extract the known fish's abdomen using labels (Full size)..."
    # Extract fixed abdomen volume
    abdomen_label_fixed_full = None
    abdomen_data_fixed_full = None
    abdomen_volume_fixed_full = None
    abdomen_labels_volume_fixed_full = None

    abdomen_volume_fixed_niigz_path_full = None
    abdomen_labels_volume_fixed_niigz_path_full = None

    if use_full_size:
        abdomen_label_fixed_full = open_data(labels_image_path_abdomen_tr_full)
        abdomen_data_fixed_full = open_data(moving_image_path_abdomen_reg_full)
        abdomen_volume_fixed_full, abdomen_fixed_bbox_full = extract_largest_volume_by_label(abdomen_data_fixed_full, abdomen_label_fixed_full, bb_side_offset=10)
        abdomen_labels_volume_fixed_full = abdomen_label_fixed[abdomen_fixed_bbox_full]

        abdomen_volume_fixed_niigz_path_full = fixed_data_env.get_new_volume_niigz_path(abdomen_volume_fixed_full.shape, 'abdomen_extracted')
        abdomen_labels_volume_fixed_niigz_path_full = fixed_data_env.get_new_volume_niigz_path(abdomen_labels_volume_fixed_full.shape, 'abdomen_extracted_labels')

        print abdomen_volume_fixed_niigz_path_full
        print abdomen_labels_volume_fixed_niigz_path_full

        if not os.path.exists(abdomen_volume_fixed_niigz_path_full):
            save_as_nifti(abdomen_volume_fixed_full, abdomen_volume_fixed_niigz_path_full)

        if not os.path.exists(abdomen_labels_volume_fixed_niigz_path_full):
            save_as_nifti(abdomen_labels_volume_fixed_full, abdomen_labels_volume_fixed_niigz_path_full)

    print "--Register the known fish's abdomen guts to the unknown's one..."
    # Register the fixed abdomen guts to the moving one
    ants_prefix_abdomen_guts_reg = 'abdomen_guts_registration'
    ants_abdomen_guts_reg_paths = fixed_data_env.get_aligned_data_paths(ants_prefix_abdomen_guts_reg)
    working_env_abdomen_guts_reg = fixed_data_env
    fixed_image_path_abdomen_guts_reg = abdomen_volume_moving_niigz_path
    moving_image_path_abdomen_guts_reg = abdomen_volume_fixed_niigz_path
    output_name_abdomen_guts_reg = ants_abdomen_guts_reg_paths['out_name']
    warped_path_abdomen_guts_reg = ants_abdomen_guts_reg_paths['warped']
    iwarped_path_abdomen_guts_reg = ants_abdomen_guts_reg_paths['iwarp']

    if not os.path.exists(warped_path_abdomen_guts_reg):
        align_fish(working_env_abdomen_guts_reg, fixed_image_path_abdomen_guts_reg, moving_image_path_abdomen_guts_reg, \
                   output_name_abdomen_guts_reg, warped_path_abdomen_guts_reg, iwarped_path_abdomen_guts_reg, \
                   reg_prefix=ants_prefix_abdomen_guts_reg, use_syn=True, small_volume=False)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "The guts of the abdomen of the fixed data is already registered to the guts of the abdomen of moving one"

    print "--Register the known fish's abdomen guts to the unknown's one (Full size)..."
    # Register the fixed abdomen guts to the moving one
    ants_prefix_abdomen_guts_reg_full = None
    ants_abdomen_guts_reg_paths_full = None
    working_env_abdomen_guts_reg_full = None
    fixed_image_path_abdomen_guts_reg_full = None
    moving_image_path_abdomen_guts_reg_full = None
    output_name_abdomen_guts_reg_full = None
    warped_path_abdomen_guts_reg_full = None
    iwarped_path_abdomen_guts_reg_full = None

    if use_full_size:
        ants_prefix_abdomen_guts_reg_full = 'abdomen_guts_registration_full'
        ants_abdomen_guts_reg_paths_full = fixed_data_env.get_aligned_data_paths(ants_prefix_abdomen_guts_reg_full)
        working_env_abdomen_guts_reg_full = fixed_data_env
        fixed_image_path_abdomen_guts_reg_full = abdomen_volume_moving_niigz_path_full
        moving_image_path_abdomen_guts_reg_full = abdomen_volume_fixed_niigz_path_full
        output_name_abdomen_guts_reg_full = ants_abdomen_guts_reg_paths_full['out_name']
        warped_path_abdomen_guts_reg_full = ants_abdomen_guts_reg_paths_full['warped']
        iwarped_path_abdomen_guts_reg_full = ants_abdomen_guts_reg_paths_full['iwarp']

        if not os.path.exists(warped_path_abdomen_guts_reg_full):
            align_fish(working_env_abdomen_guts_reg_full, fixed_image_path_abdomen_guts_reg_full, moving_image_path_abdomen_guts_reg_full, \
                       output_name_abdomen_guts_reg_full, warped_path_abdomen_guts_reg_full, iwarped_path_abdomen_guts_reg_full, \
                       reg_prefix=ants_prefix_abdomen_guts_reg_full, use_syn=True, small_volume=False)

            fixed_data_env.save()
            moving_data_env.save()
        else:
            print "The guts of the abdomen of the fixed data is already registered to the guts of the abdomen of moving one (Full size)"

    print "--Transform the known fish's abdomen guts labels into the unknown's one..."
    # Transforming labels of the guts of abdomen of fixed fish to the guts of the abdomen of moving one
    wokring_env_abdomen_guts_tr = moving_data_env
    ref_image_path_abdomen_guts_tr = abdomen_volume_moving_niigz_path
    transformation_path_abdomen_guts_tr = ants_abdomen_guts_reg_paths['gen_affine']
    labels_image_path_abdomen_guts_tr = abdomen_labels_volume_fixed_niigz_path
    test_data_abdomen_guts_tr = open_data(ref_image_path_abdomen_guts_tr)
    transformation_output_abdomen_guts_tr = moving_data_env.get_new_volume_niigz_path(test_data_abdomen_guts_tr.shape, 'zoomed_0p5_abdomen_extracted_labels', bits=8)
    reg_prefix_abdomen_guts_tr = 'abdomen_guts_label_deforming'

    if not os.path.exists(transformation_output_abdomen_guts_tr):
        apply_transform_fish(wokring_env_abdomen_guts_tr, ref_image_path_abdomen_guts_tr, transformation_path_abdomen_guts_tr,\
                             labels_image_path_abdomen_guts_tr, transformation_output_abdomen_guts_tr, reg_prefix=reg_prefix_abdomen_guts_tr)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "The guts of the fixed abdomen data is already transformed"

    print "--Transform the known fish's abdomen guts labels into the unknown's one (Full size)..."
    # Transforming labels of the guts of abdomen of fixed fish to the guts of the abdomen of moving one
    wokring_env_abdomen_guts_tr_full = None
    ref_image_path_abdomen_guts_tr_full = None
    transformation_path_abdomen_guts_tr_full = None
    labels_image_path_abdomen_guts_tr_full = None
    test_data_abdomen_guts_tr_full = None
    transformation_output_abdomen_guts_tr_full = None
    reg_prefix_abdomen_guts_tr_full = None

    if use_full_size:
        wokring_env_abdomen_guts_tr_full = moving_data_env
        ref_image_path_abdomen_guts_tr_full = abdomen_volume_moving_niigz_path_full
        transformation_path_abdomen_guts_tr_full = ants_abdomen_guts_reg_paths_full['gen_affine']
        labels_image_path_abdomen_guts_tr_full = abdomen_labels_volume_fixed_niigz_path_full
        test_data_abdomen_guts_tr_full = open_data(ref_image_path_abdomen_guts_tr_full)
        transformation_output_abdomen_guts_tr_full = moving_data_env.get_new_volume_niigz_path(test_data_abdomen_guts_tr_full.shape, 'abdomen_extracted_labels', bits=8)
        reg_prefix_abdomen_guts_tr_full = 'abdomen_guts_label_deforming_full'

        if not os.path.exists(transformation_output_abdomen_guts_tr_full):
            apply_transform_fish(wokring_env_abdomen_guts_tr_full, ref_image_path_abdomen_guts_tr_full, transformation_path_abdomen_guts_tr_full,\
                                 labels_image_path_abdomen_guts_tr_full, transformation_output_abdomen_guts_tr_full, reg_prefix=reg_prefix_abdomen_guts_tr_full)

            fixed_data_env.save()
            moving_data_env.save()
        else:
            print "The guts of the fixed abdomen data is already transformed"

    print "--Extract the unknown fish's abdomen spine using transformed abdomen spine labels..."
    # Extract moving abdomen spine volume

    abdomen_spine_label_moving = open_data(transformation_output_abdomen_spine_tr)
    abdomen_spine_data_moving = open_data(ref_image_path_abdomen_spine_tr)
    abdomen_spine_volume_moving, _ = extract_largest_volume_by_label(abdomen_spine_data_moving, abdomen_spine_label_moving, bb_side_offset=5)
    abdomen_spine_volume_moving_niigz_path = moving_data_env.get_new_volume_niigz_path(abdomen_spine_volume_moving.shape, 'zoomed_0p5_abdomen_spine_extracted')

    print abdomen_spine_volume_moving_niigz_path

    if not os.path.exists(abdomen_spine_volume_moving_niigz_path):
        save_as_nifti(abdomen_spine_volume_moving, abdomen_spine_volume_moving_niigz_path)

    print "--Extract the unknown fish's abdomen spine using transformed abdomen spine labels (Full size)..."
    # Extract moving abdomen spine volume
    abdomen_spine_label_moving = None
    abdomen_spine_data_moving = None
    abdomen_spine_volume_moving = None
    abdomen_spine_volume_moving_niigz_path = None

    if use_full_size:
        abdomen_spine_label_moving_full = open_data(transformation_output_abdomen_spine_tr_full)
        abdomen_spine_data_moving_full = open_data(ref_image_path_abdomen_spine_tr_full)
        abdomen_spine_volume_moving_full, _ = extract_largest_volume_by_label(abdomen_spine_data_moving_full, abdomen_spine_label_moving_full, bb_side_offset=5)
        abdomen_spine_volume_moving_niigz_path_full = moving_data_env.get_new_volume_niigz_path(abdomen_spine_volume_moving_full.shape, 'abdomen_spine_extracted')

        print abdomen_spine_volume_moving_niigz_path_full

        if not os.path.exists(abdomen_spine_volume_moving_niigz_path_full):
            save_as_nifti(abdomen_spine_volume_moving_full, abdomen_spine_volume_moving_niigz_path_full)

    print "--Extract the known fish's abdomen spine using labels..."
    # Extract fixed abdomen spine volume
    abdomen_spine_label_fixed = open_data(labels_image_path_abdomen_spine_tr)
    abdomen_spine_data_fixed = open_data(moving_image_path_abdomen_reg)
    abdomen_spine_volume_fixed, abdomen_spine_fixed_bbox = extract_largest_volume_by_label(abdomen_spine_data_fixed, abdomen_spine_label_fixed, bb_side_offset=5)
    abdomen_spine_labels_volume_fixed = abdomen_spine_label_fixed[abdomen_spine_fixed_bbox]

    abdomen_spine_volume_fixed_niigz_path = fixed_data_env.get_new_volume_niigz_path(abdomen_spine_volume_fixed.shape, 'zoomed_0p5_abdomen_spine_extracted')
    abdomen_spine_labels_volume_fixed_niigz_path = fixed_data_env.get_new_volume_niigz_path(abdomen_spine_labels_volume_fixed.shape, 'zoomed_0p5_abdomen_spine_extracted_labels')

    print abdomen_spine_volume_fixed_niigz_path
    print abdomen_spine_labels_volume_fixed_niigz_path

    if not os.path.exists(abdomen_spine_volume_fixed_niigz_path):
        save_as_nifti(abdomen_spine_volume_fixed, abdomen_spine_volume_fixed_niigz_path)

    if not os.path.exists(abdomen_spine_labels_volume_fixed_niigz_path):
        save_as_nifti(abdomen_spine_labels_volume_fixed, abdomen_spine_labels_volume_fixed_niigz_path)

    print "--Extract the known fish's abdomen spine using labels (Full size)..."
    # Extract fixed abdomen spine volume
    abdomen_spine_label_fixed_full = None
    abdomen_spine_data_fixed_full = None
    abdomen_spine_volume_fixed_full, abdomen_spine_fixed_bbox_full  = None, None
    abdomen_spine_labels_volume_fixed_full = None

    abdomen_spine_volume_fixed_niigz_path_full = None
    abdomen_spine_labels_volume_fixed_niigz_path_full = None

    if use_full_size:
        abdomen_spine_label_fixed_full = open_data(labels_image_path_abdomen_spine_tr_full)
        abdomen_spine_data_fixed_full = open_data(moving_image_path_abdomen_reg_full)
        abdomen_spine_volume_fixed_full, abdomen_spine_fixed_bbox_full = \
                    extract_largest_volume_by_label(abdomen_spine_data_fixed_full, abdomen_spine_label_fixed_full, bb_side_offset=5)
        abdomen_spine_labels_volume_fixed_full = abdomen_spine_label_fixed_full[abdomen_spine_fixed_bbox_full]

        abdomen_spine_volume_fixed_niigz_path_full = fixed_data_env.get_new_volume_niigz_path(abdomen_spine_volume_fixed_full.shape, 'abdomen_spine_extracted')
        abdomen_spine_labels_volume_fixed_niigz_path_full = fixed_data_env.get_new_volume_niigz_path(abdomen_spine_labels_volume_fixed_full.shape, 'abdomen_spine_extracted_labels')

        print abdomen_spine_volume_fixed_niigz_path_full
        print abdomen_spine_labels_volume_fixed_niigz_path_full

        if not os.path.exists(abdomen_spine_volume_fixed_niigz_path_full):
            save_as_nifti(abdomen_spine_volume_fixed_full, abdomen_spine_volume_fixed_niigz_path_full)

        if not os.path.exists(abdomen_spine_labels_volume_fixed_niigz_path_full):
            save_as_nifti(abdomen_spine_labels_volume_fixed_full, abdomen_spine_labels_volume_fixed_niigz_path_full)

    print "--Register the known fish's abdomen spine to the unknown's one..."
    # Register the fixed abdomen spine to the moving one
    ants_prefix_abdomen_spine_reg = 'abdomen_spine_registration'
    ants_abdomen_spine_reg_paths = fixed_data_env.get_aligned_data_paths(ants_prefix_abdomen_spine_reg)
    working_env_abdomen_spine_reg = fixed_data_env
    fixed_image_path_abdomen_spine_reg = abdomen_spine_volume_moving_niigz_path
    moving_image_path_abdomen_spine_reg = abdomen_spine_volume_fixed_niigz_path
    output_name_abdomen_spine_reg = ants_abdomen_spine_reg_paths['out_name']
    warped_path_abdomen_spine_reg = ants_abdomen_spine_reg_paths['warped']
    iwarped_path_abdomen_spine_reg = ants_abdomen_spine_reg_paths['iwarp']

    if not os.path.exists(warped_path_abdomen_spine_reg):
        align_fish(working_env_abdomen_spine_reg, fixed_image_path_abdomen_spine_reg, moving_image_path_abdomen_spine_reg, \
                   output_name_abdomen_spine_reg, warped_path_abdomen_spine_reg, iwarped_path_abdomen_spine_reg, \
                   reg_prefix=ants_prefix_abdomen_spine_reg, use_syn=True, small_volume=True)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "The spine of the abdomen of the fixed data is already registered to the spine of the abdomen of moving one"

    print "--Register the known fish's abdomen spine to the unknown's one (Full size)..."
    # Register the fixed abdomen spine to the moving one
    ants_prefix_abdomen_spine_reg_full = None
    ants_abdomen_spine_reg_paths_full = None
    working_env_abdomen_spine_reg_full = None
    fixed_image_path_abdomen_spine_reg_full = None
    moving_image_path_abdomen_spine_reg_full = None
    output_name_abdomen_spine_reg_full = None
    warped_path_abdomen_spine_reg_full = None
    iwarped_path_abdomen_spine_reg_full = None

    if use_full_size:
        ants_prefix_abdomen_spine_reg_full = 'abdomen_spine_registration_full'
        ants_abdomen_spine_reg_paths_full = fixed_data_env.get_aligned_data_paths(ants_prefix_abdomen_spine_reg_full)
        working_env_abdomen_spine_reg_full = fixed_data_env
        fixed_image_path_abdomen_spine_reg_full = abdomen_spine_volume_moving_niigz_path_full
        moving_image_path_abdomen_spine_reg_full = abdomen_spine_volume_fixed_niigz_path_full
        output_name_abdomen_spine_reg_full = ants_abdomen_spine_reg_paths_full['out_name']
        warped_path_abdomen_spine_reg_full = ants_abdomen_spine_reg_paths_full['warped']
        iwarped_path_abdomen_spine_reg_full = ants_abdomen_spine_reg_paths_full['iwarp']

        if not os.path.exists(warped_path_abdomen_spine_reg_full):
            align_fish(working_env_abdomen_spine_reg_full, fixed_image_path_abdomen_spine_reg_full, moving_image_path_abdomen_spine_reg_full, \
                       output_name_abdomen_spine_reg_full, warped_path_abdomen_spine_reg_full, iwarped_path_abdomen_spine_reg_full, \
                       reg_prefix=ants_prefix_abdomen_spine_reg_full, use_syn=True, small_volume=True)

            fixed_data_env.save()
            moving_data_env.save()
        else:
            print "The spine of the abdomen of the fixed data is already registered to the spine of the abdomen of moving one"

    print "--Transform the known fish's abdomen spine labels into the unknown's one..."
    # Transforming labels of the spine of abdomen of fixed fish to the spine of the abdomen of moving one
    wokring_env_abdomen_spine_tr = moving_data_env
    ref_image_path_abdomen_spine_tr = abdomen_spine_volume_moving_niigz_path
    transformation_path_abdomen_spine_tr = ants_abdomen_spine_reg_paths['gen_affine']
    labels_image_path_abdomen_spine_tr = abdomen_spine_labels_volume_fixed_niigz_path
    test_data_abdomen_spine_tr = open_data(ref_image_path_abdomen_spine_tr)
    transformation_output_abdomen_spine_tr = moving_data_env.get_new_volume_niigz_path(test_data_abdomen_spine_tr.shape, 'zoomed_0p5_abdomen_spine_extracted_labels', bits=8)
    reg_prefix_abdomen_spine_tr = 'abdomen_spine_label_deforming'

    if not os.path.exists(transformation_output_abdomen_spine_tr):
        apply_transform_fish(wokring_env_abdomen_spine_tr, ref_image_path_abdomen_spine_tr, transformation_path_abdomen_spine_tr,\
                             labels_image_path_abdomen_spine_tr, transformation_output_abdomen_spine_tr, reg_prefix=reg_prefix_abdomen_spine_tr)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "The spine of the fixed abdomen data is already transformed"

    print "--Transform the known fish's abdomen spine labels into the unknown's one (Full size)..."
    # Transforming labels of the spine of abdomen of fixed fish to the spine of the abdomen of moving one
    wokring_env_abdomen_spine_tr_full = None
    ref_image_path_abdomen_spine_tr_full = None
    transformation_path_abdomen_spine_tr_full = None
    labels_image_path_abdomen_spine_tr_full = None
    test_data_abdomen_spine_tr_full = None
    transformation_output_abdomen_spine_tr_full = None
    reg_prefix_abdomen_spine_tr_full = None

    if use_full_size:
        wokring_env_abdomen_spine_tr_full = moving_data_env
        ref_image_path_abdomen_spine_tr_full = abdomen_spine_volume_moving_niigz_path_full
        transformation_path_abdomen_spine_tr_full = ants_abdomen_spine_reg_paths_full['gen_affine']
        labels_image_path_abdomen_spine_tr_full = abdomen_spine_labels_volume_fixed_niigz_path_full
        test_data_abdomen_spine_tr_full = open_data(ref_image_path_abdomen_spine_tr_full)
        transformation_output_abdomen_spine_tr_full = moving_data_env.get_new_volume_niigz_path(test_data_abdomen_spine_tr_full.shape, 'abdomen_spine_extracted_labels', bits=8)
        reg_prefix_abdomen_spine_tr_full = 'abdomen_spine_label_deforming_full'

        if not os.path.exists(transformation_output_abdomen_spine_tr_full):
            apply_transform_fish(wokring_env_abdomen_spine_tr_full, ref_image_path_abdomen_spine_tr_full, transformation_path_abdomen_spine_tr_full,\
                                 labels_image_path_abdomen_spine_tr_full, transformation_output_abdomen_spine_tr_full, reg_prefix=reg_prefix_abdomen_spine_tr_full)

            fixed_data_env.save()
            moving_data_env.save()
        else:
            print "The spine of the fixed abdomen data is already transformed (Full size)"


def brain_segmentation_nifty(fixed_data_env, moving_data_env, use_full_size=False):

    fixed_data_env.load()
    moving_data_env.load()

    # Crop the raw data
    print "--Extracting net volumes"
    fixed_data_results = produce_cropped_data(fixed_data_env)
    moving_data_results = produce_cropped_data(moving_data_env)

    print moving_data_env.get_effective_volume_bbox()

    fixed_data_env.save()
    moving_data_env.save()

    #generate_stats(fixed_data_env)
    #generate_stats(moving_data_env)

    print "--Pre-alignment of the unknown fish to the known one"
    # Pre-alignment fish1 to fish_aligned
    nifty_prefix_prealign = 'pre_alignment'
    nifty_prealign_paths = moving_data_env.get_aligned_data_paths_nifty(nifty_prefix_prealign)

    working_env_prealign = moving_data_env
    fixed_image_path_prealign = fixed_data_env.envs['zoomed_0p5_extracted_input_data_path_niigz']
    fixed_image_path_prealign_raw = fixed_data_env.envs['zoomed_0p5_extracted_input_data_path']
    moving_image_path_prealign = moving_data_env.envs['zoomed_0p5_extracted_input_data_path_niigz']
    output_path_prealign = nifty_prealign_paths['reg_result']
    affine_matrix_path_prealign = nifty_prealign_paths['affine_mat']
    inv_affine_matrix_path_prealign = nifty_prealign_paths['inv_affine_mat']

    if not os.path.exists(output_path_prealign):
        align_fish_nifty(working_env_prealign, fixed_image_path_prealign, \
                moving_image_path_prealign, output_path_prealign, \
                affine_matrix_path_prealign, inv_affine_matrix_path_prealign, \
                num_levels=3, reg_prefix=nifty_prefix_prealign, maxit_rigid=10)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Data is already prealigned"

    print  "--Registration of the known fish to the unknown one"
    # Registration of fish_aligned to fish1
    nifty_prefix_sep = 'parts_separation'
    nifty_separation_paths = fixed_data_env.get_aligned_data_paths_nifty(nifty_prefix_sep)
    working_env_sep = fixed_data_env
    fixed_image_path_sep = output_path_prealign
    moving_image_path_sep = fixed_image_path_prealign

    output_path_sep = nifty_separation_paths['reg_result']
    affine_matrix_path_sep = nifty_separation_paths['affine_mat']
    inv_affine_matrix_path_sep = nifty_separation_paths['inv_affine_mat']
    non_rigid_trans_sep = nifty_separation_paths['non_rigid_trans']

    if not os.path.exists(non_rigid_trans_sep):
        align_fish_nifty(working_env_sep, fixed_image_path_sep, \
                moving_image_path_sep, output_path_sep, \
                affine_matrix_path_sep, inv_affine_matrix_path_sep, \
                cpp_image_path=non_rigid_trans_sep, use_bspline=True, \
                num_levels=3, reg_prefix=nifty_prefix_sep, maxit_rigid=10)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Data is already registered for separation"

    print "--Transforming brain and abdomen labels of the known fish to the unknown's one"
    # Transforming labels of fish_aligned to fish1
    wokring_env_tr = moving_data_env
    ref_image_path_tr = nifty_prealign_paths['reg_result']
    transformation_path_tr = nifty_separation_paths['non_rigid_trans']
    labels_image_path_tr = fixed_data_env.envs['zoomed_0p5_extracted_input_data_labels_path_niigz']

    __, __, new_size, __ = parse_filename(fixed_image_path_prealign_raw)

    transformation_output_tr = moving_data_env.get_new_volume_niigz_path(new_size, 'zoomed_0p5_extracted_labels', bits=8)
    reg_prefix_tr = 'label_deforming'

    if not os.path.exists(transformation_output_tr):
        apply_transform_fish_nifty(wokring_env_tr, ref_image_path_tr, \
                transformation_path_tr, labels_image_path_tr, \
                transformation_output_tr, reg_prefix=reg_prefix_tr)


        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Abdomen and brain data is already transformed"

    #Separate head and tail of fixed image
    print "--Fish separation (Fixed image)..."
    abdomen_data_part_fixed_niigz_path = None
    head_data_part_fixed_niigz_path = None
    abdomen_data_part_labels_fixed_niigz_path = None
    head_data_part_labels_fixed_niigz_path = None

    if not os.path.exists(fixed_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['zoomed_0p5_head_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']):

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
    else:
        abdomen_data_part_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']
        head_data_part_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_head_input_data_path_niigz']
        abdomen_data_part_labels_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']
        head_data_part_labels_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']

    #Separate head and tail of moving image
    print "--Fish separation (Moving image)..."
    abdomen_data_part_moving_niigz_path = None
    abdomen_data_part_labels_moving_niigz_path = None
    head_data_part_moving_niigz_path = None
    head_data_part_labels_moving_niigz_path = None

    if not os.path.exists(moving_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['zoomed_0p5_head_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']):

        aligned_data_moving = open_data(nifty_prealign_paths['reg_result'])
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
    else:
        abdomen_data_part_moving_niigz_path = moving_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']
        abdomen_data_part_labels_moving_niigz_path = moving_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']
        head_data_part_moving_niigz_path = moving_data_env.envs['zoomed_0p5_head_input_data_path_niigz']
        head_data_part_labels_moving_niigz_path = moving_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']


    print "--Register known fish's head to the unknown's one..."
    #Register fixed head to moving one
    nifty_prefix_head_reg = 'head_registration'
    nifty_head_reg_paths = fixed_data_env.get_aligned_data_paths_nifty(nifty_prefix_head_reg)
    working_env_head_reg = fixed_data_env
    fixed_image_path_head_reg = head_data_part_moving_niigz_path
    moving_image_path_head_reg = head_data_part_fixed_niigz_path


    output_path_head_reg = nifty_head_reg_paths['reg_result']
    affine_matrix_path_head_reg = nifty_head_reg_paths['affine_mat']
    inv_affine_matrix_path_head_reg = nifty_head_reg_paths['inv_affine_mat']
    non_rigid_trans_head_reg = nifty_head_reg_paths['non_rigid_trans']

    if not os.path.exists(output_path_head_reg):
        align_fish_nifty(working_env_head_reg, fixed_image_path_head_reg, \
                moving_image_path_head_reg, output_path_head_reg, \
                affine_matrix_path_head_reg, inv_affine_matrix_path_head_reg, \
                cpp_image_path=non_rigid_trans_head_reg, use_bspline=True, \
                reg_prefix=nifty_prefix_head_reg, maxit_rigid=10)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Head of the fixed data is already registered to the head of moving one"

    print "--Transfrom labels of known fish's head into the unknown's one..."
    # Transforming labels of head of fixed fish to the head of moving one
    wokring_env_htr = moving_data_env
    ref_image_path_htr = head_data_part_moving_niigz_path

    transformation_path_htr = nifty_head_reg_paths['non_rigid_trans']
    labels_image_path_htr = head_data_part_labels_fixed_niigz_path
    test_data_htr = open_data(ref_image_path_htr)
    transformation_output_htr = moving_data_env.get_new_volume_niigz_path(test_data_htr.shape, 'zoomed_0p5_head_brain_labels', bits=8)
    reg_prefix_htr = 'head_label_deforming'

    if not os.path.exists(transformation_output_htr):
        apply_transform_fish_nifty(wokring_env_htr, ref_image_path_htr, \
                transformation_path_htr, labels_image_path_htr, \
                transformation_output_htr, reg_prefix=reg_prefix_htr)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Head data is already transformed"

    print "--Extract the unknown fish's brain using transformed head labels..."
    # Extract moving brain volume
    head_brain_label_moving = open_data(transformation_output_htr)
    head_brain_data_moving = open_data(ref_image_path_htr)
    brain_data_volume_moving, brain_data_volume_moving_bbox = extract_largest_volume_by_label(head_brain_data_moving, head_brain_label_moving, bb_side_offset=10)
    brain_data_volume_moving_niigz_path = moving_data_env.get_new_volume_niigz_path(brain_data_volume_moving.shape, 'zoomed_0p5_head_extracted_brain')

    print brain_data_volume_moving_niigz_path

    if not os.path.exists(brain_data_volume_moving_niigz_path):
        save_as_nifti(brain_data_volume_moving, brain_data_volume_moving_niigz_path)

    print "--Extract the known fish's head using labels..."
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

    print "--Register the known fish's brain to the unknown's one..."
    # Register the fixed brain to the moving one
    nifty_prefix_head_brain_reg = 'head_brain_registration'
    nifty_head_brain_reg_paths = fixed_data_env.get_aligned_data_paths_nifty(nifty_prefix_head_brain_reg)

    working_env_head_brain_reg = fixed_data_env
    fixed_image_path_head_brain_reg = brain_data_volume_moving_niigz_path
    moving_image_path_head_brain_reg = brain_data_volume_fixed_niigz_path

    output_path_brain_reg = nifty_head_brain_reg_paths['reg_result']
    affine_matrix_path_brain_reg = nifty_head_brain_reg_paths['affine_mat']
    inv_affine_matrix_path_brain_reg = nifty_head_brain_reg_paths['inv_affine_mat']
    non_rigid_trans_brain_reg = nifty_head_brain_reg_paths['non_rigid_trans']
    grid_spacing_brain_reg = -4

    if not os.path.exists(output_path_brain_reg):
        align_fish_nifty(working_env_head_brain_reg, fixed_image_path_head_brain_reg, \
                moving_image_path_head_brain_reg, output_path_brain_reg, \
                affine_matrix_path_brain_reg, inv_affine_matrix_path_brain_reg, \
                cpp_image_path=non_rigid_trans_brain_reg, use_bspline=True, \
                reg_prefix=nifty_prefix_head_brain_reg, \
                grid_spacing=grid_spacing_brain_reg, maxit_rigid=30)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "The brain of the head of the fixed data is already registered to the brain of the head of moving one"


    print "--Transform the known fish's brain labels into the unknown's one..."
    # Transforming labels of the brain of head of fixed fish to the brain of the head of moving one
    wokring_env_brain_tr = moving_data_env
    ref_image_path_brain_tr = brain_data_volume_moving_niigz_path

    transformation_path_brain_tr = nifty_head_brain_reg_paths['non_rigid_trans']
    labels_image_path_brain_tr = brain_data_labels_volume_fixed_niigz_path
    test_data_brain_tr = open_data(ref_image_path_brain_tr)
    transformation_output_brain_tr = moving_data_env.get_new_volume_niigz_path(test_data_brain_tr.shape, 'zoomed_0p5_head_extracted_brain_labels', bits=8)
    reg_prefix_brain_tr = 'head_brain_label_deforming'

    if not os.path.exists(transformation_output_brain_tr):
        apply_transform_fish_nifty(wokring_env_brain_tr, ref_image_path_brain_tr, transformation_path_brain_tr,\
                             labels_image_path_brain_tr, transformation_output_brain_tr, reg_prefix=reg_prefix_brain_tr)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "The brain of the fixed head data is already transformed"

    print "--Complete the unknown fish's brain labels to full volume..."
    test_data_complete_vol_brain_moving = open_data(output_path_prealign)
    complete_vol_unknown_brain_labels_niigz_path = moving_data_env.get_new_volume_niigz_path(test_data_complete_vol_brain_moving.shape, 'zoomed_0p5_complete_volume_brain_labels', bits=8)

    if not os.path.exists(complete_vol_unknown_brain_labels_niigz_path):
        complete_vol_unknown_brain_labels = complete_brain_to_full_volume(abdomen_data_part_labels_moving_niigz_path, \
                                                                         head_data_part_labels_moving_niigz_path, \
                                                                         transformation_output_brain_tr, \
                                                                         brain_data_volume_moving_bbox, \
                                                                         separation_overlap=20);
        save_as_nifti(complete_vol_unknown_brain_labels, complete_vol_unknown_brain_labels_niigz_path)
    else:
        print "The brain labels of the moving data (unknown fish) is already transformed."

    print "--Inverse transfrom the completed unknown fish's brain labels to the initial alignment..."
    wokring_env_brain_labels_inverse_tr = moving_data_env
    ref_image_space_path_brain_labels_inverse_tr = moving_image_path_prealign

    transformation_path_brain_labels_inverse_tr = nifty_prealign_paths['inv_affine_mat']
    labels_image_path_brain_inverse_tr = complete_vol_unknown_brain_labels_niigz_path
    test_data_brain_inverse_tr = open_data(ref_image_space_path_brain_labels_inverse_tr)
    transformation_output_brain_labels_inverse_tr = moving_data_env.get_new_volume_niigz_path(test_data_brain_inverse_tr.shape, 'zoomed_0p5_complete_volume_brain_labels_initial_alignment', bits=8)
    reg_prefix_brain_labels_inverse_tr = 'complete_volume_brain_labels_deforming_to_initial_alignment'

    if not os.path.exists(transformation_output_brain_labels_inverse_tr):
        apply_inverse_transform_fish_nifty(wokring_env_brain_labels_inverse_tr, \
                                         ref_image_space_path_brain_labels_inverse_tr, \
                                         transformation_path_brain_labels_inverse_tr, \
                                         labels_image_path_brain_inverse_tr, \
                                         transformation_output_brain_labels_inverse_tr, \
                                         reg_prefix=reg_prefix_brain_labels_inverse_tr,
                                         interp_order=3)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "The completed unknown fish's brain labels is already transformed to the initial alignment."

    print "--Upscale the initial aligned completed unknown fish's brain labels to the input volume size..."
    scaled_initally_aligned_data_brain_labels_path = transformation_output_brain_labels_inverse_tr
    target_orignal_data_fish_path = moving_data_env.get_input_path()
    upscaled_initially_aligned_complete_vol_unknown_brain_labels_niigz_path = moving_data_env.get_new_volume_niigz_path(test_data_complete_vol_brain_moving.shape, 'complete_volume_brain_labels_initial_alignment', bits=8)
    zoomed_volume_bbox = moving_data_env.get_zoomed_effective_volume_bbox()

    print scaled_initally_aligned_data_brain_labels_path
    print target_orignal_data_fish_path
    print upscaled_initially_aligned_complete_vol_unknown_brain_labels_niigz_path
    print moving_data_env.get_effective_volume_bbox()

    if not os.path.exists(upscaled_initially_aligned_complete_vol_unknown_brain_labels_niigz_path):
        upscaled_initally_aligned_data_brain_labels = scale_to_size(target_orignal_data_fish_path, \
                                                                    scaled_initally_aligned_data_brain_labels_path, \
                                                                    zoomed_volume_bbox, \
                                                                    scale=2.0, \
                                                                    order=0)
        save_as_nifti(upscaled_initally_aligned_data_brain_labels, \
                      upscaled_initially_aligned_complete_vol_unknown_brain_labels_niigz_path)
    else:
        print "The initially aligned completed unknown fish's brain labels is already upscaled to the input volume size."


def brain_segmentation(fixed_data_env, moving_data_env, use_full_size=False):

    fixed_data_env.load()
    moving_data_env.load()

    # Crop the raw data
    print "--Extracting net volumes"
    fixed_data_results = produce_cropped_data(fixed_data_env)
    moving_data_results = produce_cropped_data(moving_data_env)

    print moving_data_env.get_effective_volume_bbox()

    fixed_data_env.save()
    moving_data_env.save()

    #generate_stats(fixed_data_env)
    #generate_stats(moving_data_env)

    print "--Pre-alignment of the unknown fish to the known one"
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

    print "--Pre-alignment of the unknown fish to the known one (Full size)"
    # Pre-alignment fish1 to fish_aligned
    ants_prefix_prealign_full = None
    ants_prealign_paths_full = None
    ants_prealign_names_full = None

    working_env_prealign_full = None
    fixed_image_path_prealign_full = None
    fixed_image_path_prealign_raw_full = None
    moving_image_path_prealign_full = None
    output_name_prealign_full = None
    warped_path_prealign_full = None
    iwarped_path_prealign_full = None

    if use_full_size:
        ants_prefix_prealign_full = 'pre_alignment_full'
        ants_prealign_paths_full = moving_data_env.get_aligned_data_paths(ants_prefix_prealign_full)
        ants_prealign_names_full = moving_data_env.get_aligned_data_paths(ants_prefix_prealign_full, produce_paths=False)

        working_env_prealign_full = moving_data_env
        fixed_image_path_prealign_full = fixed_data_env.envs['extracted_input_data_path_niigz']
        fixed_image_path_prealign_raw_full = fixed_data_env.envs['extracted_input_data_path']
        moving_image_path_prealign_full = moving_data_env.envs['extracted_input_data_path_niigz']
        output_name_prealign_full = ants_prealign_names_full['out_name']
        warped_path_prealign_full = ants_prealign_paths_full['warped']
        iwarped_path_prealign_full = ants_prealign_paths_full['iwarp']

        if not os.path.exists(warped_path_prealign_full):
            align_fish(working_env_prealign_full, fixed_image_path_prealign_full, moving_image_path_prealign_full, \
                        output_name_prealign_full, warped_path_prealign_full, iwarped_path_prealign_full, reg_prefix=ants_prefix_prealign_full, \
                        rigid_case=2)


            fixed_data_env.save()
            moving_data_env.save()
        else:
            print "Data is already prealigned (Full size)"

    print  "--Registration of the known fish to the unknown one"
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

    print  "--Registration of the known fish to the unknown one (Full size)"
    # Registration of fish_aligned to fish1
    ants_prefix_sep_full = None
    ants_separation_paths_full = None
    working_env_sep_full = None
    fixed_image_path_sep_full = None
    moving_image_path_sep_full = None
    output_name_sep_full = None
    warped_path_sep_full = None
    iwarped_path_sep_full = None

    if use_full_size:
        ants_prefix_sep_full = 'parts_separation_full'
        ants_separation_paths_full = fixed_data_env.get_aligned_data_paths(ants_prefix_sep_full)
        working_env_sep_full = fixed_data_env
        fixed_image_path_sep_full = warped_path_prealign_full
        moving_image_path_sep_full = fixed_image_path_prealign_full
        output_name_sep_full = ants_separation_paths_full['out_name']
        warped_path_sep_full = ants_separation_paths_full['warped']
        iwarped_path_sep_full = ants_separation_paths_full['iwarp']

        if not os.path.exists(warped_path_sep_full):
            align_fish(working_env_sep_full, fixed_image_path_sep_full, moving_image_path_sep_full, \
                       output_name_sep_full, warped_path_sep_full, iwarped_path_sep_full, \
                       reg_prefix=ants_prefix_sep_full, use_syn=True, syn_big_data_case=2)

            fixed_data_env.save()
            moving_data_env.save()
        else:
            print "Data is already registered for separation (Full size)"

    print "--Transforming brain and abdomen labels of the known fish to the unknown's one"
    # Transforming labels of fish_aligned to fish1
    wokring_env_tr = moving_data_env
    ref_image_path_tr = ants_prealign_paths['warped']
    transformation_path_tr = ants_separation_paths['gen_affine']
    labels_image_path_tr = fixed_data_env.envs['zoomed_0p5_extracted_input_data_labels_path_niigz']

    __, __, new_size, __ = parse_filename(fixed_image_path_prealign_raw)

    transformation_output_tr = moving_data_env.get_new_volume_niigz_path(new_size, 'zoomed_0p5_extracted_labels', bits=8)
    reg_prefix_tr = 'label_deforming'

    if not os.path.exists(transformation_output_tr):
        apply_transform_fish(wokring_env_tr, ref_image_path_tr, transformation_path_tr, labels_image_path_tr, transformation_output_tr, reg_prefix=reg_prefix_tr)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Abdomen and brain data is already transformed"

    print "--Transforming brain and abdomen labels of the known fish to the unknown's one (Full size)"
    # Transforming labels of fish_aligned to fish1
    wokring_env_tr_full = None
    ref_image_path_tr_full = None
    transformation_path_tr_full = None
    labels_image_path_tr_full = None
    new_size_full = None
    transformation_output_tr_full = None
    reg_prefix_tr_full = None

    if use_full_size:
        wokring_env_tr_full = moving_data_env
        ref_image_path_tr_full = ants_prealign_paths_full['warped']
        transformation_path_tr_full = ants_separation_paths_full['gen_affine']
        labels_image_path_tr_full = fixed_data_env.envs['extracted_input_data_labels_path_niigz']

        __, __, new_size_full, __ = parse_filename(fixed_image_path_prealign_raw_full)

        transformation_output_tr_full = moving_data_env.get_new_volume_niigz_path(new_size_full, 'extracted_labels', bits=8)
        reg_prefix_tr_full = 'label_deforming_full'

        if not os.path.exists(transformation_output_tr_full):
            apply_transform_fish(wokring_env_tr_full, ref_image_path_tr_full, transformation_path_tr_full, \
                                    labels_image_path_tr_full, transformation_output_tr_full, reg_prefix=reg_prefix_tr_full)

            fixed_data_env.save()
            moving_data_env.save()
        else:
            print "Abdomen and brain data is already transformed (Full size)"

    print "--Transforming spin labels of the known fish to the unknown's one"
    # Transforming spin labels of the known fish to the unknown's one
    wokring_env_spine_tr = moving_data_env
    ref_image_path_spine_tr = ants_prealign_paths['warped']
    transformation_path_spine_tr = ants_separation_paths['gen_affine']
    labels_image_path_spine_tr = fixed_data_env.envs['zoomed_0p5_extracted_spine_input_data_labels_path_niigz']

    __, __, new_size_spine, __ = parse_filename(fixed_image_path_prealign_raw)

    transformation_output_spine_tr = moving_data_env.get_new_volume_niigz_path(new_size_spine, 'zoomed_0p5_extracted_spine_labels', bits=8)
    reg_prefix_spine_tr = 'spine_label_deforming'

    if not os.path.exists(transformation_output_spine_tr):
        apply_transform_fish(wokring_env_spine_tr, ref_image_path_spine_tr, transformation_path_spine_tr, labels_image_path_spine_tr, transformation_output_spine_tr, reg_prefix=reg_prefix_spine_tr)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Spine data is already transformed"

    print "--Transforming spin labels of the known fish to the unknown's one (Full size)"
    # Transforming spin labels of the known fish to the unknown's one
    wokring_env_spine_tr_full = None
    ref_image_path_spine_tr_full = None
    transformation_path_spine_tr_full = None
    labels_image_path_spine_tr_full = None

    new_size_spine_full = None

    transformation_output_spine_tr_full = None
    reg_prefix_spine_tr_full = None

    if use_full_size:
        wokring_env_spine_tr_full = moving_data_env
        ref_image_path_spine_tr_full = ants_prealign_paths_full['warped']
        transformation_path_spine_tr_full = ants_separation_paths_full['gen_affine']
        labels_image_path_spine_tr_full = fixed_data_env.envs['extracted_spine_input_data_labels_path_niigz']

        __, __, new_size_spine_full, __ = parse_filename(fixed_image_path_prealign_raw_full)

        transformation_output_spine_tr_full = moving_data_env.get_new_volume_niigz_path(new_size_spine_full, 'extracted_spine_labels', bits=8)
        reg_prefix_spine_tr_full = 'spine_label_deforming_full'

        if not os.path.exists(transformation_output_spine_tr_full):
            apply_transform_fish(wokring_env_spine_tr_full, ref_image_path_spine_tr_full, transformation_path_spine_tr_full, \
                                    labels_image_path_spine_tr_full, transformation_output_spine_tr_full, reg_prefix=reg_prefix_spine_tr_full)

            fixed_data_env.save()
            moving_data_env.save()
        else:
            print "Spine data is already transformed (Full size)"

    #Separate head and tail of fixed image
    print "--Fish separation (Fixed image)..."
    abdomen_data_part_fixed_niigz_path = None
    head_data_part_fixed_niigz_path = None
    abdomen_data_part_labels_fixed_niigz_path = None
    head_data_part_labels_fixed_niigz_path = None

    if not os.path.exists(fixed_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['zoomed_0p5_head_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']):

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
    else:
        abdomen_data_part_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']
        head_data_part_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_head_input_data_path_niigz']
        abdomen_data_part_labels_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']
        head_data_part_labels_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']

    print "--Fish separation (Fixed image) (Full size)..."
    aligned_data_fixed_full = None
    aligned_data_labels_fixed_full = None
    separation_pos_fixed_full, abdomen_label_fixed_full_full, head_label_fixed_full_full = None, None, None
    abdomen_data_part_fixed_full, head_data_part_fixed_full = None, None
    abdomen_label_fixed_full = None
    head_label_fixed_full = None
    abdomen_data_part_fixed_niigz_path_full = None
    head_data_part_fixed_niigz_path_full = None
    abdomen_data_part_labels_fixed_niigz_path_full = None
    head_data_part_labels_fixed_niigz_path_full = None

    if not os.path.exists(fixed_data_env.envs['abdomen_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['head_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['abdomen_labels_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['head_labels_input_data_path_niigz']):

        if use_full_size:
            aligned_data_fixed_full = open_data(fixed_image_path_prealign_full)
            aligned_data_labels_fixed_full = open_data(labels_image_path_tr_full)

            separation_pos_fixed_full, abdomen_label_fixed_full_full, head_label_fixed_full_full = find_separation_pos(aligned_data_labels_fixed_full)
            abdomen_data_part_fixed_full, head_data_part_fixed_full = split_fish_by_pos(aligned_data_fixed_full, separation_pos_fixed_full, overlap=20)
            abdomen_label_fixed_full, _ = split_fish_by_pos(abdomen_label_fixed_full_full, separation_pos_fixed_full, overlap=20)
            _, head_label_fixed_full = split_fish_by_pos(head_label_fixed_full_full, separation_pos_fixed_full, overlap=20)

            abdomen_data_part_fixed_niigz_path_full = fixed_data_env.get_new_volume_niigz_path(abdomen_data_part_fixed_full.shape, 'abdomen')
            if not os.path.exists(abdomen_data_part_fixed_niigz_path_full):
                save_as_nifti(abdomen_data_part_fixed_full, abdomen_data_part_fixed_niigz_path_full)

            head_data_part_fixed_niigz_path_full = fixed_data_env.get_new_volume_niigz_path(head_data_part_fixed_full.shape, 'head')
            if not os.path.exists(head_data_part_fixed_niigz_path_full):
                save_as_nifti(head_data_part_fixed_full, head_data_part_fixed_niigz_path_full)

            abdomen_data_part_labels_fixed_niigz_path_full = fixed_data_env.get_new_volume_niigz_path(abdomen_label_fixed_full.shape, 'abdomen_labels')
            if not os.path.exists(abdomen_data_part_labels_fixed_niigz_path_full):
                save_as_nifti(abdomen_label_fixed_full, abdomen_data_part_labels_fixed_niigz_path_full)

            head_data_part_labels_fixed_niigz_path_full = fixed_data_env.get_new_volume_niigz_path(head_label_fixed_full.shape, 'head_labels')
            if not os.path.exists(head_data_part_labels_fixed_niigz_path_full):
                save_as_nifti(head_label_fixed_full, head_data_part_labels_fixed_niigz_path_full)

            print abdomen_data_part_labels_fixed_niigz_path_full
            print head_data_part_labels_fixed_niigz_path_full

        fixed_data_env.save()
    else:
        abdomen_data_part_fixed_niigz_path_full = fixed_data_env.envs['abdomen_input_data_path_niigz']
        head_data_part_fixed_niigz_path_full = fixed_data_env.envs['head_input_data_path_niigz']
        abdomen_data_part_labels_fixed_niigz_path_full = fixed_data_env.envs['abdomen_labels_input_data_path_niigz']
        head_data_part_labels_fixed_niigz_path_full = fixed_data_env.envs['head_labels_input_data_path_niigz']

    print "--Fish's spine separation (Fixed image)..."
    #Separate spine of tail region of fixed image from head
    spine_data_part_fixed_niigz_path = None
    spine_data_part_labels_fixed_niigz_path = None

    if not os.path.exists(fixed_data_env.envs['zoomed_0p5_spine_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['zoomed_0p5_spine_labels_input_data_path_niigz']):

        aligned_data_spine_fixed = open_data(fixed_image_path_prealign)
        aligned_data_spine_labels_fixed = open_data(labels_image_path_spine_tr)

        spine_data_part_fixed, _ = split_fish_by_pos(aligned_data_spine_fixed, separation_pos_fixed, overlap=20)
        spine_label_fixed, _ = split_fish_by_pos(aligned_data_spine_labels_fixed, separation_pos_fixed, overlap=20)

        spine_data_part_fixed_niigz_path = fixed_data_env.get_new_volume_niigz_path(spine_data_part_fixed.shape, 'zoomed_0p5_spine')
        if not os.path.exists(spine_data_part_fixed_niigz_path):
            save_as_nifti(spine_data_part_fixed, spine_data_part_fixed_niigz_path)

        spine_data_part_labels_fixed_niigz_path = fixed_data_env.get_new_volume_niigz_path(spine_label_fixed.shape, 'zoomed_0p5_spine_labels')
        if not os.path.exists(spine_data_part_labels_fixed_niigz_path):
            save_as_nifti(spine_label_fixed, spine_data_part_labels_fixed_niigz_path)

        print spine_data_part_fixed_niigz_path
        print spine_data_part_labels_fixed_niigz_path

        fixed_data_env.save()
    else:
        spine_data_part_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_spine_input_data_path_niigz']
        spine_data_part_labels_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_spine_labels_input_data_path_niigz']

    print "--Fish's spine separation (Fixed image) (Full size)..."
    #Separate spine of tail region of fixed image from head
    aligned_data_spine_fixed_full = None
    aligned_data_spine_labels_fixed_full = None
    spine_data_part_fixed_full = None
    spine_label_fixed_full = None
    spine_data_part_fixed_niigz_path_full = None
    spine_data_part_labels_fixed_niigz_path_full = None

    if not os.path.exists(fixed_data_env.envs['spine_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['spine_labels_input_data_path_niigz']):

        if use_full_size:
            aligned_data_spine_fixed_full = open_data(fixed_image_path_prealign_full)
            aligned_data_spine_labels_fixed_full = open_data(labels_image_path_spine_tr_full)

            spine_data_part_fixed_full, _ = split_fish_by_pos(aligned_data_spine_fixed_full, separation_pos_fixed_full, overlap=20)
            spine_label_fixed_full, _ = split_fish_by_pos(aligned_data_spine_labels_fixed_full, separation_pos_fixed_full, overlap=20)

            spine_data_part_fixed_niigz_path_full = fixed_data_env.get_new_volume_niigz_path(spine_data_part_fixed_full.shape, 'spine')
            if not os.path.exists(spine_data_part_fixed_niigz_path_full):
                save_as_nifti(spine_data_part_fixed_full, spine_data_part_fixed_niigz_path_full)

            spine_data_part_labels_fixed_niigz_path_full = fixed_data_env.get_new_volume_niigz_path(spine_label_fixed_full.shape, 'spine_labels')
            if not os.path.exists(spine_data_part_labels_fixed_niigz_path_full):
                save_as_nifti(spine_label_fixed_full, spine_data_part_labels_fixed_niigz_path_full)

            print spine_data_part_fixed_niigz_path_full
            print spine_data_part_labels_fixed_niigz_path_full

        fixed_data_env.save()
    else:
        spine_data_part_fixed_niigz_path_full = fixed_data_env.envs['spine_input_data_path_niigz']
        spine_data_part_labels_fixed_niigz_path_full = fixed_data_env.envs['spine_labels_input_data_path_niigz']

    #Separate head and tail of moving image
    print "--Fish separation (Moving image)..."
    abdomen_data_part_moving_niigz_path = None
    abdomen_data_part_labels_moving_niigz_path = None
    head_data_part_moving_niigz_path = None
    head_data_part_labels_moving_niigz_path = None

    if not os.path.exists(moving_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['zoomed_0p5_head_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']):

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
    else:
        abdomen_data_part_moving_niigz_path = moving_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']
        abdomen_data_part_labels_moving_niigz_path = moving_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']
        head_data_part_moving_niigz_path = moving_data_env.envs['zoomed_0p5_head_input_data_path_niigz']
        head_data_part_labels_moving_niigz_path = moving_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']

    print "--Fish separation (Moving image) (Full size)..."
    aligned_data_moving_full = None
    aligned_data_labels_moving_full = None
    separation_pos_moving_full, abdomen_label_moving_full_full, head_label_moving_full_full = None, None, None
    abdomen_data_part_moving_full, head_data_part_moving_full = None, None
    abdomen_label_moving_full = None
    head_label_moving_full = None
    abdomen_data_part_moving_niigz_path_full = None
    head_data_part_moving_niigz_path_full = None
    abdomen_data_part_labels_moving_niigz_path_full = None
    head_data_part_labels_moving_niigz_path_full = None

    if not os.path.exists(moving_data_env.envs['abdomen_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['head_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['abdomen_labels_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['head_labels_input_data_path_niigz']):

        if use_full_size:
            aligned_data_moving_full = open_data(ants_prealign_paths_full['warped'])
            aligned_data_labels_moving_full = open_data(transformation_output_tr_full)

            separation_pos_moving_full, abdomen_label_moving_full_full, head_label_moving_full_full = find_separation_pos(aligned_data_labels_moving_full)

            abdomen_data_part_moving_full, head_data_part_moving_full = split_fish_by_pos(aligned_data_moving_full, separation_pos_moving_full, overlap=20)
            abdomen_label_moving_full, _ = split_fish_by_pos(abdomen_label_moving_full_full, separation_pos_moving_full, overlap=20)
            _, head_label_moving_full = split_fish_by_pos(head_label_moving_full_full, separation_pos_moving_full, overlap=20)

            abdomen_data_part_moving_niigz_path_full = moving_data_env.get_new_volume_niigz_path(abdomen_data_part_moving_full.shape, 'abdomen')
            if not os.path.exists(abdomen_data_part_moving_niigz_path_full):
                save_as_nifti(abdomen_data_part_moving_full, abdomen_data_part_moving_niigz_path_full)

            head_data_part_moving_niigz_path_full = moving_data_env.get_new_volume_niigz_path(head_data_part_moving_full.shape, 'head')
            if not os.path.exists(head_data_part_moving_niigz_path_full):
                save_as_nifti(head_data_part_moving_full, head_data_part_moving_niigz_path_full)

            abdomen_data_part_labels_moving_niigz_path_full = moving_data_env.get_new_volume_niigz_path(abdomen_label_moving_full.shape, 'abdomen_labels')
            if not os.path.exists(abdomen_data_part_labels_moving_niigz_path_full):
                save_as_nifti(abdomen_label_moving_full, abdomen_data_part_labels_moving_niigz_path_full)

            head_data_part_labels_moving_niigz_path_full = moving_data_env.get_new_volume_niigz_path(head_label_moving_full.shape, 'head_labels')
            if not os.path.exists(head_data_part_labels_moving_niigz_path_full):
                save_as_nifti(head_label_moving_full, head_data_part_labels_moving_niigz_path_full)

            print abdomen_data_part_labels_moving_niigz_path_full
            print head_data_part_labels_moving_niigz_path_full

        moving_data_env.save()
    else:
        abdomen_data_part_moving_niigz_path_full = moving_data_env.envs['abdomen_input_data_path_niigz']
        head_data_part_moving_niigz_path_full = moving_data_env.envs['head_input_data_path_niigz']
        abdomen_data_part_labels_moving_niigz_path_full = moving_data_env.envs['abdomen_labels_input_data_path_niigz']
        head_data_part_labels_moving_niigz_path_full = moving_data_env.envs['head_labels_input_data_path_niigz']

    spine_data_part_moving_niigz_path = None
    spine_data_part_labels_moving_niigz_path = None

    if not os.path.exists(moving_data_env.envs['zoomed_0p5_spine_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['zoomed_0p5_spine_labels_input_data_path_niigz']):

        print "--Fish's spine separation (Moving image)..."
        aligned_data_spine_moving = open_data(ants_prealign_paths['warped'])
        aligned_data_spine_labels_moving = open_data(transformation_output_spine_tr)

        spine_data_part_moving, _ = split_fish_by_pos(aligned_data_spine_moving, separation_pos_moving, overlap=20)
        spine_label_moving, _ = split_fish_by_pos(aligned_data_spine_labels_moving, separation_pos_moving, overlap=20)

        spine_data_part_moving_niigz_path = moving_data_env.get_new_volume_niigz_path(spine_data_part_moving.shape, 'zoomed_0p5_spine')
        if not os.path.exists(spine_data_part_moving_niigz_path):
            save_as_nifti(spine_data_part_moving, spine_data_part_moving_niigz_path)

        spine_data_part_labels_moving_niigz_path = moving_data_env.get_new_volume_niigz_path(spine_label_moving.shape, 'zoomed_0p5_spine_labels')
        if not os.path.exists(spine_data_part_labels_moving_niigz_path):
            save_as_nifti(spine_label_moving, spine_data_part_labels_moving_niigz_path)

        print spine_data_part_moving_niigz_path
        print spine_data_part_labels_moving_niigz_path

        moving_data_env.save()
    else:
        spine_data_part_moving_niigz_path = moving_data_env.envs['zoomed_0p5_spine_input_data_path_niigz']
        spine_data_part_labels_moving_niigz_path = moving_data_env.envs['zoomed_0p5_spine_labels_input_data_path_niigz']

    print "--Fish's spine separation (Moving image) (Full size)..."
    aligned_data_spine_moving_full = None
    aligned_data_spine_labels_moving_full = None
    spine_data_part_moving_full = None
    spine_label_moving_full = None
    spine_data_part_moving_niigz_path_full = None
    spine_data_part_labels_moving_niigz_path_full = None

    if not os.path.exists(moving_data_env.envs['spine_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['spine_labels_input_data_path_niigz']):

        if use_full_size:
            aligned_data_spine_moving_full = open_data(ants_prealign_paths_full['warped'])
            aligned_data_spine_labels_moving_full = open_data(transformation_output_spine_tr_full)

            spine_data_part_moving_full, _ = split_fish_by_pos(aligned_data_spine_moving_full, separation_pos_moving_full, overlap=20)
            spine_label_moving_full, _ = split_fish_by_pos(aligned_data_spine_labels_moving_full, separation_pos_moving_full, overlap=20)

            spine_data_part_moving_niigz_path_full = moving_data_env.get_new_volume_niigz_path(spine_data_part_moving_full.shape, 'spine')
            if not os.path.exists(spine_data_part_moving_niigz_path_full):
                save_as_nifti(spine_data_part_moving_full, spine_data_part_moving_niigz_path_full)

            spine_data_part_labels_moving_niigz_path_full = moving_data_env.get_new_volume_niigz_path(spine_label_moving_full.shape, 'spine_labels')
            if not os.path.exists(spine_data_part_labels_moving_niigz_path_full):
                save_as_nifti(spine_label_moving_full, spine_data_part_labels_moving_niigz_path_full)

            print spine_data_part_moving_niigz_path_full
            print spine_data_part_labels_moving_niigz_path_full

        moving_data_env.save()
    else:
        spine_data_part_moving_niigz_path_full = moving_data_env.envs['spine_input_data_path_niigz']
        spine_data_part_labels_moving_niigz_path_full = moving_data_env.envs['spine_labels_input_data_path_niigz']

    print "--Register known fish's head to the unknown's one..."
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

    print "--Transfrom labels of known fish's head into the unknown's one..."
    # Transforming labels of head of fixed fish to the head of moving one
    wokring_env_htr = moving_data_env
    ref_image_path_htr = head_data_part_moving_niigz_path
    print ref_image_path_htr
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

    print "--Extract the unknown fish's brain using transformed head labels..."
    # Extract moving brain volume
    head_brain_label_moving = open_data(transformation_output_htr)
    head_brain_data_moving = open_data(ref_image_path_htr)
    brain_data_volume_moving, brain_data_volume_moving_bbox = extract_largest_volume_by_label(head_brain_data_moving, head_brain_label_moving, bb_side_offset=10)
    brain_data_volume_moving_niigz_path = moving_data_env.get_new_volume_niigz_path(brain_data_volume_moving.shape, 'zoomed_0p5_head_extracted_brain')

    print brain_data_volume_moving_niigz_path

    if not os.path.exists(brain_data_volume_moving_niigz_path):
        save_as_nifti(brain_data_volume_moving, brain_data_volume_moving_niigz_path)

    print "--Extract the known fish's head using labels..."
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

    print "--Register the known fish's brain to the unknown's one..."
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


    print "--Transform the known fish's brain labels into the unknown's one..."
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

    print "--Complete the unknown fish's brain labels to full volume..."
    test_data_complete_vol_brain_moving = open_data(warped_path_prealign)
    complete_vol_unknown_brain_labels_niigz_path = moving_data_env.get_new_volume_niigz_path(test_data_complete_vol_brain_moving.shape, 'zoomed_0p5_complete_volume_brain_labels', bits=8)

    if not os.path.exists(complete_vol_unknown_brain_labels_niigz_path):
        complete_vol_unknown_brain_labels = complete_brain_to_full_volume(abdomen_data_part_labels_moving_niigz_path, \
                                                                         head_data_part_labels_moving_niigz_path, \
                                                                         transformation_output_brain_tr, \
                                                                         brain_data_volume_moving_bbox, \
                                                                         separation_overlap=20);
        save_as_nifti(complete_vol_unknown_brain_labels, complete_vol_unknown_brain_labels_niigz_path)
    else:
        print "The brain labels of the moving data (unknown fish) is already transformed."

    print "--Inverse transfrom the completed unknown fish's brain labels to the initial alignment..."
    wokring_env_brain_labels_inverse_tr = moving_data_env
    ref_image_space_path_brain_labels_inverse_tr = moving_image_path_prealign
    transformation_path_brain_labels_inverse_tr = ants_prealign_paths['gen_affine']
    labels_image_path_brain_inverse_tr = complete_vol_unknown_brain_labels_niigz_path
    test_data_brain_inverse_tr = open_data(ref_image_space_path_brain_labels_inverse_tr)
    transformation_output_brain_labels_inverse_tr = moving_data_env.get_new_volume_niigz_path(test_data_brain_inverse_tr.shape, 'zoomed_0p5_complete_volume_brain_labels_initial_alignment', bits=8)
    reg_prefix_brain_labels_inverse_tr = 'complete_volume_brain_labels_deforming_to_initial_alignment'

    if not os.path.exists(transformation_output_brain_labels_inverse_tr):
        apply_inverse_transform_fish(wokring_env_brain_labels_inverse_tr, \
                                     ref_image_space_path_brain_labels_inverse_tr, \
                                     transformation_path_brain_labels_inverse_tr, \
                                     labels_image_path_brain_inverse_tr, \
                                     transformation_output_brain_labels_inverse_tr, \
                                     reg_prefix=reg_prefix_brain_labels_inverse_tr)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "The completed unknown fish's brain labels is already transformed to the initial alignment."

    print "--Upscale the initial aligned completed unknown fish's brain labels to the input volume size..."
    scaled_initally_aligned_data_brain_labels_path = transformation_output_brain_labels_inverse_tr
    target_orignal_data_fish_path = moving_data_env.get_input_path()
    upscaled_initially_aligned_complete_vol_unknown_brain_labels_niigz_path = moving_data_env.get_new_volume_niigz_path(test_data_complete_vol_brain_moving.shape, 'complete_volume_brain_labels_initial_alignment', bits=8)
    zoomed_effective_bbox = moving_data_env.get_zoomed_effective_volume_bbox()

    if not os.path.exists(upscaled_initially_aligned_complete_vol_unknown_brain_labels_niigz_path):
        upscaled_initally_aligned_data_brain_labels = scale_to_size(target_orignal_data_fish_path, \
                                                                    scaled_initally_aligned_data_brain_labels_path, \
                                                                    zoomed_effective_bbox, \
                                                                    scale=2.0, \
                                                                    order=0)
        save_as_nifti(upscaled_initally_aligned_data_brain_labels, \
                      upscaled_initially_aligned_complete_vol_unknown_brain_labels_niigz_path)
    else:
        print "The initially aligned completed unknown fish's brain labels is already upscaled to the input volume size."

def brain_segmentation_ants(fixed_data_env, moving_data_env):

    fixed_data_env.load()
    moving_data_env.load()

    # Crop the raw data
    print "--Extracting net volumes"
    fixed_data_results = produce_cropped_data(fixed_data_env)
    moving_data_results = produce_cropped_data(moving_data_env)

    fixed_data_env.save()
    moving_data_env.save()

    #generate_stats(fixed_data_env)
    #generate_stats(moving_data_env)

    print "--Pre-alignment of the unknown fish to the known one"
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
        align_fish_simple_ants(working_env_prealign, fixed_image_path_prealign,\
                               moving_image_path_prealign, output_name_prealign, \
                               warped_path_prealign, iwarped_path_prealign, \
                               reg_prefix=ants_prefix_prealign, use_syn=False, \
                               use_full_iters=False)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Data is already prealigned"

    print  "--Registration of the known fish to the unknown one"
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
        align_fish_simple_ants(working_env_sep, fixed_image_path_sep, moving_image_path_sep, \
                               output_name_sep, warped_path_sep, iwarped_path_sep, \
                               reg_prefix=ants_prefix_sep, use_syn=True, use_full_iters=False)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Data is already registered for separation"

    print "--Transforming brain and abdomen labels of the known fish to the unknown's one"
    # Transforming labels of fish_aligned to fish1
    wokring_env_tr = moving_data_env
    ref_image_path_tr = ants_prealign_paths['warped']
    affine_transformation_path_tr = ants_separation_paths['gen_affine']
    def_transformation_path_tr = ants_separation_paths['warp']
    labels_image_path_tr = fixed_data_env.envs['zoomed_0p5_extracted_input_data_labels_path_niigz']

    __, __, new_size, __ = parse_filename(fixed_image_path_prealign_raw)

    transformation_output_tr = moving_data_env.get_new_volume_niigz_path(new_size, 'zoomed_0p5_extracted_labels', bits=8)
    reg_prefix_tr = 'label_deforming'

    if not os.path.exists(transformation_output_tr):
        apply_transform_fish(wokring_env_tr, ref_image_path_tr, \
                             affine_transformation_path_tr, \
                             labels_image_path_tr, transformation_output_tr, \
                             def_transformation_path=def_transformation_path_tr, \
                             reg_prefix=reg_prefix_tr)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Abdomen and brain data is already transformed"

    #Separate head and tail of fixed image
    print "--Fish separation (Fixed image)..."
    abdomen_data_part_fixed_niigz_path = None
    head_data_part_fixed_niigz_path = None
    abdomen_data_part_labels_fixed_niigz_path = None
    head_data_part_labels_fixed_niigz_path = None

    if not os.path.exists(fixed_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['zoomed_0p5_head_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']) or \
       not os.path.exists(fixed_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']):

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
    else:
        abdomen_data_part_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']
        head_data_part_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_head_input_data_path_niigz']
        abdomen_data_part_labels_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']
        head_data_part_labels_fixed_niigz_path = fixed_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']

    #Separate head and tail of moving image
    print "--Fish separation (Moving image)..."
    abdomen_data_part_moving_niigz_path = None
    abdomen_data_part_labels_moving_niigz_path = None
    head_data_part_moving_niigz_path = None
    head_data_part_labels_moving_niigz_path = None

    if not os.path.exists(moving_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['zoomed_0p5_head_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']) or \
       not os.path.exists(moving_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']):

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
    else:
        abdomen_data_part_moving_niigz_path = moving_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']
        abdomen_data_part_labels_moving_niigz_path = moving_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']
        head_data_part_moving_niigz_path = moving_data_env.envs['zoomed_0p5_head_input_data_path_niigz']
        head_data_part_labels_moving_niigz_path = moving_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']

    print "--Register known fish's head to the unknown's one..."
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
        align_fish_simple_ants(working_env_head_reg, fixed_image_path_head_reg, \
                               moving_image_path_head_reg, output_name_head_reg, \
                               warped_path_head_reg, iwarped_path_head_reg, \
                               reg_prefix=ants_prefix_head_reg, use_syn=True, \
                               use_full_iters=False)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Head of the fixed data is already registered to the head of moving one"

    print "--Transfrom labels of known fish's head into the unknown's one..."
    # Transforming labels of head of fixed fish to the head of moving one
    wokring_env_htr = moving_data_env
    ref_image_path_htr = head_data_part_moving_niigz_path
    print ref_image_path_htr
    transformation_path_htr = ants_head_reg_paths['gen_affine']
    def_transformation_path_htr = ants_head_reg_paths['warp']
    labels_image_path_htr = head_data_part_labels_fixed_niigz_path
    test_data_htr = open_data(ref_image_path_htr)
    transformation_output_htr = moving_data_env.get_new_volume_niigz_path(test_data_htr.shape, 'zoomed_0p5_head_brain_labels', bits=8)
    reg_prefix_htr = 'head_label_deforming'

    if not os.path.exists(transformation_output_htr):
        apply_transform_fish(wokring_env_tr, ref_image_path_htr, \
                             transformation_path_htr, labels_image_path_htr, \
                             transformation_output_htr, \
                             def_transformation_path=def_transformation_path_htr, \
                             reg_prefix=reg_prefix_htr)
        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "Head data is already transformed"

    print "--Extract the unknown fish's brain using transformed head labels..."
    # Extract moving brain volume
    head_brain_label_moving = open_data(transformation_output_htr)
    head_brain_data_moving = open_data(ref_image_path_htr)
    brain_data_volume_moving, brain_data_volume_moving_bbox = extract_largest_volume_by_label(head_brain_data_moving, head_brain_label_moving, bb_side_offset=10)
    brain_data_volume_moving_niigz_path = moving_data_env.get_new_volume_niigz_path(brain_data_volume_moving.shape, 'zoomed_0p5_head_extracted_brain')

    print brain_data_volume_moving_niigz_path

    if not os.path.exists(brain_data_volume_moving_niigz_path):
        save_as_nifti(brain_data_volume_moving, brain_data_volume_moving_niigz_path)

    print "--Extract the known fish's head using labels..."
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

    print "--Register the known fish's brain to the unknown's one..."
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
        align_fish_simple_ants(working_env_head_brain_reg, \
                               fixed_image_path_head_brain_reg, \
                               moving_image_path_head_brain_reg, \
                               output_name_head_brain_reg, \
                               warped_path_head_brain_reg, \
                               iwarped_path_head_brain_reg, \
                               reg_prefix=ants_prefix_head_brain_reg, \
                               use_syn=True, \
                               use_full_iters=True)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "The brain of the head of the fixed data is already registered to the brain of the head of moving one"


    print "--Transform the known fish's brain labels into the unknown's one..."
    # Transforming labels of the brain of head of fixed fish to the brain of the head of moving one
    wokring_env_brain_tr = moving_data_env
    ref_image_path_brain_tr = brain_data_volume_moving_niigz_path
    transformation_path_brain_tr = ants_head_brain_reg_paths['gen_affine']
    labels_image_path_brain_tr = brain_data_labels_volume_fixed_niigz_path
    test_data_brain_tr = open_data(ref_image_path_brain_tr)
    transformation_output_brain_tr = moving_data_env.get_new_volume_niigz_path(test_data_brain_tr.shape, 'zoomed_0p5_head_extracted_brain_labels', bits=8)
    reg_prefix_brain_tr = 'head_brain_label_deforming'
    def_transformation_path_brain_tr = ants_head_brain_reg_paths['warp']

    if not os.path.exists(transformation_output_brain_tr):
        apply_transform_fish(wokring_env_brain_tr, ref_image_path_brain_tr, \
                             transformation_path_brain_tr, labels_image_path_brain_tr, \
                             transformation_output_brain_tr, \
                             def_transformation_path=def_transformation_path_brain_tr, \
                             reg_prefix=reg_prefix_brain_tr)


        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "The brain of the fixed head data is already transformed"

    print "--Complete the unknown fish's brain labels to full volume..."
    test_data_complete_vol_brain_moving = open_data(warped_path_prealign)
    complete_vol_unknown_brain_labels_niigz_path = moving_data_env.get_new_volume_niigz_path(test_data_complete_vol_brain_moving.shape, 'zoomed_0p5_complete_volume_brain_labels', bits=8)

    if not os.path.exists(complete_vol_unknown_brain_labels_niigz_path):
        complete_vol_unknown_brain_labels = complete_brain_to_full_volume(abdomen_data_part_labels_moving_niigz_path, \
                                                                         head_data_part_labels_moving_niigz_path, \
                                                                         transformation_output_brain_tr, \
                                                                         brain_data_volume_moving_bbox, \
                                                                         separation_overlap=20);
        save_as_nifti(complete_vol_unknown_brain_labels, complete_vol_unknown_brain_labels_niigz_path)
    else:
        print "The brain labels of the moving data (unknown fish) is already transformed."

    print "--Inverse transfrom the completed unknown fish's brain labels to the initial alignment..."
    wokring_env_brain_labels_inverse_tr = moving_data_env
    ref_image_space_path_brain_labels_inverse_tr = moving_image_path_prealign
    affine_transformation_path_brain_labels_inverse_tr = ants_prealign_paths['gen_affine']
    labels_image_path_brain_inverse_tr = complete_vol_unknown_brain_labels_niigz_path
    test_data_brain_inverse_tr = open_data(ref_image_space_path_brain_labels_inverse_tr)
    transformation_output_brain_labels_inverse_tr = moving_data_env.get_new_volume_niigz_path(test_data_brain_inverse_tr.shape, 'zoomed_0p5_complete_volume_brain_labels_initial_alignment', bits=8)
    reg_prefix_brain_labels_inverse_tr = 'complete_volume_brain_labels_deforming_to_initial_alignment'

    if not os.path.exists(transformation_output_brain_labels_inverse_tr):
        apply_inverse_transform_fish(wokring_env_brain_labels_inverse_tr, \
                                     ref_image_space_path_brain_labels_inverse_tr, \
                                     affine_transformation_path_brain_labels_inverse_tr, \
                                     labels_image_path_brain_inverse_tr, \
                                     transformation_output_brain_labels_inverse_tr, \
                                     reg_prefix=reg_prefix_brain_labels_inverse_tr)

        fixed_data_env.save()
        moving_data_env.save()
    else:
        print "The completed unknown fish's brain labels is already transformed to the initial alignment."

    print "--Upscale the initial aligned completed unknown fish's brain labels to the input volume size..."
    scaled_initally_aligned_data_brain_labels_path = transformation_output_brain_labels_inverse_tr
    target_orignal_data_fish_path = moving_data_env.get_input_path()
    upscaled_initially_aligned_complete_vol_unknown_brain_labels_niigz_path = moving_data_env.get_new_volume_niigz_path(test_data_complete_vol_brain_moving.shape, 'complete_volume_brain_labels_initial_alignment', bits=8)
    zoomed_volume_bbox = moving_data_env.get_zoomed_effective_volume_bbox()

    if not os.path.exists(upscaled_initially_aligned_complete_vol_unknown_brain_labels_niigz_path):
        upscaled_initally_aligned_data_brain_labels = scale_to_size(target_orignal_data_fish_path, \
                                                                    scaled_initally_aligned_data_brain_labels_path, \
                                                                    zoomed_volume_bbox, \
                                                                    scale=2.0, \
                                                                    order=0)
        save_as_nifti(upscaled_initally_aligned_data_brain_labels, \
                      upscaled_initially_aligned_complete_vol_unknown_brain_labels_niigz_path)
    else:
        print "The initially aligned completed unknown fish's brain labels is already upscaled to the input volume size."


def scale_to_size_old(target_data_path, extracted_scaled_data_path, extracted_data_bbox, scale=2.0, order=0):
    target_data = open_data(target_data_path)
    extracted_scaled_data = open_data(extracted_scaled_data_path)

    #complete to full volume
    scaled_target_size = tuple(v/scale for v in target_data.shape)
    print "target_data.shape = %s" % str(target_data.shape)
    print "scaled_target_size = %s" % str(scaled_target_size)
    print "extracted_scaled_data.shape = %s" % str(extracted_scaled_data.shape)

    complete_scaled_data = complete_volume_to_full_volume(scaled_target_size, extracted_scaled_data_path, extracted_data_bbox)

    #calculate scales
    axes_scales = tuple(round(a/b, int(-np.floor(np.log10(a/b)))) \
                                for a,b in zip(target_data.shape,complete_scaled_data.shape))

    scaled_dims = tuple(round(x*s) for x,s in zip(complete_scaled_data.shape,axes_scales))

    print target_data.shape
    print scaled_dims

    if target_data.shape != scaled_dims:
        print "ERROR: Incorrect scale factors, the scaled data don't fit the target one!"
        sys.exit(0)

    rescaled_data = zoom(complete_scaled_data, axes_scales, order=order)

    return rescaled_data

def scale_to_size(target_data_path, extracted_scaled_data_path, \
                  extracted_scaled_data_bbox, scale=2.0, order=0):
    target_data = open_data(target_data_path)
    extracted_scaled_data = open_data(extracted_scaled_data_path)

    rescaled_extracted_data = zoom(extracted_scaled_data, scale, order=order)
    rescaled_extracted_bbox = _zoom_bbox(extracted_scaled_data_bbox, scale)

    complete_scaled_data = complete_volume_to_full_volume(target_data.shape, rescaled_extracted_data, rescaled_extracted_bbox)

    return complete_scaled_data
def complete_brain_to_full_volume(abdomed_part_path, head_part_path, extracted_brain_volume_path, extracted_brain_volume_bbox, separation_overlap=1):
    abdomed_part = open_data(abdomed_part_path)
    head_part = open_data(head_part_path)
    extracted_brain_volume = open_data(extracted_brain_volume_path)

    mask_volume_abdomed = np.zeros_like(abdomed_part, dtype=np.uint8)

    mask_volume_head = np.zeros_like(head_part, dtype=np.uint8)
    mask_volume_head[extracted_brain_volume_bbox] = extracted_brain_volume

    # separation_overlap*2 - 1 because two parts overlap at some point and share it
    mask_full_volume = np.concatenate((mask_volume_abdomed[:-(separation_overlap*2 - 1),:,:], mask_volume_head), axis=0)

    return mask_full_volume

def complete_volume_to_full_volume(target_data_shape, extracted_data, extracted_data_bbox):
    completed_data = np.zeros(shape=target_data_shape, dtype=extracted_data.dtype)
    completed_data[extracted_data_bbox] = extracted_data

    return completed_data

def extract_largest_volume_by_label(stack_data, stack_labels, bb_side_offset=0):
    stack_stats, _ = object_counter(stack_labels)
    print "INPUT extract_largest_volume_by_label = %s" % str(stack_data.shape)
    largest_volume_region, bbox = extract_largest_area_data(stack_data, stack_stats, bb_side_offset)
    print "BBOX extract_largest_volume_by_label = %s" % str(bbox)

    return largest_volume_region, bbox

def extract_effective_volume(stack_data, eyes_stats=None, bb_side_offset=0):
    timer_total = Timer()

    timer = Timer()
    print 'Binarizing...'
    binarized_stack, bbox, eyes_stats = binarizator(stack_data)
    print bbox
    print binarized_stack.shape
    binarized_stack.tofile('/home/rshkarin/ANKA_work/AFS-playground/Segmentation/fish200/fish_binary_%s.raw' \
            % str(binarized_stack.shape))
    timer.elapsed('Binarizing')

    timer = Timer()
    print 'Object counting...'
    binary_stack_stats, _ = object_counter(binarized_stack)
    timer.elapsed('Object counting')

    timer = Timer()
    print 'Big volume extraction...'
    largest_volume_region, largest_volume_region_bbox = extract_largest_area_data(stack_data, binary_stack_stats, bb_side_offset)
    print largest_volume_region_bbox
    timer.elapsed('Big volume extraction')

    timer_total.elapsed('Total')

    return largest_volume_region, largest_volume_region_bbox

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

def align_fish(working_env, fixed_image_path, moving_image_path, output_name, \
        warped_path, iwarped_path, reg_prefix=None, use_syn=False, \
        small_volume=False, syn_big_data_case=1, rigid_case=1):

    working_path = working_env.get_working_path()
    os.environ["ANTSPATH"] = working_env.ANTSPATH

    args_fmt = {'out_name': output_name, 'warped_path': warped_path, 'iwarped_path': iwarped_path, 'fixedImagePath': fixed_image_path, 'movingImagePath': moving_image_path}

    app = None

    if not use_syn:
        if rigid_case == 1:
            app = 'antsRegistration --dimensionality 3 --float 1 --output [{out_name},{warped_path},{iwarped_path}] --interpolation BSpline --use-histogram-matching 0 --initial-moving-transform [{fixedImagePath},{movingImagePath},1] --transform Rigid[0.01] --metric MI[{fixedImagePath},{movingImagePath},1,32,Regular,0.25] --convergence [1000x500x250x10,1e-8,10] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox'.format(**args_fmt)
        elif rigid_case == 2:
            app = 'antsRegistration --dimensionality 3 --float 1 --output [{out_name},{warped_path},{iwarped_path}] --interpolation BSpline --use-histogram-matching 0 --initial-moving-transform [{fixedImagePath},{movingImagePath},1] --transform Rigid[0.01] --metric MI[{fixedImagePath},{movingImagePath},1,32,Regular,0.25] --convergence [1000x500x250x10,1e-6,10] --shrink-factors 6x4x2x1 --smoothing-sigmas 4x3x2x1vox'.format(**args_fmt)
    else:
        if not small_volume:
            if syn_big_data_case == 1:
                app = 'antsRegistration --dimensionality 3 --float 1 --output [{out_name},{warped_path},{iwarped_path}] --interpolation BSpline --use-histogram-matching 0 --initial-moving-transform [{fixedImagePath},{movingImagePath},1] --transform Rigid[0.1] --metric MI[{fixedImagePath},{movingImagePath},1,32,Regular,0.25] --convergence [1000x500x250x100,1e-6,10] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox --transform Affine[0.1] --metric MI[{fixedImagePath},{movingImagePath},1,32,Regular,0.25] --convergence [1000x500x250x100,1e-6,10] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox --transform SyN[0.1,3,0] --metric CC[{fixedImagePath},{movingImagePath},1,10] --metric MI[{fixedImagePath},{movingImagePath},1,32,Regular,0.25] --convergence [200x200x200x20x0,1e-6,10] --shrink-factors 8x6x4x2x1 --smoothing-sigmas 4x3x2x1x0vox'.format(**args_fmt)
            elif syn_big_data_case == 2:
                app = 'antsRegistration --dimensionality 3 --float 1 --output [{out_name},{warped_path},{iwarped_path}] --interpolation BSpline --use-histogram-matching 0 --initial-moving-transform [{fixedImagePath},{movingImagePath},1] --transform Rigid[0.1] --metric MI[{fixedImagePath},{movingImagePath},1,32,Regular,0.25] --convergence [1000x500x250x100,1e-6,10] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox --transform Affine[0.1] --metric MI[{fixedImagePath},{movingImagePath},1,32,Regular,0.25] --convergence [1000x500x250x100,1e-6,10] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox --transform SyN[0.25,3,1] --metric CC[{fixedImagePath},{movingImagePath},1,4] --convergence [400x200x50x10x0,1e-6,10] --shrink-factors 5x4x3x2x1 --smoothing-sigmas 4x3x2x1x0vox'.format(**args_fmt)
        else:
            app = 'antsRegistration --dimensionality 3 --float 1 --output [{out_name},{warped_path},{iwarped_path}] --interpolation BSpline --use histogram-matching 0 --initial-moving-transform [{fixedImagePath},{movingImagePath},1] --transform Rigid[0.1] --metric MI[{fixedImagePath},{movingImagePath},1,32,Regular,0.25] --convergence [1000x500x250x100,1e-6,10] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox --transform Affine[0.1] --metric MI[{fixedImagePath},{movingImagePath},1,32,Regular,0.25] --convergence [1000x500x250x100,1e-6,10] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox --transform SyN[0.1,3,0] --metric CC[{fixedImagePath},{movingImagePath},1,10] --metric MI[{fixedImagePath},{movingImagePath},1,32,Regular,0.25] --convergence [200x150x20x5,1e-6,10] --shrink-factors 6x4x2x1 --smoothing-sigmas 3x2x1x0vox'.format(**args_fmt)

    process = subpr.Popen(app, cwd=working_path)
    streamdata = process.communicate()[0]
    rc = process.returncode
    print "antsRegistration = %d" % rc

def align_fish_simple_ants(working_env, fixed_image_path, moving_image_path, output_name, \
        warped_path, iwarped_path, reg_prefix=None, use_syn=True, \
        use_full_iters=True, num_threads=24, \
        path_ants_scripts_fmt=ANTS_SCRIPTS_PATH_FMT):

    working_path = working_env.get_working_path()
    os.environ["ANTSPATH"] = working_env.ANTSPATH

    if not working_env.ANTSPATH:
        sys.exit(1)

    shell_app = "bash "

    args_fmt = {'out_name': output_name, 'warped_path': warped_path, \
                'iwarped_path': iwarped_path, \
                'fixedImagePath': fixed_image_path, \
                'movingImagePath': moving_image_path, \
                'num_threads': num_threads}

    app = None

    if not use_syn:
        app = 'antsRegistrationSyNMid.sh -d 3 -f {fixedImagePath} ' \
                '-m {movingImagePath} -o {out_name} -n {num_threads} -t r -p f'.format(**args_fmt)
    else:
        if use_full_iters:
            app = 'antsRegistrationSyN.sh -d 3 -f {fixedImagePath} ' \
                    '-m {movingImagePath} -o {out_name} -n {num_threads} -t b -p f'.format(**args_fmt)
        else:
            app = 'antsRegistrationSyNMid.sh -d 3 -f {fixedImagePath} ' \
                    '-m {movingImagePath} -o {out_name} -n {num_threads} -t s -p f'.format(**args_fmt)

    app = path_ants_scripts_fmt + app
    print app
    process = subpr.Popen(app, cwd=working_path, shell=True)
    streamdata = process.communicate()[0]
    rc = process.returncode
    print "antsRegistration (Simple) = %d" % rc



def align_fish_nifty(working_env, fixed_image_path, moving_image_path, \
        output_image_path, aff_path, inv_aff_path, cpp_image_path=None, \
        reg_prefix=None, use_bspline=False, rigid_case=1, num_levels=3, \
        grid_spacing=-5, omp=24, maxit_rigid=100, maxit_bspline=1000):

    working_path = working_env.get_working_path()

    args_fmt_rigid, args_fmt_bspline = None, None

    args_fmt_rigid = {'fixed_image': fixed_image_path, 'moving_image': \
            moving_image_path, 'affine_matrix': aff_path, 'output_image': \
            output_image_path, 'num_ln': num_levels, 'omp': omp, \
            'maxit_rigid': maxit_rigid}

    if use_bspline:
        args_fmt_bspline = {'fixed_image': fixed_image_path, 'moving_image': \
            moving_image_path, 'affine_matrix': aff_path, 'output_image': \
            output_image_path, 'cpp_image': cpp_image_path, \
            'num_ln': num_levels, 'grid_spacing': grid_spacing, \
            'maxit_bspline': maxit_bspline}


    run_rigid, run_bspline  = None, None

    if not use_bspline:
        if rigid_case == 1:
            run_rigid = 'reg_aladin -ref {fixed_image} -flo {moving_image} \
            -res {output_image} -aff {affine_matrix} -ln {num_ln} -omp {omp} \
            -maxit {maxit_rigid}'.format(**args_fmt_rigid)
    else:
        run_rigid = 'reg_aladin -ref {fixed_image} -flo {moving_image} -res \
            {output_image} -aff {affine_matrix} -ln {num_ln} -omp {omp} \
            -maxit {maxit_rigid}'.format(**args_fmt_rigid)
        run_bspline = 'reg_f3d -ref {fixed_image} -flo {moving_image} -res \
            {output_image} -aff {affine_matrix} -cpp {cpp_image} \
            -ln {num_ln} -sx {grid_spacing} -maxit {maxit_bspline} -gpu'.format(**args_fmt_bspline)

    process = subpr.Popen(run_rigid, cwd=working_path, shell=True)
    streamdata = process.communicate()[0]
    rc = process.returncode
    print "Nifty Reg Rigid = %d" % rc

    if use_bspline:
        print run_bspline

        process = subpr.Popen(run_bspline, cwd=working_path, shell=True)
        streamdata = process.communicate()[0]
        rc = process.returncode
        print "Nifty Reg BSpline = %d" % rc

    args_fmt_inv_aff = {'aff': aff_path, 'inv_aff': inv_aff_path }
    run_inv_transform = 'reg_transform -invAff {aff} {inv_aff}'.format(**args_fmt_inv_aff)

    process = subpr.Popen(run_inv_transform, cwd=working_path, shell=True)
    streamdata = process.communicate()[0]
    rc = process.returncode
    print "Nifty Reg Inv Aff Transform = %d" % rc

# Warping from "Fixed" to "Moving" space
def apply_transform_fish(wokring_env, ref_image_path, affine_transformation_path, \
                         labels_image_path, transformation_output, \
                         def_transformation_path=None, reg_prefix=None):
    working_path = wokring_env.get_working_path()
    os.environ["ANTSPATH"] = wokring_env.ANTSPATH

    args_fmt = {'refImage': ref_image_path, \
                'affineTransformation': affine_transformation_path, \
                'labelImage': labels_image_path, \
                'newSegmentationImage': transformation_output}

    cmd_template = 'antsApplyTransforms -d 3 -r {refImage} ' \
            '-t {affineTransformation} -n NearestNeighbor ' \
            '-i {labelImage} -o {newSegmentationImage}'

    if def_transformation_path:
        cmd_template = cmd_template + ' -t {defTransformation}'
        args_fmt['defTransformation'] = def_transformation_path

    app = cmd_template.format(**args_fmt)

    process = subpr.Popen(app, cwd=working_path, shell=True)
    streamdata = process.communicate()[0]
    rc = process.returncode
    print "antsApplyTransforms = %d" % rc

def apply_transform_fish_nifty(wokring_env, ref_image_path, transformation_path, labels_image_path, transformation_output, reg_prefix=None):
    working_path = wokring_env.get_working_path()

    args_fmt = {'refImage': ref_image_path, 'affineTransformation': transformation_path, 'labelImage': labels_image_path, 'newSegmentationImage': transformation_output}
    app3 = 'reg_resample -ref {refImage} -flo {labelImage} -trans \
    {affineTransformation} -res {newSegmentationImage} -inter 0'.format(**args_fmt)

    process = subpr.Popen(app3, cwd=working_path, shell=True)
    streamdata = process.communicate()[0]
    rc = process.returncode
    print "Nifty Reg Resample = %d" % rc

def apply_inverse_transform_fish(wokring_env, ref_image_path, affine_transformation_path, \
                                 labels_image_path, transformation_output, \
                                 inv_def_transformation_path=None, reg_prefix=None):
    working_path = wokring_env.get_working_path()
    os.environ["ANTSPATH"] = wokring_env.ANTSPATH

    args_fmt = {'refImage': ref_image_path, \
                'affineTransformation': affine_transformation_path, \
                'labelImage': labels_image_path, \
                'newSegmentationImage': transformation_output}

    cmd_template = 'antsApplyTransforms -d 3 -r {refImage} ' \
            '-t [{affineTransformation},1] -n NearestNeighbor ' \
            '-i {labelImage} -o {newSegmentationImage}'

    if inv_def_transformation_path:
        cmd_template = cmd_template + ' -t {invDefTransformation}'
        args_fmt['invDefTransformation'] = inv_def_transformation_path

    app = cmd_template.format(**args_fmt)

    process = subpr.Popen(app, cwd=working_path, shell=True)
    streamdata = process.communicate()[0]
    rc = process.returncode
    print "antsApplyTransforms (Inverse) = %d" % rc

def apply_inverse_transform_fish_nifty(wokring_env, ref_image_path, \
        inv_transformation_path, labels_image_path, transformation_output, \
        reg_prefix=None, interp_order=0):
    working_path = wokring_env.get_working_path()

    args_fmt = {'refImage': ref_image_path, 'invAffineTransformation': \
            inv_transformation_path, 'labelImage': labels_image_path, \
            'newSegmentationImage': transformation_output, 'interp_order': \
            interp_order}

    app3 = 'reg_resample -ref {refImage} -flo {labelImage} -trans \
    {invAffineTransformation} -res {newSegmentationImage} -inter {interp_order}'.format(**args_fmt)

    process = subpr.Popen(app3, cwd=working_path, shell=True)
    streamdata = process.communicate()[0]
    rc = process.returncode
    print "Nifty Reg Inverse Resample = %d" % rc

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
    objects_stats, _ = object_counter(stack_labels)
    objects_stats = objects_stats.sort(['area'], ascending=False)

    abdomen_part_z = objects_stats.loc[0, 'bb_z'] + objects_stats.loc[0, 'bb_depth']

    abdomen_data_part = stack_data[:abdomen_part_z,:,:]
    head_data_part = stack_data[(abdomen_part_z + 1):,:,:]

    return abdomen_data_part, head_data_part

def _zoom_bbox(bbox, scale):
    return tuple([slice(int(round(v.start * scale)), \
                        int(round(v.stop * scale)), \
                        int(round(v.step * scale)) if v.step else None) \
                                    for v in bbox])

# def _zoom_bbox(bbox, scale, target_size=None, is_downscale=True):
#     def _even_check(v, t):
#         return int(round(v * scale)) if t % 2 == 0 else int(round(v * scale)) - 1
#
#     if is_downscale:
#         return tuple([slice(int(round(v.start * scale)), \
#                             int(round(v.stop * scale)), \
#                             int(round(v.step * scale)) if v.step else None) \
#                      for v in bbox])
#     else:
#         return tuple([slice(_even_check(v.start * scale, t), \
#                             _even_check(v.stop * scale, t), \
#                             _even_check(v.step * scale, t) if v.step else None) \
#                     for v, t in bbox, target_size])

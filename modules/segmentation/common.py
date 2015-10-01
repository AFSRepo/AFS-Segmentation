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

    ext_volume_spine_labels_niigz_path = None
    zoomed_ext_volume_spine_labels_niigz_path = None

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

                print "Abdomen and brain labels are written and zoomed"

            if data_env.get_input_spine_labels_path():
                input_data_spine_labels = open_data(data_env.get_input_spine_labels_path())

                ext_volume_spine_labels = input_data_spine_labels[bbox]
                zoomed_0p5_ext_volume_spine_labels = zoom(ext_volume_spine_labels, 0.5, order=0)

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
            'scaled_0p5_extracted_spine_labels': zoomed_ext_volume_spine_labels_niigz_path, 'extracted_spine_labels': ext_volume_spine_labels_niigz_path}

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
    binary_stack_stats, _ = object_counter(binarized_stack)
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

def align_fish(working_env, fixed_image_path, moving_image_path, output_name, warped_path, iwarped_path, reg_prefix=None, use_syn=False, small_volume=False, syn_big_data_case=1, rigid_case=1):

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
    app3 = 'antsApplyTransforms -d 3 -r {refImage} -t {affineTransformation} -n NearestNeighbor -i {labelImage} -o {newSegmentationImage} --float 1'.format(**args_fmt)

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
    objects_stats, _ = object_counter(stack_labels)
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

import os
import sys
import pickle
import subprocess as subpr
import numpy as np
from modules.segmentation.spine import run_spine_segmentation
from modules.segmentation.eyes import eyes_statistics, eyes_zrange
from modules.tools.morphology import object_counter, gather_statistics
from modules.tools.morphology import extract_largest_area_data, extract_label_by_name
from modules.tools.morphology import extract_largest_volume_by_label, extract_effective_volume
from modules.tools.io import get_filename, open_data, save_as_nifti, check_files, parse_filename
from modules.tools.processing import binarizator, get_aligned_fish
from modules.tools.env import DataEnvironment
from modules.tools.misc import Timer, timing
from scipy.ndimage.interpolation import zoom, rotate
from modules.tools.io import ANTS_SCRIPTS_PATH_FMT, ORGAN_LABEL_TEMPLATE, ORGAN_DATA_TEMPLATE
from scipy.ndimage import binary_dilation, generate_binary_structure

# TODO:
# 1. We detect and find data+labels in initialize_env ---> NO NEED trying to
#    find labels on the stage of DataEnvironment feeding.
# 2. Add checking of label existense, if a label is already aligned and located
#    in AFS-output/Aligned/fish#, then we don't try to find anything in
#    MedakaRawData/fish#


#ANTs - 0 , NiftyReg - 1
REG_TOOL = 1
ZOOM_KEY = "zoomed"
NORMAL_KEY = "normal"

def initialize_env(data_env, zoom_level=2, min_zoom_level=2, organs_labels=None):
    print '--Aligning fish%d' % data_env.fish_num

    t = Timer()
    phase_name = 'extracted_input_data_path_niigz'
    phase_name_zoomed = 'zoomed_0p5_extracted_input_data_path_niigz'

    data_env.load()

    ext_volume_niigz_path = None
    zoomed_ext_volume_niigz_path = None

    ext_volume_labels_niigz_path = None
    zoomed_ext_volume_labels_niigz_path = None

    #if True:
    if not data_env.is_entry_exists(phase_name):
        if data_env.get_input_path():
            #aligned_data, aligned_data_label = get_aligned_fish(data_env.fish_num, zoom_level=2)
            #zoomed_aligned_data, zoomed_aligned_data_label = get_aligned_fish(data_env.fish_num, zoom_level=4)zoom_level=2, min_zoom_level=2

            #aligned_data, aligned_data_label = np.zeros((1,1), dtype=np.float32), np.zeros((1,1), dtype=np.float32)
            aligned_data, aligned_data_label, aligned_data_organs_labels = get_aligned_fish(data_env.fish_num, zoom_level=zoom_level, min_zoom_level=min_zoom_level, organs_labels=organs_labels)
            zoomed_aligned_data, zoomed_aligned_data_label, zoomed_aligned_data_organs_labels = get_aligned_fish(data_env.fish_num, zoom_level=zoom_level*2, min_zoom_level=min_zoom_level, organs_labels=organs_labels)

            ext_volume_niigz_path = data_env.get_new_volume_niigz_path(aligned_data.shape, 'extracted')
            zoomed_ext_volume_niigz_path = data_env.get_new_volume_niigz_path(zoomed_aligned_data.shape, 'zoomed_0p5_extracted')

            ext_volume_path = data_env.get_new_volume_path(aligned_data.shape, 'extracted')
            zoomed_ext_volume_path = data_env.get_new_volume_path(zoomed_aligned_data.shape, 'zoomed_0p5_extracted')

            save_as_nifti(aligned_data, ext_volume_niigz_path)
            save_as_nifti(zoomed_aligned_data, zoomed_ext_volume_niigz_path)

            aligned_data.tofile(ext_volume_path)
            zoomed_aligned_data.tofile(zoomed_ext_volume_path)

            data_shape = aligned_data.shape
            zoomed_data_shape = zoomed_aligned_data.shape

            data_bbox = np.index_exp[:data_shape[0], :data_shape[1], :data_shape[2]]
            zoomed_data_bbox = np.index_exp[:zoomed_data_shape[0], :zoomed_data_shape[1], :zoomed_data_shape[2]]

            data_env.set_effective_volume_bbox(data_bbox)
            data_env.set_zoomed_effective_volume_bbox(zoomed_data_bbox)
            data_env.set_input_align_data_path(ext_volume_path)

            if aligned_data_label is not None:
                print "#################### aligned_data_label = %s" % str(aligned_data_label.shape)
                ext_volume_labels_niigz_path = data_env.get_new_volume_labels_niigz_path(aligned_data_label.shape, 'extracted')
                zoomed_ext_volume_labels_niigz_path = data_env.get_new_volume_labels_niigz_path(zoomed_aligned_data_label.shape, 'zoomed_0p5_extracted')

                ext_volume_labels_path = data_env.get_new_volume_labels_path(aligned_data_label.shape, 'extracted')
                zoomed_ext_volume_labels_path = data_env.get_new_volume_labels_path(zoomed_aligned_data_label.shape, 'zoomed_0p5_extracted')

                save_as_nifti(aligned_data_label, ext_volume_labels_niigz_path)
                save_as_nifti(zoomed_aligned_data_label, zoomed_ext_volume_labels_niigz_path)

                aligned_data_label.tofile(ext_volume_labels_path)
                zoomed_aligned_data_label.tofile(zoomed_ext_volume_labels_path)

                data_env.set_input_aligned_data_labels_path(ext_volume_labels_path)

                print "Abdomen and brain labels are written and zoomed"

            if (aligned_data_organs_labels is not None) and (zoomed_aligned_data_organs_labels is not None):
                print "#################### aligned_data_organs_labels"

                for organ_name in aligned_data_organs_labels.keys():
                    ext_normal_organ_labels = aligned_data_organs_labels[organ_name]
                    zoomed_ext_normal_organ_labels = zoomed_aligned_data_organs_labels[organ_name]

                    ext_normal_organ_labels_niigz_path = data_env.get_new_volume_labels_niigz_path(ext_normal_organ_labels.shape, 'extracted_%s_labels' % organ_name)
                    zoomed_ext_normal_organ_labels_niigz_path = data_env.get_new_volume_labels_niigz_path(zoomed_ext_normal_organ_labels.shape, 'zoomed_0p5_extracted_%s_labels' % organ_name)

                    save_as_nifti(ext_normal_organ_labels, ext_normal_organ_labels_niigz_path)
                    save_as_nifti(zoomed_ext_normal_organ_labels, zoomed_ext_normal_organ_labels_niigz_path)

                    dict_organs_labels = data_env.get_organs_labels()
                    dict_organs_labels[organ_name] = { 'normal': ext_normal_organ_labels_niigz_path, 'zoomed': zoomed_ext_normal_organ_labels_niigz_path }
                    data_env.set_organs_labels(dict_organs_labels)

                    update_organs_envs(data_env, "original_organs", organ_name, \
                                       'label', ZOOM_KEY, \
                                       zoomed_ext_normal_organ_labels_niigz_path)

                    update_organs_envs(data_env, "original_organs", organ_name, \
                                       'label', NORMAL_KEY, \
                                       ext_normal_organ_labels_niigz_path)

                    update_organs_envs(data_env, "original_organs", organ_name, \
                                       'data', ZOOM_KEY, \
                                       zoomed_ext_volume_niigz_path)

                    update_organs_envs(data_env, "original_organs", organ_name, \
                                       'data', NORMAL_KEY, \
                                       ext_volume_niigz_path)

                    print "ORGANS are INITIALIZED!"

        else:
            print 'There\'s no input data'

        data_env.save()

        t.elapsed('Data preparing is finished')
    else:
        print 'Files of \'%s\' phase are already in working directory: %s' % (phase_name, data_env.get_working_path())

    return {'scaled_0p5_extracted': zoomed_ext_volume_niigz_path, 'extracted': ext_volume_niigz_path, \
            'scaled_0p5_extracted_labels': zoomed_ext_volume_labels_niigz_path, 'extracted_labels': ext_volume_labels_niigz_path}

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

@timing
def full_body_registration_ants(reference_data_env, target_data_env):
    reference_data_env.load()
    target_data_env.load()

    # Crop the raw data
    print "--Aligning and volumes' extraction"

    reference_data_results = initialize_env(reference_data_env, zoom_level=2, min_zoom_level=2, organs_labels=organs_labels)
    moving_data_results = initialize_env(target_data_env, zoom_level=2, min_zoom_level=2)

    reference_data_env.save()
    target_data_env.save()

    print "--Pre-alignment of the target fish to the known one"
    # Pre-alignment fish1 to fish_aligned
    ants_prefix_prealign = 'full_body_pre_alignment'
    ants_prealign_paths = target_data_env.get_aligned_data_paths(ants_prefix_prealign)
    ants_prealign_names = target_data_env.get_aligned_data_paths(ants_prefix_prealign, produce_paths=False)

    working_env_prealign = target_data_env
    reference_image_path_prealign = reference_data_env.envs['zoomed_0p5_extracted_input_data_path_niigz']
    reference_image_path_prealign_raw = reference_data_env.envs['zoomed_0p5_extracted_input_data_path']
    target_image_path_prealign = target_data_env.envs['zoomed_0p5_extracted_input_data_path_niigz']
    output_name_prealign = ants_prealign_names['out_name']
    warped_path_prealign = ants_prealign_paths['warped']
    iwarped_path_prealign = ants_prealign_paths['iwarp']

    print 'reference_image_path_prealign = %s' % reference_image_path_prealign
    print 'target_image_path_prealign = %s' % target_image_path_prealign

    if not os.path.exists(warped_path_prealign):
        align_fish_simple_ants(working_env_prealign, reference_image_path_prealign,\
                               target_image_path_prealign, output_name_prealign, \
                               warped_path_prealign, iwarped_path_prealign, \
                               reg_prefix=ants_prefix_prealign, use_syn=False, \
                               use_full_iters=False)

        reference_data_env.save()
        target_data_env.save()
    else:
        print "Data is already prealigned"

    #print  "--FINITO LA COMEDIA!"
    #return

    print  "--Registration of the reference fish to the target one"
    # Registration of fish_aligned to fish1
    ants_prefix_sep = 'full_body_registration'
    ants_separation_paths = reference_data_env.get_aligned_data_paths(ants_prefix_sep)
    working_env_sep = reference_data_env
    reference_image_path_sep = warped_path_prealign
    target_image_path_sep = reference_image_path_prealign
    output_name_sep = ants_separation_paths['out_name']
    warped_path_sep = ants_separation_paths['warped']
    iwarped_path_sep = ants_separation_paths['iwarp']

    if not os.path.exists(warped_path_sep):
        align_fish_simple_ants(working_env_sep, reference_image_path_sep, target_image_path_sep, \
                               output_name_sep, warped_path_sep, iwarped_path_sep, \
                               reg_prefix=ants_prefix_sep, use_syn=True, use_full_iters=True)

        reference_data_env.save()
        target_data_env.save()
    else:
        print "Data is already registered for separation"

    print "--Transforming brain and abdomen labels of the reference fish to the target's one"
    # Transforming labels of fish_aligned to fish1
    wokring_env_tr = target_data_env
    ref_image_path_tr = ants_prealign_paths['warped']
    affine_transformation_path_tr = ants_separation_paths['gen_affine']
    def_transformation_path_tr = ants_separation_paths['warp']
    labels_image_path_tr = reference_data_env.envs['zoomed_0p5_extracted_input_data_labels_path_niigz']

    print ref_image_path_tr
    print affine_transformation_path_tr
    print def_transformation_path_tr
    print labels_image_path_tr


    __, __, new_size, __ = parse_filename(reference_image_path_prealign_raw)

    transformation_output_tr = target_data_env.get_new_volume_niigz_path(new_size, 'full_body_zoomed_0p5_extracted_labels', bits=8)
    reg_prefix_tr = 'full_body_label_deforming'

    print transformation_output_tr

    if not os.path.exists(transformation_output_tr):
        apply_transform_fish(wokring_env_tr, ref_image_path_tr, \
                             affine_transformation_path_tr, \
                             labels_image_path_tr, transformation_output_tr, \
                             def_transformation_path=def_transformation_path_tr, \
                             reg_prefix=reg_prefix_tr)

        reference_data_env.save()
        target_data_env.save()
    else:
        print "Abdomen and brain data is already transformed"

def simple_heart_segmentation_ants(reference_data_env, target_data_env):
    reference_data_env.load()
    target_data_env.load()

    print "--Transfrom labels of reference fish's organs into the target's one..."

    target_organs_labels_dict = target_data_env.get_organs_labels()
    reference_organs_labels_dict = reference_data_env.get_organs_labels()

    print 'reference_organs_labels_dict = %s' % str(reference_organs_labels_dict)
    print 'target_labels_dict = %s' % str(target_organs_labels_dict)

    ants_prefix_sep = 'parts_separation'
    ants_registration_paths = reference_data_env.get_aligned_data_paths(ants_prefix_sep)

    if True:
    #if not target_organs_labels_dict:
        for organ_name, organ_labels in reference_organs_labels_dict.iteritems():
            print '----Extracting abdomenal organ %s' % organ_name

            organ_ref_image_path_ext_htr = target_data_env.envs['zoomed_0p5_extracted_input_data_path_niigz']
            organ_transformation_path_ext_htr = ants_registration_paths['gen_affine']
            organ_def_transformation_path_ext_htr = ants_registration_paths['warp']
            organ_labels_image_path_ext_htr = organ_labels['zoomed']

            organ_test_data_ext_htr = open_data(organ_ref_image_path_ext_htr)
            organ_transformation_output_ext_htr = target_data_env.get_new_volume_niigz_path(organ_test_data_ext_htr.shape, 'zoomed_0p5_extracted_organ_%s_labels' % organ_name)
            organ_reg_prefix_ext_htr = 'extracted_organ_%s_label_deforming_fullbody' % organ_name

            if True:
            #if not os.path.exists(organ_transformation_output_ext_htr):
                apply_transform_fish(target_data_env, organ_ref_image_path_ext_htr, \
                                     organ_transformation_path_ext_htr, organ_labels_image_path_ext_htr, \
                                     organ_transformation_output_ext_htr, \
                                     def_transformation_path=organ_def_transformation_path_ext_htr, \
                                     reg_prefix=organ_reg_prefix_ext_htr)

                target_organs_labels_dict[organ_name] = { 'normal': None, 'zoomed': organ_transformation_output_ext_htr }
                target_data_env.set_organs_labels(target_organs_labels_dict)

                reference_data_env.save()
                target_data_env.save()
            else:
                print "Extracted organ '%s' data is already transformed" % organ_name

    reference_data_env.save()
    target_data_env.save()

    print 'reference_organs_labels_dict = %s' % str(reference_data_env.get_organs_labels())
    print 'target_labels_dict = %s' % str(target_data_env.get_organs_labels())

    print "--Upscale the target fish's organs labels to the initial volume size..."
    target_organs_labels_dict = target_data_env.get_organs_labels()

    if True:
    #if target_organs_labels_dict:
        for organ_name, organ_labels in target_organs_labels_dict.iteritems():
            zoomed_organ_label_path = organ_labels['zoomed']

            original_aligned_data_path = target_data_env.get_input_align_data_path()
            original_aligned_data = open_data(original_aligned_data_path)

            upscaled_organ_label_niigz_path = target_data_env.get_new_volume_niigz_path(original_aligned_data.shape, 'extracted_organ_%s_labels' % organ_name, bits=8)

            if True:
            #if not os.path.exists(upscaled_organ_label_niigz_path):
                upscaled_organ_label_data = scale_to_size(original_aligned_data_path, \
                                                          zoomed_organ_label_path, \
                                                          scale=2.0, \
                                                          order=0)
                save_as_nifti(upscaled_organ_label_data, upscaled_organ_label_niigz_path)
                target_organs_labels_dict[organ_name]['normal'] = upscaled_organ_label_niigz_path

        target_data_env.set_organs_labels(target_organs_labels_dict)

        reference_data_env.save()
        target_data_env.save()
    else:
        print "The target fish's organs labels are already upscaled to the input volume size."

    print 'reference_organs_labels_dict = %s' % str(reference_data_env.get_organs_labels())
    print 'target_labels_dict = %s' % str(target_data_env.get_organs_labels())

    bb_side_offset = 20

    print "--Extract the target fish's organs labels from the initial volume..."
    target_extracted_organs_dict = target_data_env.get_extracted_organs()
    target_extracted_organs_labels_dict = target_data_env.get_extracted_organs_labels()
    target_organs_labels_dict = target_data_env.get_organs_labels()

    if True:
    #if (not target_extracted_organs_labels_dict) and (not target_extracted_organs_dict):
        for organ_name, organ_labels in target_organs_labels_dict.iteritems():
            normal_organ_label_path = organ_labels['normal']
            normal_organ_label_data = open_data(normal_organ_label_path)

            original_aligned_data_path = target_data_env.get_input_align_data_path()
            original_aligned_data = open_data(original_aligned_data_path)

            extracted_organ, extracted_organ_bbox = extract_largest_volume_by_label(original_aligned_data, normal_organ_label_data, bb_side_offset=bb_side_offset)
            extracted_organ_label = normal_organ_label_data[extracted_organ_bbox]

            extracted_organ_niigz_path = target_data_env.get_new_volume_niigz_path(extracted_organ.shape, 'extracted_roi_organ_%s' % organ_name)
            extracted_organ_label_niigz_path = target_data_env.get_new_volume_niigz_path(extracted_organ_label.shape, 'extracted_roi_organ_%s_label' % organ_name, bits=8)

            print extracted_organ_niigz_path
            print extracted_organ_label_niigz_path

            if True:
            #if not os.path.exists(extracted_organ_niigz_path):
                save_as_nifti(extracted_organ, extracted_organ_niigz_path)

            if True:
            #if not os.path.exists(extracted_organ_label_niigz_path):
                save_as_nifti(extracted_organ_label, extracted_organ_label_niigz_path)

            target_extracted_organs_dict[organ_name] = { 'normal': extracted_organ_niigz_path, 'zoomed': None }
            target_extracted_organs_labels_dict[organ_name] = { 'normal': extracted_organ_label_niigz_path, 'zoomed': None }

            target_data_env.set_extracted_organs(target_extracted_organs_dict)
            target_data_env.set_extracted_organs_labels(target_extracted_organs_labels_dict)

            reference_data_env.save()
            target_data_env.save()

    print 'target_data_env.get_extracted_organs = %s' % str(target_data_env.get_extracted_organs())
    print 'target_data_env.get_extracted_organs_labels = %s' % str(target_data_env.get_extracted_organs_labels())

    print "--Extract the reference fish's organs labels from the initial volume..."
    reference_extracted_organs_dict = reference_data_env.get_extracted_organs()
    reference_extracted_organs_labels_dict = reference_data_env.get_extracted_organs_labels()
    reference_organs_labels_dict = reference_data_env.get_organs_labels()

    if True:
    #if (not reference_extracted_organs_labels_dict) and (not reference_extracted_organs_dict):
        for organ_name, organ_labels in reference_organs_labels_dict.iteritems():
            normal_organ_label_path = organ_labels['normal']
            normal_organ_label_data = open_data(normal_organ_label_path)

            original_aligned_data_path = reference_data_env.get_input_align_data_path()
            original_aligned_data = open_data(original_aligned_data_path)

            extracted_organ, extracted_organ_bbox = extract_largest_volume_by_label(original_aligned_data, normal_organ_label_data, bb_side_offset=bb_side_offset)
            extracted_organ_label = normal_organ_label_data[extracted_organ_bbox]

            extracted_organ_niigz_path = reference_data_env.get_new_volume_niigz_path(extracted_organ.shape, 'extracted_roi_organ_%s' % organ_name)
            extracted_organ_label_niigz_path = reference_data_env.get_new_volume_niigz_path(extracted_organ_label.shape, 'extracted_roi_organ_%s_label' % organ_name, bits=8)

            print extracted_organ_niigz_path
            print extracted_organ_label_niigz_path

            if True:
            #if not os.path.exists(extracted_organ_niigz_path):
                save_as_nifti(extracted_organ, extracted_organ_niigz_path)

            if True:
            #if not os.path.exists(extracted_organ_label_niigz_path):
                save_as_nifti(extracted_organ_label, extracted_organ_label_niigz_path)

            reference_extracted_organs_dict[organ_name] = { 'normal': extracted_organ_niigz_path, 'zoomed': None }
            reference_extracted_organs_labels_dict[organ_name] = { 'normal': extracted_organ_label_niigz_path, 'zoomed': None }

            reference_data_env.set_extracted_organs(reference_extracted_organs_dict)
            reference_data_env.set_extracted_organs_labels(reference_extracted_organs_labels_dict)

            reference_data_env.save()
            target_data_env.save()

    print 'reference_data_env.get_extracted_organs = %s' % str(reference_data_env.get_extracted_organs())
    print 'reference_data_env.get_extracted_organs_labels = %s' % str(reference_data_env.get_extracted_organs_labels())

    print "--Register the extracted reference fish's orgnas to the target's one..."
    target_extracted_organs_dict = target_data_env.get_extracted_organs()
    reference_extracted_organs_dict = reference_data_env.get_extracted_organs()

    reference_extracted_organs_registration_dict = reference_data_env.get_extracted_organs_registration()
    if True:
    #if not reference_extracted_organs_registration_dict:
        for organ_name, _ in reference_extracted_organs_dict.iteritems():

            ants_prefix_reg = 'andomen_extracted_%s_registration' % organ_name
            ants_reg_paths = reference_data_env.get_aligned_data_paths(ants_prefix_reg)
            output_name_reg = ants_reg_paths['out_name']
            warped_path_reg = ants_reg_paths['warped']
            iwarped_path_reg = ants_reg_paths['iwarp']

            if True:
            #if not os.path.exists(warped_path_reg):
                align_fish_simple_ants(reference_data_env, \
                               target_extracted_organs_dict[organ_name]['normal'], \
                               reference_extracted_organs_dict[organ_name]['normal'], \
                               output_name_reg, \
                               warped_path_reg, \
                               iwarped_path_reg, \
                               reg_prefix=ants_prefix_reg, \
                               use_syn=True, \
                               use_full_iters=True)

                reference_extracted_organs_registration_dict[organ_name] = { 'normal': ants_prefix_reg, 'zoomed': None }
                reference_data_env.set_extracted_organs_registration(reference_extracted_organs_registration_dict)

                reference_data_env.save()
                target_data_env.save()
    else:
        print "The organs of the reference data is already registered to the target one"


    print "--Transfrom extracted labels of reference fish's organs into the target's one..."
    target_extracted_organs_dict = target_data_env.get_extracted_organs()
    reference_extracted_organs_labels_dict = reference_data_env.get_extracted_organs_labels()
    reference_extracted_organs_registration_dict = reference_data_env.get_extracted_organs_registration()
    target_extracted_organs_registration_labels_dict = target_data_env.get_extracted_organs_registration_labels()

    if True:
    #if not target_extracted_organs_registration_labels_dict:
        for organ_name, organ_reg_prefixes in reference_extracted_organs_registration_dict.iteritems():
            print '----Transforming abdomenal organ %s' % organ_name
            ants_reg_paths = reference_data_env.get_aligned_data_paths(organ_reg_prefixes['normal'])

            organ_ref_image_path = target_extracted_organs_dict[organ_name]['normal']
            organ_transformation_path = ants_reg_paths['gen_affine']
            organ_def_transformation_path = ants_reg_paths['warp']
            organ_labels_image_path = reference_extracted_organs_labels_dict[organ_name]['normal']

            organ_test_data = open_data(organ_ref_image_path)
            organ_transformation_output = target_data_env.get_new_volume_niigz_path(organ_test_data.shape, 'extracted_deformed_organ_%s_labels' % organ_name)
            organ_reg_prefix = 'extracted_organ_%s_label_deforming_extracted_region' % organ_name

            if True:
            #if not os.path.exists(organ_transformation_output):
                apply_transform_fish(target_data_env, organ_ref_image_path, \
                                     organ_transformation_path, organ_labels_image_path, \
                                     organ_transformation_output, \
                                     def_transformation_path=organ_def_transformation_path, \
                                     reg_prefix=organ_reg_prefix)

                target_extracted_organs_registration_labels_dict[organ_name] = { 'normal': None, 'zoomed': organ_transformation_output }
                target_data_env.set_extracted_organs_registration_labels(target_extracted_organs_registration_labels_dict)

                reference_data_env.save()
                target_data_env.save()
            else:
                print "Extracted organ '%s' data is already transformed" % organ_name

    reference_data_env.save()
    target_data_env.save()

    print "--Segment target fish's organs data..."
    target_extracted_organs_dict = target_data_env.get_extracted_organs()
    target_extracted_organs_labels_dict = target_data_env.get_extracted_organs_labels()

    target_extracted_organs_masked_dict = target_data_env.get_extracted_organs_masked()
    if True:
    #if not target_extracted_organs_masked_dict:
        for organ_name, _ in target_extracted_organs_dict.iteritems():
            normal_organ_label_path = target_extracted_organs_labels_dict[organ_name]['normal']
            normal_organ_data_path = target_extracted_organs_dict[organ_name]['normal']

            normal_organ_label = open_data(normal_organ_label_path)
            normal_organ_data = open_data(normal_organ_data_path)

            masked_normal_organ_data = normal_organ_data * normal_organ_label

            masked_normal_organ_data_niigz_path = \
            target_data_env.get_new_volume_niigz_path(masked_normal_organ_data.shape, 'extracted_segmented_roi_organ_%s' % organ_name)

            if True:
            #if not os.path.exists(masked_normal_organ_data_niigz_path):
                save_as_nifti(masked_normal_organ_data, masked_normal_organ_data_niigz_path)

            target_extracted_organs_masked_dict[organ_name] = { 'normal': masked_normal_organ_data_niigz_path, 'zoomed': None }
            target_data_env.set_extracted_organs_masked(target_extracted_organs_masked_dict)

            reference_data_env.save()
            target_data_env.save()

    print "--Segment reference fish's organs data..."
    reference_extracted_organs_dict = reference_data_env.get_extracted_organs()
    reference_extracted_organs_labels_dict = reference_data_env.get_extracted_organs_labels()

    reference_extracted_organs_masked_dict = reference_data_env.get_extracted_organs_masked()
    if True:
    #if not reference_extracted_organs_masked_dict:
        for organ_name, _ in reference_extracted_organs_dict.iteritems():
            normal_organ_label_path = reference_extracted_organs_labels_dict[organ_name]['normal']
            normal_organ_data_path = reference_extracted_organs_dict[organ_name]['normal']

            normal_organ_label = open_data(normal_organ_label_path)
            normal_organ_data = open_data(normal_organ_data_path)

            masked_normal_organ_data = normal_organ_data * normal_organ_label

            masked_normal_organ_data_niigz_path = \
            reference_data_env.get_new_volume_niigz_path(masked_normal_organ_data.shape, 'extracted_segmented_roi_organ_%s' % organ_name)

            if True:
            # if not os.path.exists(masked_normal_organ_data_niigz_path):
                save_as_nifti(masked_normal_organ_data, masked_normal_organ_data_niigz_path)

            reference_extracted_organs_masked_dict[organ_name] = { 'normal': masked_normal_organ_data_niigz_path, 'zoomed': None }
            reference_data_env.set_extracted_organs_masked(reference_extracted_organs_masked_dict)

            reference_data_env.save()
            target_data_env.save()

def heart_segmentation_ants(reference_data_env, target_data_env):
    bb_side_offset = 10
    reference_data_env.load()
    target_data_env.load()

    print 'reference_data_env.zoomed_0p5_abdomen_input_data_path_niigz = %s' % reference_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']
    print 'target_data_env.zoomed_0p5_abdomen_input_data_path_niigz = %s' % target_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']


    print "--Register reference fish's abdomenal part to the target's one..."
    #Register reference abdomen to target one
    ants_prefix_abdomen_reg = 'abdomen_registration'
    ants_abdomen_reg_paths = reference_data_env.get_aligned_data_paths(ants_prefix_abdomen_reg)
    reference_image_path_abdomen_reg = reference_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']
    target_image_path_abdomen_reg = target_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']
    output_name_abdomen_reg = ants_abdomen_reg_paths['out_name']
    warped_path_abdomen_reg = ants_abdomen_reg_paths['warped']
    iwarped_path_abdomen_reg = ants_abdomen_reg_paths['iwarp']

    if True:
    #if not os.path.exists(warped_path_abdomen_reg):
        align_fish_simple_ants(reference_data_env, target_image_path_abdomen_reg, \
                               reference_image_path_abdomen_reg, output_name_abdomen_reg, \
                               warped_path_abdomen_reg, iwarped_path_abdomen_reg, \
                               reg_prefix=ants_prefix_abdomen_reg, use_syn=True, \
                               use_full_iters=False)

        reference_data_env.save()
        target_data_env.save()
    else:
        print "Abdomenal part of the reference data is already registered to the abdomenal part of target one"

    print "--Transfrom labels of reference fish's abdomenal part into the target's one..."
    # Transforming labels of abdomenal part of reference fish to the abdomenal part of target one
    ref_image_path_htr = target_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']
    transformation_path_htr = ants_abdomen_reg_paths['gen_affine']
    def_transformation_path_htr = ants_abdomen_reg_paths['warp']
    labels_image_path_htr = reference_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']
    test_data_htr = open_data(ref_image_path_htr)
    transformation_output_htr = target_data_env.get_new_volume_niigz_path(test_data_htr.shape, 'zoomed_0p5_abdomen_labels', bits=8)
    reg_prefix_htr = 'abdomen_label_deforming'

    if True:
    #if not os.path.exists(transformation_output_htr):
        apply_transform_fish(target_data_env, ref_image_path_htr, \
                             transformation_path_htr, labels_image_path_htr, \
                             transformation_output_htr, \
                             def_transformation_path=def_transformation_path_htr, \
                             reg_prefix=reg_prefix_htr)
        reference_data_env.save()
        target_data_env.save()
    else:
        print "Abdomenal part data is already transformed"

    print "--Extract the target fish's abdomenal part using transformed abdomenal part labels..."
    # Extract target abdomenal part volume
    abdomen_part_label_target = open_data(transformation_output_htr)
    abdomen_part_data_target = open_data(ref_image_path_htr)
    abdomen_part_data_volume_target, abdomen_part_data_volume_target_bbox = extract_largest_volume_by_label(abdomen_part_data_target, abdomen_part_label_target, bb_side_offset=bb_side_offset)
    abdomen_part_data_volume_target_niigz_path = target_data_env.get_new_volume_niigz_path(abdomen_part_data_volume_target.shape, 'zoomed_0p5_extracted_abdomen_part')

    print abdomen_part_data_volume_target_niigz_path

    # if True:
    if not os.path.exists(abdomen_part_data_volume_target_niigz_path):
        save_as_nifti(abdomen_part_data_volume_target, abdomen_part_data_volume_target_niigz_path)

    reference_data_env.save()
    target_data_env.save()

    print "--Extract the reference fish's abdomenal part using transformed abdomenal part labels..."
    # Extract reference brain volume
    abdomen_part_data_reference = open_data(reference_image_path_abdomen_reg)
    abdomen_part_label_reference = open_data(labels_image_path_htr)
    abdomen_part_data_volume_reference, abdomen_part_reference_bbox = extract_largest_volume_by_label(abdomen_part_data_reference, abdomen_part_label_reference, bb_side_offset=bb_side_offset)
    abdomen_part_data_labels_volume_reference = abdomen_part_label_reference[abdomen_part_reference_bbox]

    abdomen_part_data_volume_reference_niigz_path = reference_data_env.get_new_volume_niigz_path(abdomen_part_data_volume_reference.shape, 'zoomed_0p5_extracted_abdomen_part')
    abdomen_part_data_labels_volume_reference_niigz_path = reference_data_env.get_new_volume_niigz_path(abdomen_part_data_labels_volume_reference.shape, 'zoomed_0p5_extracted_abdomen_part_labels')

    print abdomen_part_data_volume_reference_niigz_path
    print abdomen_part_data_labels_volume_reference_niigz_path

    # if True:
    if not os.path.exists(abdomen_part_data_volume_reference_niigz_path):
        save_as_nifti(abdomen_part_data_volume_reference, abdomen_part_data_volume_reference_niigz_path)

    # if True:
    if not os.path.exists(abdomen_part_data_labels_volume_reference_niigz_path):
        save_as_nifti(abdomen_part_data_labels_volume_reference, abdomen_part_data_labels_volume_reference_niigz_path)

    reference_data_env.save()
    target_data_env.save()

    print "--Extract the reference fish's abdomenal part organs using transformed abdomenal part labels..."
    extracted_abdomen_part_separated_organs_labels_dict = reference_data_env.get_extracted_abdomen_part_separated_organs_labels()

    print 'extracted_abdomen_part_separated_organs_labels_dict = %s' % str(reference_data_env.get_extracted_abdomen_part_separated_organs_labels())

    if not extracted_abdomen_part_separated_organs_labels_dict:
        abdomen_separated_organs_labels_dict = reference_data_env.get_abdomen_separated_organs_labels()

        for organ_name, organ_labels in abdomen_separated_organs_labels_dict.iteritems():
            zoomed_abdomen_separated_organ_label_path = organ_labels['zoomed']
            zoomed_abdomen_separated_organ_label_data = open_data(zoomed_abdomen_separated_organ_label_path)

            abdomen_part_data_organ_label_volume_reference = zoomed_abdomen_separated_organ_label_data[abdomen_part_reference_bbox]

            abdomen_part_data_organ_label_volume_reference_niigz_path = reference_data_env.get_new_volume_niigz_path(abdomen_part_data_organ_label_volume_reference.shape, 'zoomed_0p5_extracted_abdomen_part_organ_%s_labels' % organ_name)
            if not os.path.exists(abdomen_part_data_organ_label_volume_reference_niigz_path):
                save_as_nifti(abdomen_part_data_organ_label_volume_reference, abdomen_part_data_organ_label_volume_reference_niigz_path)

            extracted_abdomen_part_separated_organs_labels_dict[organ_name] = { 'normal': None, 'zoomed': abdomen_part_data_organ_label_volume_reference_niigz_path }

    reference_data_env.set_extracted_abdomen_part_separated_organs_labels(extracted_abdomen_part_separated_organs_labels_dict)
    reference_data_env.save()

    print 'extracted_abdomen_part_separated_organs_labels_dict = %s' % str(reference_data_env.get_extracted_abdomen_part_separated_organs_labels())

    print "--Register reference fish's extracted abdomenal part to the target's one..."
    #Register reference abdomen to target one
    ants_prefix_ext_abdomen_reg = 'extracted_abdomen_part_registration'
    ants_ext_abdomen_reg_paths = reference_data_env.get_aligned_data_paths(ants_prefix_ext_abdomen_reg)

    reference_image_path_ext_abdomen_reg = abdomen_part_data_volume_reference_niigz_path
    target_image_path_ext_abdomen_reg = abdomen_part_data_volume_target_niigz_path

    output_name_ext_abdomen_reg = ants_ext_abdomen_reg_paths['out_name']
    warped_path_ext_abdomen_reg = ants_ext_abdomen_reg_paths['warped']
    iwarped_path_ext_abdomen_reg = ants_ext_abdomen_reg_paths['iwarp']

    if not os.path.exists(warped_path_ext_abdomen_reg):
        align_fish_simple_ants(reference_data_env, target_image_path_ext_abdomen_reg, \
                               reference_image_path_ext_abdomen_reg, output_name_ext_abdomen_reg, \
                               warped_path_ext_abdomen_reg, iwarped_path_ext_abdomen_reg, \
                               reg_prefix=ants_prefix_ext_abdomen_reg, use_syn=True, \
                               use_full_iters=True)

        reference_data_env.save()
        target_data_env.save()
    else:
        print "Extracted abdomenal part of the reference data is already registered to the extracted one of target"

    print "--Transfrom labels of reference fish's extracted abdomenal part into the target's one..."
    # Transforming labels of abdomenal part of reference fish to the abdomenal part of target one
    ref_image_path_ext_htr = target_image_path_ext_abdomen_reg
    transformation_path_ext_htr = ants_ext_abdomen_reg_paths['gen_affine']
    def_transformation_path_ext_htr = ants_ext_abdomen_reg_paths['warp']
    labels_image_path_ext_htr = abdomen_part_data_labels_volume_reference_niigz_path

    test_data_ext_htr = open_data(ref_image_path_ext_htr)
    transformation_output_ext_htr = target_data_env.get_new_volume_niigz_path(test_data_ext_htr.shape, 'zoomed_0p5_extracted_abdomen_part_labels')
    reg_prefix_ext_htr = 'extracted_abdomen_label_deforming'

    if not os.path.exists(transformation_output_ext_htr):
        apply_transform_fish(target_data_env, ref_image_path_ext_htr, \
                             transformation_path_ext_htr, labels_image_path_ext_htr, \
                             transformation_output_ext_htr, \
                             def_transformation_path=def_transformation_path_ext_htr, \
                             reg_prefix=reg_prefix_ext_htr)
        reference_data_env.save()
        target_data_env.save()
    else:
        print "Extracted abdomenal part data is already transformed"

    print "--Transfrom labels of reference fish's organs extracted abdomenal part into the target's one..."
    #extracted_abdomen_part_separated_organs_labels_dict

    reference_extracted_abdomen_part_separated_organs_labels_dict = reference_data_env.get_extracted_abdomen_part_separated_organs_labels()
    print 'reference_extracted_abdomen_part_separated_organs_labels_dict = %s' % str(reference_extracted_abdomen_part_separated_organs_labels_dict)

    print str(target_data_env.envs)
    target_extracted_abdomen_part_separated_organs_labels_dict = target_data_env.get_extracted_abdomen_part_separated_organs_labels()
    print 'target_extracted_abdomen_part_separated_organs_labels_dict = %s' % str(target_extracted_abdomen_part_separated_organs_labels_dict)

    if not target_extracted_abdomen_part_separated_organs_labels_dict:
        for organ_name, organ_labels in reference_extracted_abdomen_part_separated_organs_labels_dict.iteritems():
            print '----Extracting abdomenal organ %s' % organ_name

            organ_ref_image_path_ext_htr = target_image_path_ext_abdomen_reg
            organ_transformation_path_ext_htr = ants_ext_abdomen_reg_paths['gen_affine']
            organ_def_transformation_path_ext_htr = ants_ext_abdomen_reg_paths['warp']
            organ_labels_image_path_ext_htr = organ_labels['zoomed']

            organ_test_data_ext_htr = open_data(ref_image_path_ext_htr)
            organ_transformation_output_ext_htr = target_data_env.get_new_volume_niigz_path(test_data_ext_htr.shape, 'zoomed_0p5_extracted_abdomen_part_organ_%s_labels' % organ_name)
            organ_reg_prefix_ext_htr = 'extracted_abdomen_organ_%s_label_deforming' % organ_name

            if not os.path.exists(organ_transformation_output_ext_htr):
                apply_transform_fish(target_data_env, organ_ref_image_path_ext_htr, \
                                     organ_transformation_path_ext_htr, organ_labels_image_path_ext_htr, \
                                     organ_transformation_output_ext_htr, \
                                     def_transformation_path=organ_def_transformation_path_ext_htr, \
                                     reg_prefix=organ_reg_prefix_ext_htr)

                target_extracted_abdomen_part_separated_organs_labels_dict[organ_name] = { 'normal': None, 'zoomed': organ_transformation_output_ext_htr }
                target_data_env.set_extracted_abdomen_part_separated_organs_labels(target_extracted_abdomen_part_separated_organs_labels_dict)

                reference_data_env.save()
                target_data_env.save()
            else:
                print "Extracted abdomenal organ '%s' part data is already transformed" % organ_name

    reference_data_env.save()
    target_data_env.save()

    print 'reference_extracted_abdomen_part_separated_organs_labels_dict = %s' % str(reference_extracted_abdomen_part_separated_organs_labels_dict)
    print 'target_extracted_abdomen_part_separated_organs_labels_dict = %s' % str(target_extracted_abdomen_part_separated_organs_labels_dict)

def split_fish_body(input_data_path, input_data_label_path, output_name, \
               wokring_env, separation_overlap=5, zoom_key='zoomed'):
    input_data = open_data(input_data_path)
    input_data_labels = open_data(input_data_label_path)

    separation_pos, abdomen_label_full_volume, head_label_full_volume = find_separation_pos(input_data_labels)

    abdomen_data, head_data = split_fish_by_pos(input_data, separation_pos, overlap=separation_overlap)
    abdomen_data_label, _ = split_fish_by_pos(abdomen_label_full_volume, separation_pos, overlap=separation_overlap)
    _, head_data_label = split_fish_by_pos(head_label_full_volume, separation_pos, overlap=separation_overlap)

    abdomen_data_niigz_path = wokring_env.get_new_volume_niigz_path(abdomen_data.shape, \
                                        '_'.join([zoom_key, output_name, 'abdomen']))
    if True:
    #if not os.path.exists(abdomen_data_niigz_path):
        save_as_nifti(abdomen_data, abdomen_data_niigz_path)

    head_data_niigz_path = wokring_env.get_new_volume_niigz_path(head_data.shape, \
                                        '_'.join([zoom_key, output_name, 'head']))
    if True:
    #if not os.path.exists(head_data_niigz_path):
        save_as_nifti(head_data, head_data_niigz_path)

    abdomen_data_label_niigz_path = wokring_env.get_new_volume_niigz_path(abdomen_data_label.shape, \
                                        '_'.join([zoom_key, output_name, 'abdomen', 'label']))
    if True:
    #if not os.path.exists(abdomen_data_label_niigz_path):
        save_as_nifti(abdomen_data_label, abdomen_data_label_niigz_path)

    head_data_label_niigz_path = wokring_env.get_new_volume_niigz_path(head_data_label.shape, \
                                        '_'.join([zoom_key, output_name, 'head', 'label']))
    if True:
    #if not os.path.exists(head_data_label_niigz_path):
        save_as_nifti(head_data_label, head_data_label_niigz_path)

    wokring_env.save()

    return abdomen_data_niigz_path, abdomen_data_label_niigz_path, \
           head_data_niigz_path, head_data_label_niigz_path

def extract_largest_label_by_name(data_path, data_label_path, output_name, working_env, bb_side_offset=5, label_name='brain'):
    data_label = open_data(data_label_path)

    brain_data_label = extract_label_by_name(data_label, label_name='brain')

    brain_data_label_niigz_path = \
            working_env.get_new_volume_niigz_path(brain_data_label.shape, \
                                '_'.join([label_name, 'label']))

    if True:
    #if not os.path.exists(brain_data_label_niigz_path):
        save_as_nifti(brain_data_label, brain_data_label_niigz_path)

    return extract_largest_label(data_path, \
                                 brain_data_label_niigz_path, \
                                 output_name, \
                                 working_env, \
                                 bb_side_offset=bb_side_offset)

def extract_largest_label(data_path, data_label_path, output_name, working_env, bb_side_offset=5):
    data = open_data(data_path)
    data_label = open_data(data_label_path)

    extracted_data_volume, extracted_data_volume_bbox = \
            extract_largest_volume_by_label(data, data_label, bb_side_offset=bb_side_offset)

    extracted_data_volume_labels = data_label[extracted_data_volume_bbox]

    extracted_data_volume_niigz_path = \
        working_env.get_new_volume_niigz_path(extracted_data_volume.shape, output_name)

    extracted_data_volume_labels_niigz_path = \
        working_env.get_new_volume_niigz_path(extracted_data_volume_labels.shape, \
                '_'.join([output_name, 'label']))

    if True:
    #if not os.path.exists(extracted_data_volume_niigz_path):
        save_as_nifti(extracted_data_volume, extracted_data_volume_niigz_path)

    if True:
    #if not os.path.exists(extracted_data_volume_labels_niigz_path):
        save_as_nifti(extracted_data_volume_labels, extracted_data_volume_labels_niigz_path)

    working_env.save()

    return extracted_data_volume_niigz_path, \
           extracted_data_volume_labels_niigz_path, \
           extracted_data_volume_bbox

"""
Organs --
        |
        Organs specific set #1--
        |                      |
        |                    Organ #1--- Data
        |                           |        |
        |                           |        'normal': path
        |                           |        'zoomed': path
        |                           ---- Labels
        |                           |          |
        |                           |          'normal': path
        |                           |          'zoomed': path
        |                           ---- Bbox
        |                                      |
        |                                      'normal': path
        |                                      'zoomed': path

"""

def update_organs_envs(working_env, env_output_name, organ_name, \
                       data_type_key, zoom_key, path, bbox_key='bbox', bbox=None):
    organs_envs = working_env.get_organs_envs()

    empty_data_dict = { 'data':  {'normal':None, 'zoomed': None }, \
                        'label': {'normal':None, 'zoomed': None }, \
                        'bbox':  {'normal':None, 'zoomed': None } }

    if env_output_name in organs_envs:
        if organ_name in organs_envs[env_output_name]:
            if not isinstance(organs_envs[env_output_name][organ_name], dict):
                raise ValueError("""Organs data dictionary contains organ entry
                                which is not dictionary.""")
        else:
            organs_envs[env_output_name][organ_name] = empty_data_dict
    else:
        organs_envs[env_output_name] = { organ_name: empty_data_dict }

    organs_envs[env_output_name][organ_name][data_type_key][zoom_key] = path
    organs_envs[env_output_name][organ_name][bbox_key][zoom_key] = bbox

    working_env.set_organs_envs(organs_envs)
    working_env.save()

def split_fish_organs(input_organs_env_name, \
                      output_organs_env_name, \
                      input_data_labels_path, \
                      input_data_path, \
                      working_env, \
                      zoom_key, \
                      overlap=20):
    aligned_data_labels = open_data(input_data_labels_path)
    organ_separation_pos, _, _ = find_separation_pos(aligned_data_labels)

    input_data = open_data(input_data_path)

    abdomen_data, _ = split_fish_by_pos(input_data, organ_separation_pos, overlap=overlap)
    # print "\033[1;31m abdomen_data.shape = %s \033[0m" % str(abdomen_data.shape)

    abdomen_data_niigz_path = \
                working_env.get_new_volume_niigz_path(abdomen_data.shape, \
                        ORGAN_DATA_TEMPLATE % (output_organs_env_name, 'general'))

    organs_envs = working_env.get_organs_envs()

    # print "\033[1;31m Input data = %s \033[0m" % working_env.envs['zoomed_0p5_extracted_input_data_path_niigz']
    # print "\033[1;31m organs_envs = %s \033[0m" % str(organs_envs)

    if input_organs_env_name in organs_envs:
        input_organs = organs_envs[input_organs_env_name]

        for organ_name, organ_data_dict in input_organs.iteritems():
            organ_data_label_path = organ_data_dict['label'][zoom_key]
            organ_data_label = open_data(organ_data_label_path)

            abdomen_data_label_organ, _ = split_fish_by_pos(organ_data_label, organ_separation_pos, overlap=overlap)
            print "\033[1;31m abdomen_data_label_organ.shape = %s \033[0m" % str(abdomen_data_label_organ.shape)

            abdomen_data_label_organ_niigz_path = \
                    working_env.get_new_volume_niigz_path(abdomen_data_label_organ.shape, \
                            ORGAN_LABEL_TEMPLATE % (output_organs_env_name, organ_name))
            if True:
            #if not os.path.exists(abdomen_data_label_organ_niigz_path):
                save_as_nifti(abdomen_data_label_organ, abdomen_data_label_organ_niigz_path)

                if True:
                #if not os.path.exists(abdomen_data_niigz_path):
                    save_as_nifti(abdomen_data, abdomen_data_niigz_path)

                update_organs_envs(working_env, \
                                   output_organs_env_name, \
                                   organ_name, \
                                   'label', \
                                   zoom_key, \
                                   abdomen_data_label_organ_niigz_path)

                update_organs_envs(working_env, \
                                   output_organs_env_name, \
                                   organ_name, \
                                   'data', \
                                   zoom_key, \
                                   abdomen_data_niigz_path)

    # print "\033[1;31m working_env.get_organs_envs() = %s \033[0m" % str(working_env.get_organs_envs()[output_organs_env_name])

    return output_organs_env_name

def extract_organs_by_bbox(input_organs_env_name, \
                           output_organs_env_name, \
                           extraction_bbox, working_env, \
                           zoom_key, save_label=True):
    organs_envs = working_env.get_organs_envs()

    print 'organs_envs = %s' % str(organs_envs)

    if input_organs_env_name in organs_envs:
        input_organs = organs_envs[input_organs_env_name]

        for organ_name, organ_data_dict in input_organs.iteritems():
            organ_data_path = organ_data_dict['data'][zoom_key]
            organ_data = open_data(organ_data_path)

            print "\033[1;31m organ_data.shape = %s \033[0m" % str(organ_data.shape)
            print "\033[1;31m extraction_bbox = %s \033[0m" % str(extraction_bbox)
            extracted_organ_data_volume = organ_data[extraction_bbox]
            print "\033[1;31m extracted_organ_data_volume = %s \033[0m" % str(extracted_organ_data_volume.shape)

            extracted_organ_data_volume_niigz_path = \
                    working_env.get_new_volume_niigz_path(extracted_organ_data_volume.shape, \
                        ORGAN_DATA_TEMPLATE % (output_organs_env_name, organ_name))


            if save_label:
                organ_data_label_path = organ_data_dict['label'][zoom_key]
                organ_data_label = open_data(organ_data_label_path)

                extracted_organ_data_label_volume = organ_data_label[extraction_bbox]

                extracted_organ_data_label_volume_niigz_path = \
                    working_env.get_new_volume_niigz_path(extracted_organ_data_label_volume.shape, \
                        ORGAN_LABEL_TEMPLATE % (output_organs_env_name, organ_name))

                if extracted_organ_data_label_volume_niigz_path is not None:
                    if True:
                    #if not os.path.exists(extracted_organ_data_label_volume_niigz_path):
                        save_as_nifti(extracted_organ_data_label_volume, extracted_organ_data_label_volume_niigz_path)

                        update_organs_envs(working_env, output_organs_env_name, \
                                           organ_name, 'label', zoom_key, \
                                           extracted_organ_data_label_volume_niigz_path, \
                                           bbox=pickle.dumps(extraction_bbox))

            if True:
            #if not os.path.exists(extracted_organ_data_volume_niigz_path):
                save_as_nifti(extracted_organ_data_volume, extracted_organ_data_volume_niigz_path)

                update_organs_envs(working_env, output_organs_env_name, \
                                   organ_name, 'data', zoom_key, \
                                   extracted_organ_data_volume_niigz_path, \
                                   bbox=pickle.dumps(extraction_bbox))

    return output_organs_env_name

def extract_organs_by_labels(input_organs_env_name, \
                             output_organs_env_name, \
                             working_env, \
                             zoom_key, \
                             save_label=True, \
                             bb_side_offset=10):
    #if True:
    organs_envs = working_env.get_organs_envs()

    if input_organs_env_name in organs_envs:
        input_organs = organs_envs[input_organs_env_name]

        for organ_name, organ_data_dict in input_organs.iteritems():
            organ_data_label_path = organ_data_dict['label'][zoom_key]
            organ_data_label = open_data(organ_data_label_path)

            organ_data_path = organ_data_dict['data'][zoom_key]
            organ_data = open_data(organ_data_path)

            extracted_organ_data_volume, extracted_organ_bbox = extract_largest_volume_by_label(organ_data, organ_data_label, bb_side_offset=bb_side_offset)

            extracted_organ_data_volume_niigz_path = \
                    working_env.get_new_volume_niigz_path(extracted_organ_data_volume.shape, \
                        ORGAN_DATA_TEMPLATE % (output_organs_env_name, organ_name))


            if True:
            #if not os.path.exists(extracted_organ_data_volume_niigz_path):
                save_as_nifti(extracted_organ_data_volume, extracted_organ_data_volume_niigz_path)

                update_organs_envs(working_env, output_organs_env_name, \
                                   organ_name, 'data', zoom_key, \
                                   extracted_organ_data_volume_niigz_path, \
                                   bbox=pickle.dumps(extracted_organ_bbox))

            if save_label:
                extracted_organ_data_label_volume = organ_data_label[extracted_organ_bbox]

                extracted_organ_data_label_volume_niigz_path = \
                        working_env.get_new_volume_niigz_path(extracted_organ_data_label_volume.shape, \
                            ORGAN_LABEL_TEMPLATE % (output_organs_env_name, organ_name))

                if extracted_organ_data_label_volume_niigz_path is not None:
                    if True:
                    #if not os.path.exists(extracted_organ_data_label_volume_niigz_path):
                        save_as_nifti(extracted_organ_data_label_volume, extracted_organ_data_label_volume_niigz_path)

                        update_organs_envs(working_env, output_organs_env_name, \
                                           organ_name, 'label', zoom_key, \
                                           extracted_organ_data_label_volume_niigz_path, \
                                           bbox=pickle.dumps(extracted_organ_bbox))

    return output_organs_env_name

def register_data(reference_data_path, target_data_path, reference_env, \
                  prefix_name, use_full_iters=True, use_syn=True, \
                  num_threads=24):
    registration_paths = reference_env.get_aligned_data_paths(prefix_name)

    output_name_reg = registration_paths['out_name']
    warped_path_reg = registration_paths['warped']
    iwarped_path_reg = registration_paths['iwarp']

    #if True:
    if not os.path.exists(warped_path_reg):
        align_fish_simple_ants(reference_env, target_data_path, reference_data_path, \
                               output_name_reg, warped_path_reg, iwarped_path_reg, \
                               reg_prefix=prefix_name, use_syn=use_syn, \
                               use_full_iters=use_full_iters, \
                               num_threads=num_threads)

        reference_env.save()
    else:
        print "Reference data (%s) is already registered to the extracted one " \
        "of target." % prefix_name

    return prefix_name

def transform_data(reference_data_label_path, target_data_path, \
                   registration_prefix_name, output_name, \
                   reference_env, target_env, \
                   reg_prefix=None):
    registration_paths = reference_env.get_aligned_data_paths(registration_prefix_name)

    affine_transformation_path = registration_paths['gen_affine']
    def_transformation_path = registration_paths['warp']

    target_data = open_data(target_data_path)

    transformation_output_path = \
            target_env.get_new_volume_niigz_path(target_data.shape, \
                    '_'.join([output_name, 'label']))

    #if True:
    if not os.path.exists(transformation_output_path):
        apply_transform_fish(target_env, target_data_path, \
                             affine_transformation_path, \
                             reference_data_label_path, \
                             transformation_output_path, \
                             def_transformation_path=def_transformation_path, \
                             reg_prefix=reg_prefix)
        reference_env.save()
        target_env.save()
    else:
        print "Extracted abdomenal part data is already transformed"

    return transformation_output_path

def transform_organs_data(reference_organs_env_name, target_data_path, \
                          zoom_key, registration_prefix_name, \
                          output_organs_env_name, reference_env, target_env):
    reference_organs_envs = reference_env.get_organs_envs()

    if reference_organs_env_name in reference_organs_envs:
        reference_input_organs = reference_organs_envs[reference_organs_env_name]

        for organ_name in reference_input_organs.keys():
            reference_organ_data_label_path = reference_input_organs[organ_name]['label'][zoom_key]

            target_organ_data_label_path = \
                    transform_data(reference_organ_data_label_path, \
                                   target_data_path, \
                                   registration_prefix_name, \
                                   ORGAN_LABEL_TEMPLATE % (output_organs_env_name, organ_name), \
                                   reference_env, \
                                   target_env)

            update_organs_envs(target_env, output_organs_env_name, organ_name, \
                               'label', zoom_key, target_organ_data_label_path)
            update_organs_envs(target_env, output_organs_env_name, organ_name, \
                               'data', zoom_key, target_data_path)

    else:
        raise ValueError('Reference or target evironment name is not correct')

    return output_organs_env_name

def complete_organs_to_full_volume(input_organs_env_name, output_organs_env_name, \
                                   abdomed_part_path, head_part_path, \
                                   abdomen_local_part_path, abdomen_local_part_bbox, \
                                   input_data_path, working_env, zoom_key, \
                                   separation_overlap=1, body_part='head'):
    organs_envs = working_env.get_organs_envs()

    if input_organs_env_name in organs_envs:
        input_organs = organs_envs[input_organs_env_name]

        for organ_name, organ_data_dict in input_organs.iteritems():
            organ_data_label_path = organ_data_dict['label'][zoom_key]

            extracted_organ_data_label_bbox = None
            if organ_data_dict['bbox'][zoom_key] is not None:
                extracted_organ_data_label_bbox = pickle.loads(organ_data_dict['bbox'][zoom_key])
            else:
                raise ValueError("Organ's %s bounding box is mot specified." % organ_name)

            completed_organ_data_label = \
                        complete_data_to_full_volume(abdomed_part_path, \
                                                     head_part_path, \
                                                     organ_data_label_path, \
                                                     extracted_organ_data_label_bbox, \
                                                     abdomen_local_part_path=abdomen_local_part_path, \
                                                     abdomen_local_part_bbox=abdomen_local_part_bbox, \
                                                     separation_overlap=separation_overlap, \
                                                     body_part=body_part)

            completed_organ_data_label_niigz_path = \
                        working_env.get_new_volume_niigz_path(completed_organ_data_label.shape, \
                                ORGAN_LABEL_TEMPLATE % (output_organs_env_name, organ_name))

            #if True:
            if not os.path.exists(completed_organ_data_label_niigz_path):
                save_as_nifti(completed_organ_data_label, completed_organ_data_label_niigz_path)

                update_organs_envs(working_env, output_organs_env_name, \
                                   organ_name, 'label', zoom_key, \
                                   completed_organ_data_label_niigz_path)

                update_organs_envs(working_env, output_organs_env_name, \
                                   organ_name, 'data', zoom_key, \
                                   input_data_path)
            else:
                print "The target fish's organ %s label is already completed to the input volume size." % organ_name

    return output_organs_env_name

def complete_fish_to_full_volume(abdomed_part_path, \
                                 head_part_path, \
                                 extracted_organ_volume_path, \
                                 extracted_organ_volume_bbox, \
                                 working_env, \
                                 abdomen_local_part_path=None, \
                                 abdomen_local_part_bbox=None, \
                                 separation_overlap=1, \
                                 body_part='head'):
    completed_data_label = \
            complete_data_to_full_volume(abdomed_part_path, \
                                         head_part_path, \
                                         extracted_organ_volume_path, \
                                         extracted_organ_volume_bbox, \
                                         abdomen_local_part_path=abdomen_local_part_path, \
                                         abdomen_local_part_bbox=abdomen_local_part_bbox, \
                                         separation_overlap=separation_overlap, \
                                         body_part=body_part)

    completed_data_label_niigz_path = \
            working_env.get_new_volume_niigz_path(completed_data_label.shape, \
                    "_".join(["completed"]))

    #if True:
    if not os.path.exists(completed_data_label_niigz_path):
        save_as_nifti(completed_data_label, completed_data_label_niigz_path)

    return completed_data_label_niigz_path

def upscale_organs_labels(input_organs_env_name, output_organs_env_name, \
                          input_data_original_path, working_env, zoom_key, \
                          new_zoom_key):
    organs_envs = working_env.get_organs_envs()

    if input_organs_env_name in organs_envs:
        input_organs = organs_envs[input_organs_env_name]

        for organ_name, organ_data_dict in input_organs.iteritems():
            organ_data_label_path = organ_data_dict['label'][zoom_key]

            input_data_original = open_data(input_data_original_path)

            upscaled_organ_data_label_niigz_path = \
                        working_env.get_new_volume_niigz_path(input_data_original.shape, \
                            ORGAN_LABEL_TEMPLATE % (output_organs_env_name, organ_name))

            #if True:
            if not os.path.exists(upscaled_organ_data_label_niigz_path):
                upscaled_organ_data_label = scale_to_size(input_data_original_path, \
                                                          organ_data_label_path, \
                                                          scale=2.0, \
                                                          order=0)
                save_as_nifti(upscaled_organ_data_label, upscaled_organ_data_label_niigz_path)

                update_organs_envs(working_env, output_organs_env_name, \
                                   organ_name, 'label', new_zoom_key, \
                                   upscaled_organ_data_label_niigz_path)

                update_organs_envs(working_env, output_organs_env_name, \
                                   organ_name, 'data', new_zoom_key, \
                                   input_data_original_path)

            else:
                print "The target fish's organ %s label is already upscaled to the input volume size." % organ_name

    return output_organs_env_name

def upscale_fish_label(input_data_original_path, \
                       zoomed_data_label_path, \
                       working_env, \
                       output_name):
    input_data_original = open_data(input_data_original_path)

    upscaled_data_label_niigz_path = \
            working_env.get_new_volume_niigz_path(input_data_original.shape, \
                    '_'.join([output_name, 'label']))

    #if True:
    if not os.path.exists(upscaled_data_label_niigz_path):
        upscaled_data_label = scale_to_size(input_data_original_path, \
                                            zoomed_data_label_path, \
                                            scale=2.0, \
                                            order=0)
        save_as_nifti(upscaled_data_label, upscaled_data_label_niigz_path)
    else:
        print "The data '%s' is already upscaled." % zoomed_data_label_path

    return upscaled_data_label_niigz_path

def segment_organs_by_labels(input_organs_env_name, \
                             output_organs_env_name, \
                             working_env, \
                             zoom_key):
    #if True:
    organs_envs = working_env.get_organs_envs()

    if input_organs_env_name in organs_envs:
        input_organs = organs_envs[input_organs_env_name]

        for organ_name, organ_data_dict in input_organs.iteritems():
            organ_data_label_path = organ_data_dict['label'][zoom_key]
            organ_data_label = open_data(organ_data_label_path)

            organ_data_path = organ_data_dict['data'][zoom_key]
            organ_data = open_data(organ_data_path)

            segmented_organ_data = organ_data * organ_data_label

            segmented_organ_data_niigz_path = \
                    working_env.get_new_volume_niigz_path(segmented_organ_data.shape, \
                        ORGAN_DATA_TEMPLATE % (output_organs_env_name, organ_name))


            #if True:
            if not os.path.exists(segmented_organ_data_niigz_path):
                save_as_nifti(segmented_organ_data, segmented_organ_data_niigz_path)

                update_organs_envs(working_env, output_organs_env_name, \
                                   organ_name, 'label', zoom_key, \
                                   organ_data_label_path)

                update_organs_envs(working_env, output_organs_env_name, \
                                   organ_name, 'data', zoom_key, \
                                   segmented_organ_data_niigz_path)

    return output_organs_env_name

def segment_data_by_labels(input_data_path, \
                           input_data_label_path, \
                           working_env, \
                           output_name):
    print 'input_data_path = %s' % input_data_path
    print 'input_data_label_path = %s' % input_data_label_path

    input_data = open_data(input_data_path)
    input_data_label = open_data(input_data_label_path)

    segmented_data = input_data * input_data_label

    segmented_data_niigz_path = \
            working_env.get_new_volume_niigz_path(segmented_data.shape, \
                    '_'.join([output_name, 'segmented']))

    #if True:
    if not os.path.exists(segmented_data_niigz_path):
        save_as_nifti(segmented_data, segmented_data_niigz_path)

    return segmented_data_niigz_path

def spine_segmentation(reference_env, \
                       target_env, \
                       min_area=20.0, \
                       min_circularity=0.5, \
                       com_dist_tolerance=5, \
                       zoom_level=2, \
                       min_zoom_level=2, \
                       organs_labels = ['heart']):
    reference_env.load()
    target_env.load()

    print "--Aligning and volumes' extraction"
    reference_data_results = initialize_env(reference_env, \
                                            zoom_level=zoom_level, \
                                            min_zoom_level=min_zoom_level, \
                                            organs_labels=organs_labels)

    target_data_results = initialize_env(target_env, \
                                         zoom_level=zoom_level, \
                                         min_zoom_level=min_zoom_level)

    reference_env.save()
    target_env.save()

    reference_original_input_data_path = reference_env.envs['extracted_input_data_path_niigz']
    reference_original_input_data_labels_path = reference_env.envs['extracted_input_data_labels_path_niigz']

    reference_input_data_path = reference_env.envs['zoomed_0p5_extracted_input_data_path_niigz']
    reference_input_data_labels_path = reference_env.envs['zoomed_0p5_extracted_input_data_labels_path_niigz']

    target_original_input_data_path = target_env.envs['extracted_input_data_path_niigz']
    target_input_data_path = target_env.envs['zoomed_0p5_extracted_input_data_path_niigz']

    print  "--Registration of the reference fish to the target one"
    parts_separation_prefix = \
                register_data(reference_input_data_path, \
                              target_input_data_path, \
                              reference_env, \
                              "full_volume_registration_approx", \
                              use_full_iters=False, \
                              use_syn=True)

    print "--Transforming brain and abdomen labels of the reference fish to the target's one"
    target_input_data_labels_approx_path = \
                transform_data(reference_input_data_labels_path, \
                               target_input_data_path, \
                               parts_separation_prefix, \
                               "full_volume_label_deformation", \
                               reference_env, \
                               target_env)

    print  "--Segment target spinal cord"
    target_filled_spine_label_niigz_path = \
                run_spine_segmentation(target_input_data_path, \
                                       target_input_data_labels_approx_path, \
                                       target_env, \
                                       "spine", \
                                       min_area=min_area, \
                                       min_circularity=min_circularity, \
                                       tolerance=com_dist_tolerance)

    print  "--Segment reference spinal cord"
    reference_filled_spine_label_niigz_path = \
                run_spine_segmentation(reference_input_data_path, \
                                       reference_input_data_labels_path, \
                                       reference_env, \
                                       "spine", \
                                       min_area=min_area, \
                                       min_circularity=min_circularity, \
                                       tolerance=com_dist_tolerance)

    print "--Segment target fish's spine data..."
    if target_filled_spine_label_niigz_path is not None:
        target_filled_spine_segmented_niigz_path = \
                        segment_data_by_labels(target_input_data_path, \
                                               target_filled_spine_label_niigz_path, \
                                               target_env, \
                                               "spine_data")

    print "--Segment reference fish's spine data..."
    if reference_filled_spine_label_niigz_path is not None:
        reference_filled_spine_segmented_niigz_path = \
                        segment_data_by_labels(reference_input_data_path, \
                                               reference_filled_spine_label_niigz_path, \
                                               reference_env, \
                                               "spine_data")

def brain_segmentation_ants_v2(reference_env, \
                               target_env, \
                               bb_side_offset = 2, \
                               separation_overlap = 2, \
                               zoom_level=2, \
                               min_zoom_level=2, \
                               organs_labels = []):

    reference_env.load()
    target_env.load()

    print "--Aligning and volumes' extraction"
    reference_data_results = initialize_env(reference_env, \
                                            zoom_level=zoom_level, \
                                            min_zoom_level=min_zoom_level, \
                                            organs_labels=organs_labels)

    target_data_results = initialize_env(target_env, \
                                         zoom_level=zoom_level, \
                                         min_zoom_level=min_zoom_level)

    reference_env.save()
    target_env.save()

    reference_original_input_data_path = reference_env.envs['extracted_input_data_path_niigz']
    reference_original_input_data_labels_path = reference_env.envs['extracted_input_data_labels_path_niigz']

    reference_input_data_path = reference_env.envs['zoomed_0p5_extracted_input_data_path_niigz']
    reference_input_data_labels_path = reference_env.envs['zoomed_0p5_extracted_input_data_labels_path_niigz']

    target_original_input_data_path = target_env.envs['extracted_input_data_path_niigz']
    target_input_data_path = target_env.envs['zoomed_0p5_extracted_input_data_path_niigz']

    # reference_original_input_data_path = reference_data_results['extracted']
    # reference_original_input_data_labels_path = reference_data_results['extracted_labels']
    #
    # reference_input_data_path = reference_data_results['scaled_0p5_extracted']
    # reference_input_data_labels_path = reference_data_results['scaled_0p5_extracted_labels']
    #
    # target_original_input_data_path = target_data_results['extracted']
    # target_input_data_path = target_data_results['scaled_0p5_extracted']

    print reference_original_input_data_path
    print reference_original_input_data_labels_path
    print reference_input_data_path
    print reference_input_data_labels_path
    print target_original_input_data_path
    print target_input_data_path

    print  "--Registration of the reference fish to the target one"
    parts_separation_prefix = \
                register_data(reference_input_data_path, \
                              target_input_data_path, \
                              reference_env, \
                              "full_volume_registration_approx", \
                              use_full_iters=False, \
                              use_syn=True)

    print "--Transforming brain and abdomen labels of the reference fish to the target's one"
    target_input_data_labels_approx_path = \
                transform_data(reference_input_data_labels_path, \
                               target_input_data_path, \
                               parts_separation_prefix, \
                               "full_volume_label_deformation", \
                               reference_env, \
                               target_env)

    print "--Fish separation (reference image)..."
    reference_abdomen_data_niigz_path, \
    reference_abdomen_data_label_niigz_path, \
    reference_head_data_niigz_path, \
    reference_head_data_label_niigz_path = \
                split_fish_body(reference_input_data_path, \
                                reference_input_data_labels_path, \
                                "separated", \
                                reference_env, \
                                separation_overlap=separation_overlap, \
                                zoom_key='zoomed')

    print "--Fish separation (target image)..."
    target_abdomen_data_niigz_path, \
    target_abdomen_data_label_niigz_path, \
    target_head_data_niigz_path, \
    target_head_data_label_niigz_path = \
                split_fish_body(target_input_data_path, \
                                target_input_data_labels_approx_path, \
                                "separated", \
                                target_env, \
                                separation_overlap=separation_overlap, \
                                zoom_key='zoomed')

    print "--Register reference fish's head to the target's one..."
    head_registration_prefix = \
                register_data(reference_head_data_niigz_path, \
                              target_head_data_niigz_path, \
                              reference_env, \
                              "head_registration", \
                              use_full_iters=False, \
                              use_syn=True)

    print "--Transfrom labels of reference fish's head into the target's one..."
    target_head_data_label_approx_path = \
                transform_data(reference_head_data_label_niigz_path, \
                               target_head_data_niigz_path, \
                               head_registration_prefix, \
                               "head_label_deformation", \
                               reference_env, \
                               target_env)

    print "--Extract the reference fish's brain using labels..."
    reference_extracted_brain_niigz_path, \
    reference_extracted_brain_label_niigz_path, \
    reference_extracted_brain_bbox = \
                extract_largest_label(reference_head_data_niigz_path, \
                                      reference_head_data_label_niigz_path, \
                                      "extracted_brain", \
                                      reference_env, \
                                      bb_side_offset=bb_side_offset)

    print "--Extract the target fish's brain using transformed head labels..."
    target_extracted_brain_niigz_path, \
    target_extracted_brain_label_niigz_path, \
    target_extracted_brain_bbox = \
                extract_largest_label(target_head_data_niigz_path, \
                                      target_head_data_label_approx_path, \
                                      "extracted_brain_approx_location", \
                                      target_env, \
                                      bb_side_offset=bb_side_offset)

    print "--Register the reference fish's brain to the target's one..."
    brain_registration_prefix = \
                register_data(reference_extracted_brain_niigz_path, \
                              target_extracted_brain_niigz_path, \
                              reference_env, \
                              "brain_registration", \
                              use_full_iters=True, \
                              use_syn=True)

    print "--Transform the reference fish's brain labels into the target's one..."
    target_extracted_brain_label_approx_path = \
                transform_data(reference_extracted_brain_label_niigz_path, \
                               target_extracted_brain_niigz_path, \
                               brain_registration_prefix, \
                               "extracted_brain", \
                               reference_env, \
                               target_env)

    print "--Complete the target fish's brain labels to full volume..."
    target_completed_data_label_path = \
                complete_fish_to_full_volume(target_abdomen_data_label_niigz_path, \
                                             target_head_data_label_niigz_path, \
                                             target_extracted_brain_label_approx_path, \
                                             target_extracted_brain_bbox, \
                                             target_env, \
                                             separation_overlap=separation_overlap, \
                                             body_part='head')

    print "--Upscale the initial aligned completed target fish's brain labels to the input volume size..."
    target_upscaled_completed_data_label_path = \
                upscale_fish_label(target_original_input_data_path, \
                                   target_completed_data_label_path, \
                                   target_env, \
                                   "upscaled_data")

    print "--Extract the target fish's brain labels from the upscaled initial volume..."
    target_extracted_upscaled_brain_niigz_path, \
    target_extracted_upscaled_brain_label_niigz_path, \
    target_extracted_upscaled_brain_bbox = \
                extract_largest_label(target_original_input_data_path, \
                                      target_upscaled_completed_data_label_path, \
                                      "extracted_upscaled_brain", \
                                      target_env, \
                                      bb_side_offset=bb_side_offset)

    print "--Extract the reference fish's brain labels from the original volume..."
    reference_extracted_original_brain_niigz_path, \
    reference_extracted_original_brain_label_niigz_path, \
    reference_extracted_original_brain_bbox = \
                extract_largest_label_by_name(reference_original_input_data_path, \
                                              reference_original_input_data_labels_path, \
                                              "extracted_original_brain", \
                                              reference_env, \
                                              bb_side_offset=bb_side_offset, \
                                              label_name='brain')

    print "--Segment upscaled target fish's brain data..."
    target_extracted_upscaled_segmented_brain_niigz_path = \
                        segment_data_by_labels(target_extracted_upscaled_brain_niigz_path, \
                                               target_extracted_upscaled_brain_label_niigz_path, \
                                               target_env, \
                                               "brain")

    print "--Segment original reference fish's brain data..."
    reference_extracted_original_segmented_brain_niigz_path = \
                        segment_data_by_labels(reference_extracted_original_brain_niigz_path, \
                                               reference_extracted_original_brain_label_niigz_path, \
                                               reference_env, \
                                               "brain")

@timing
def gather_volume_statistics(reference_env, \
                             target_env, \
                             zoom_level=2, \
                             min_zoom_level=2, \
                             organs_labels = []):

    print "##################################################################################"
    print "######################### Gathering statistics has started #########################"
    print "##################################################################################"

    reference_env.load()
    target_env.load()

    print "--Aligning and volumes' extraction"
    reference_data_results = initialize_env(reference_env, \
                                            zoom_level=zoom_level, \
                                            min_zoom_level=min_zoom_level, \
                                            organs_labels=organs_labels)

    target_data_results = initialize_env(target_env, \
                                         zoom_level=zoom_level, \
                                         min_zoom_level=min_zoom_level)

    reference_env.save()
    target_env.save()

    generate_stats(reference_env)
    generate_stats(target_env)

@timing
def organs_segmentation_ants(reference_env, \
                             target_env, \
                             bb_side_offset=2, \
                             separation_overlap=5, \
                             zoom_level=2, \
                             min_zoom_level=2, \
                             organs_labels = ['heart']):

    print "##################################################################################"
    print "######################### Organ segmentation has started #########################"
    print "##################################################################################"

    reference_env.load()
    target_env.load()

    print "--Aligning and volumes' extraction"
    reference_data_results = initialize_env(reference_env, \
                                            zoom_level=zoom_level, \
                                            min_zoom_level=min_zoom_level, \
                                            organs_labels=organs_labels)

    target_data_results = initialize_env(target_env, \
                                         zoom_level=zoom_level, \
                                         min_zoom_level=min_zoom_level)

    reference_env.save()
    target_env.save()

    reference_input_aligned_data_path = reference_env.envs['zoomed_0p5_extracted_input_data_path_niigz']
    reference_input_aligned_data_label_path = reference_env.envs['zoomed_0p5_extracted_input_data_labels_path_niigz']

    target_input_aligned_data_path = target_env.envs['zoomed_0p5_extracted_input_data_path_niigz']

    print  "--Registration of the reference fish to the target one"
    parts_separation_prefix = \
                register_data(reference_input_aligned_data_path, \
                              target_input_aligned_data_path, \
                              reference_env, \
                              "full_volume_registration_approx", \
                              use_full_iters=False, \
                              use_syn=True)

    print "--Transforming brain and abdomen labels of the reference fish to the target's one"
    target_input_data_labels_approx_path = \
                transform_data(reference_input_aligned_data_label_path, \
                               target_input_aligned_data_path, \
                               parts_separation_prefix, \
                               "full_volume_label_deformation", \
                               reference_env, \
                               target_env)

    print "--Fish separation (reference image)..."
    reference_abdomen_data_niigz_path, \
    reference_abdomen_data_label_niigz_path, \
    reference_head_data_niigz_path, \
    reference_head_data_label_niigz_path = \
                split_fish_body(reference_input_aligned_data_path, \
                                reference_input_aligned_data_label_path, \
                                "separated", \
                                reference_env, \
                                separation_overlap=separation_overlap, \
                                zoom_key='zoomed')

    print "--Fish separation (target image)..."
    target_abdomen_data_niigz_path, \
    target_abdomen_data_label_niigz_path, \
    target_head_data_niigz_path, \
    target_head_data_label_niigz_path = \
                split_fish_body(target_input_aligned_data_path, \
                                target_input_data_labels_approx_path, \
                                "separated", \
                                target_env, \
                                separation_overlap=separation_overlap, \
                                zoom_key='zoomed')

    print "--Fish-organs separation (reference image)..."
    reference_abdomen_local_separation_env = \
                        split_fish_organs("original_organs", \
                                          "reference_abdomen_local_separation", \
                                          reference_input_aligned_data_label_path, \
                                          reference_input_aligned_data_path, \
                                          reference_env, \
                                          ZOOM_KEY, \
                                          overlap=separation_overlap)

    zoomed_abdomen_extraction_name = 'zoomed_0p5_extracted_abdomen_part_nondeformed'

    print "--Extract the reference fish's abdomenal part using transformed abdomenal part labels..."
    reference_abdomen_local_data_niigz_path, \
    reference_abdomen_local_data_labels_niigz_path, \
    reference_abdomen_local_data_bbox = \
                    extract_largest_label(reference_abdomen_data_niigz_path, \
                                          reference_abdomen_data_label_niigz_path, \
                                          zoomed_abdomen_extraction_name, \
                                          reference_env, \
                                          bb_side_offset=bb_side_offset)

    print "--Extract the target fish's abdomenal part using transformed abdomenal part labels..."
    target_abdomen_local_data_niigz_path, \
    _, \
    target_abdomen_local_data_bbox = \
                    extract_largest_label(target_abdomen_data_niigz_path, \
                                          target_abdomen_data_label_niigz_path, \
                                          zoomed_abdomen_extraction_name, target_env, \
                                          bb_side_offset=bb_side_offset)



    print "--Extract the reference fish's abdomenal part organs using transformed abdomenal part labels..."
    reference_abdomen_local_organs_env = \
                    extract_organs_by_bbox(reference_abdomen_local_separation_env, \
                                           "reference_abdomen_local_organs", \
                                           reference_abdomen_local_data_bbox, \
                                           reference_env, \
                                           ZOOM_KEY)

    print "--Register reference fish's extracted abdomenal part to the target's one..."
    reference_abdomen_local_registration_prefix_name = \
                              register_data(reference_abdomen_local_data_niigz_path, \
                                            target_abdomen_local_data_niigz_path, \
                                            reference_env, \
                                            'reference_abdomen_local_registration', \
                                            use_full_iters=True, num_threads=8)

    print "--Transfrom labels of reference fish's extracted abdomenal part into the target's one..."
    target_abdomen_local_data_labels_niigz_path = \
                              transform_data(reference_abdomen_local_data_labels_niigz_path, \
                                             target_abdomen_local_data_niigz_path, \
                                             reference_abdomen_local_registration_prefix_name, \
                                             zoomed_abdomen_extraction_name, \
                                             reference_env, \
                                             target_env)

    print "--Transfrom labels of reference fish's organs extracted abdomenal part into the target's one..."
    target_abdomen_local_organs_approx_env = \
                              transform_organs_data(reference_abdomen_local_organs_env, \
                                                    target_abdomen_local_data_niigz_path, \
                                                    ZOOM_KEY, \
                                                    reference_abdomen_local_registration_prefix_name, \
                                                    "target_abdomen_local_organs_labels_approx", \
                                                    reference_env, \
                                                    target_env)

    print "--Extract the target fish's organs labels..."
    target_extracted_abdomen_organs_env = \
                              extract_organs_by_labels(target_abdomen_local_organs_approx_env, \
                                                       "target_extracted_abdomen_organs", \
                                                       target_env, \
                                                       ZOOM_KEY, \
                                                       bb_side_offset=bb_side_offset)

    print "--Extract the reference fish's organs labels..."
    reference_extracted_abdomen_organs_env = \
                              extract_organs_by_labels(reference_abdomen_local_organs_env, \
                                                       "reference_extracted_abdomen_organs", \
                                                       reference_env, \
                                                       ZOOM_KEY, \
                                                       bb_side_offset=bb_side_offset)

    print "--Complete the target fish's organs labels to the full volume size..."
    target_completed_organs_env = \
                   complete_organs_to_full_volume(target_extracted_abdomen_organs_env, \
                                                  "target_completed_organs", \
                                                  target_abdomen_data_niigz_path, \
                                                  target_head_data_niigz_path, \
                                                  target_abdomen_local_data_niigz_path,\
                                                  target_abdomen_local_data_bbox, \
                                                  target_input_aligned_data_path, \
                                                  target_env, \
                                                  ZOOM_KEY, \
                                                  separation_overlap=separation_overlap, \
                                                  body_part='abdomen')

    print "--Upscale the completed target fish's organs labels to the initial volume size..."
    target_original_input_aligned_data_path = target_env.envs['extracted_input_data_path_niigz']
    target_upscaled_completed_organs_env = \
                   upscale_organs_labels(target_completed_organs_env, \
                                         "target_upscaled_completed_organs", \
                                         target_original_input_aligned_data_path, \
                                         target_env, \
                                         ZOOM_KEY, \
                                         NORMAL_KEY)

    print "--Extract the target fish's organs from the upscaled volume..."
    target_extracted_upscaled_abdomen_organs_env = \
                   extract_organs_by_labels(target_upscaled_completed_organs_env, \
                                            "target_extracted_upscaled_abdomen_organs", \
                                            target_env, \
                                            NORMAL_KEY, \
                                            bb_side_offset=bb_side_offset)

    print "--Extract the refernce fish's organs from the original volume..."
    reference_extracted_original_abdomen_organs_env = \
                   extract_organs_by_labels("original_organs", \
                                            "reference_extracted_original_abdomen_organs", \
                                            reference_env, \
                                            NORMAL_KEY, \
                                            bb_side_offset=bb_side_offset)

    print "--Segment upscaled target fish's organs data..."
    target_upscaled_segmented_organs_env = \
                   segment_organs_by_labels(target_extracted_upscaled_abdomen_organs_env, \
                                            "target_upscaled_segmented_organs", \
                                            target_env, \
                                            NORMAL_KEY)

    print "--Segment original reference fish's organs data..."
    reference_original_segmented_organs_env = \
                   segment_organs_by_labels(reference_extracted_original_abdomen_organs_env, \
                                            "reference_original_segmented_organs", \
                                            reference_env, \
                                            NORMAL_KEY)

@timing
def scaled_abdomen_based_heart_segmentation_ants(reference_data_env, target_data_env):
    # bb_side_offset = 10
    # reference_data_env.load()
    # target_data_env.load()
    #
    # zoomed_abdomen_extraction_name = 'zoomed_0p5_extracted_abdomen_part_nondeformed'
    #
    # print "--Extract the target fish's abdomenal part using transformed
    # abdomenal part labels..."
    # abdomen_part_data_volume_target_niigz_path, \
    # _, \
    # abdomen_part_data_volume_target_bbox = \
    #     extract_largest_label(target_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz'], \
    #                           target_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz'], \
    #                           zoomed_abdomen_extraction_name, target_data_env)
    #
    # print "--Extract the reference fish's abdomenal part using transformed abdomenal part labels..."
    # abdomen_part_data_volume_reference_niigz_path, \
    # abdomen_part_data_volume_labels_reference_niigz_path, \
    # abdomen_part_data_volume_reference_bbox = \
    #     extract_largest_label(reference_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz'], \
    #                           reference_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz'], \
    #                           zoomed_abdomen_extraction_name, reference_data_env)

    # print "--Extract the reference fish's abdomenal part organs using transformed abdomenal part labels..."
    # abdomen_separated_organs_labels_dict = reference_data_env.get_abdomen_separated_organs_labels()
    # extracted_abdomen_part_separated_organs_labels_dict = reference_data_env.get_extracted_abdomen_part_separated_organs_labels()
    #
    # output_data_labels_dict, _ = \
    #         extract_organs_by_largest_labels(abdomen_separated_organs_labels_dict, \
    #                                          reference_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz'], \
    #                                          extracted_abdomen_part_separated_organs_labels_dict, \
    #                                          zoomed_abdomen_extraction_name, \
    #                                          reference_data_env, \
    #                                          'zoomed', \
    #                                          bb_side_offset=bb_side_offset)
    #
    # reference_data_env.set_extracted_abdomen_part_separated_organs_labels(output_data_labels_dict)
    # reference_data_env.save()

    # print "--Register reference fish's extracted abdomenal part to the target's one..."
    # #Register reference abdomen to target one
    # abdomen_part_registration_prefix_name = \
    #                 register_data(abdomen_part_data_volume_reference_niigz_path, \
    #                               abdomen_part_data_volume_target_niigz_path, \
    #                               reference_data_env, \
    #                               'extracted_abdomen_part_registration', \
    #                               use_full_iters=True)

    # print "--Transfrom labels of reference fish's extracted abdomenal part into the target's one..."
    # abdomen_part_data_volume_labels_target_niigz_path = \
    #         transform_data(abdomen_part_data_volume_labels_reference_niigz_path, \
    #                        abdomen_part_data_volume_target_niigz_path, \
    #                        abdomen_part_registration_prefix_name, \
    #                        zoomed_abdomen_extraction_name, \
    #                        reference_data_env, \
    #                        target_data_env)

    # print "--Transfrom labels of reference fish's organs extracted abdomenal part into the target's one..."
    # reference_extracted_abdomen_part_separated_organs_labels_dict = \
    #             reference_data_env.get_extracted_abdomen_part_separated_organs_labels()
    #
    # target_extracted_abdomen_part_separated_organs_labels_dict = \
    #             target_data_env.get_extracted_abdomen_part_separated_organs_labels()
    #
    # output_data_dict = \
    #     transform_organs_data(reference_extracted_abdomen_part_separated_organs_labels_dict, \
    #                           target_extracted_abdomen_part_separated_organs_labels_dict, \
    #                           'zoomed', \
    #                           abdomen_part_data_volume_target_niigz_path, \
    #                           abdomen_part_registration_prefix_name, \
    #                           zoomed_abdomen_extraction_name, \
    #                           reference_data_env, \
    #                           target_data_env)
    #
    # target_data_env.set_extracted_abdomen_part_separated_organs_labels(output_data_dict)
    # target_data_env.save()
    #
    # zoomed_abdomed_organ_extracted_name  = 'zoomed_0p5_abdomen_part_extracted_roi'

    print "--Extract the target fish's organs labels from the initial volume..."
    target_extracted_organs_dict = target_data_env.get_extracted_roi_abdomen_part_separated_organs()
    target_extracted_organs_labels_dict = target_data_env.get_extracted_roi_abdomen_part_separated_organs_labels()
    target_organs_labels_dict = target_data_env.get_extracted_abdomen_part_separated_organs_labels()

    target_output_data_labels_dict, target_output_data_dict = \
            extract_organs_by_largest_labels(target_organs_labels_dict, \
                                             abdomen_part_data_volume_target_niigz_path, \
                                             target_extracted_organs_labels_dict, \
                                             zoomed_abdomed_organ_extracted_name, \
                                             target_data_env, \
                                             'zoomed', \
                                             bb_side_offset=bb_side_offset, \
                                             output_data_dict=target_extracted_organs_dict)

    target_data_env.set_extracted_abdomen_part_separated_organs(target_output_data_dict)
    target_data_env.set_extracted_abdomen_part_separated_organs_labels(target_output_data_labels_dict)
    target_data_env.save()

    print "--Extract the reference fish's organs labels from the initial volume..."
    reference_extracted_organs_dict = reference_data_env.get_extracted_roi_abdomen_part_separated_organs()
    reference_extracted_organs_labels_dict = reference_data_env.get_extracted_roi_abdomen_part_separated_organs_labels()
    reference_organs_labels_dict = reference_data_env.get_extracted_abdomen_part_separated_organs_labels()

    reference_output_data_labels_dict, reference_output_data_dict = \
            extract_organs_by_largest_labels(reference_organs_labels_dict, \
                                             abdomen_part_data_volume_reference_niigz_path, \
                                             reference_extracted_organs_labels_dict, \
                                             zoomed_abdomed_organ_extracted_name, \
                                             reference_data_env, \
                                             'zoomed', \
                                             bb_side_offset=bb_side_offset, \
                                             output_data_dict=reference_extracted_organs_dict)

    reference_data_env.set_extracted_abdomen_part_separated_organs(reference_output_data_dict)
    reference_data_env.set_extracted_abdomen_part_separated_organs_labels(reference_output_data_labels_dict)
    reference_data_env.save()

@timing
def abdomen_based_heart_segmentation_ants(reference_data_env, target_data_env):
    bb_side_offset = 10
    reference_data_env.load()
    target_data_env.load()

    print "--Extract the target fish's abdomenal part using transformed abdomenal part labels..."
    # Extract target abdomenal part volume
    abdomen_part_label_target = open_data(target_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz'])
    abdomen_part_data_target = open_data(target_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz'])
    abdomen_part_data_volume_target, abdomen_part_data_volume_target_bbox = \
            extract_largest_volume_by_label(abdomen_part_data_target, \
                                            abdomen_part_label_target, \
                                            bb_side_offset=bb_side_offset)
    abdomen_part_data_volume_target_niigz_path = \
        target_data_env.get_new_volume_niigz_path(abdomen_part_data_volume_target.shape, \
                                'zoomed_0p5_extracted_abdomen_part_nondeformed')

    print abdomen_part_data_volume_target_niigz_path

    if True:
    #if not os.path.exists(abdomen_part_data_volume_target_niigz_path):
        save_as_nifti(abdomen_part_data_volume_target, abdomen_part_data_volume_target_niigz_path)

    reference_data_env.save()
    target_data_env.save()

    print "--Extract the reference fish's abdomenal part using transformed abdomenal part labels..."
    # Extract reference brain volume
    abdomen_part_label_reference = open_data(reference_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz'])
    abdomen_part_data_reference = open_data(reference_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz'])
    abdomen_part_data_volume_reference, abdomen_part_reference_bbox = \
                        extract_largest_volume_by_label(abdomen_part_data_reference, \
                                                        abdomen_part_label_reference, \
                                                        bb_side_offset=bb_side_offset)
    abdomen_part_data_labels_volume_reference = abdomen_part_label_reference[abdomen_part_reference_bbox]

    abdomen_part_data_volume_reference_niigz_path = \
            reference_data_env.get_new_volume_niigz_path(abdomen_part_data_volume_reference.shape, \
                'zoomed_0p5_extracted_abdomen_part_nondeformed')
    abdomen_part_data_labels_volume_reference_niigz_path = \
            reference_data_env.get_new_volume_niigz_path(abdomen_part_data_labels_volume_reference.shape, \
                'zoomed_0p5_extracted_abdomen_part_labels_nondeformed')

    print abdomen_part_data_volume_reference_niigz_path
    print abdomen_part_data_labels_volume_reference_niigz_path

    if True:
    #if not os.path.exists(abdomen_part_data_volume_reference_niigz_path):
        save_as_nifti(abdomen_part_data_volume_reference, abdomen_part_data_volume_reference_niigz_path)

    if True:
    #if not os.path.exists(abdomen_part_data_labels_volume_reference_niigz_path):
        save_as_nifti(abdomen_part_data_labels_volume_reference, abdomen_part_data_labels_volume_reference_niigz_path)

    reference_data_env.save()
    target_data_env.save()

    print "--Extract the reference fish's abdomenal part organs using transformed abdomenal part labels..."
    extracted_abdomen_part_separated_organs_labels_dict = reference_data_env.get_extracted_abdomen_part_separated_organs_labels()

    print 'extracted_abdomen_part_separated_organs_labels_dict = %s' % str(reference_data_env.get_extracted_abdomen_part_separated_organs_labels())

    if True:
    #if not extracted_abdomen_part_separated_organs_labels_dict:
        abdomen_separated_organs_labels_dict = reference_data_env.get_abdomen_separated_organs_labels()

        for organ_name, organ_labels in abdomen_separated_organs_labels_dict.iteritems():
            zoomed_abdomen_separated_organ_label_path = organ_labels['zoomed']
            zoomed_abdomen_separated_organ_label_data = open_data(zoomed_abdomen_separated_organ_label_path)

            abdomen_part_data_organ_label_volume_reference = zoomed_abdomen_separated_organ_label_data[abdomen_part_reference_bbox]

            abdomen_part_data_organ_label_volume_reference_niigz_path = \
                    reference_data_env.get_new_volume_niigz_path(abdomen_part_data_organ_label_volume_reference.shape, \
                    'zoomed_0p5_extracted_abdomen_part_organ_%s_labels' % organ_name)
            if True:
            #if not os.path.exists(abdomen_part_data_organ_label_volume_reference_niigz_path):
                save_as_nifti(abdomen_part_data_organ_label_volume_reference, abdomen_part_data_organ_label_volume_reference_niigz_path)

            extracted_abdomen_part_separated_organs_labels_dict[organ_name] = { 'normal': None, 'zoomed': abdomen_part_data_organ_label_volume_reference_niigz_path }

    reference_data_env.set_extracted_abdomen_part_separated_organs_labels(extracted_abdomen_part_separated_organs_labels_dict)
    reference_data_env.save()

    print 'extracted_abdomen_part_separated_organs_labels_dict = %s' % str(reference_data_env.get_extracted_abdomen_part_separated_organs_labels())


    print "--Register reference fish's extracted abdomenal part to the target's one..."
    #Register reference abdomen to target one
    ants_prefix_ext_abdomen_reg = 'extracted_abdomen_part_registration'
    ants_ext_abdomen_reg_paths = reference_data_env.get_aligned_data_paths(ants_prefix_ext_abdomen_reg)

    reference_image_path_ext_abdomen_reg = abdomen_part_data_volume_reference_niigz_path
    target_image_path_ext_abdomen_reg = abdomen_part_data_volume_target_niigz_path

    output_name_ext_abdomen_reg = ants_ext_abdomen_reg_paths['out_name']
    warped_path_ext_abdomen_reg = ants_ext_abdomen_reg_paths['warped']
    iwarped_path_ext_abdomen_reg = ants_ext_abdomen_reg_paths['iwarp']

    if True:
    #if not os.path.exists(warped_path_ext_abdomen_reg):
        align_fish_simple_ants(reference_data_env, target_image_path_ext_abdomen_reg, \
                               reference_image_path_ext_abdomen_reg, output_name_ext_abdomen_reg, \
                               warped_path_ext_abdomen_reg, iwarped_path_ext_abdomen_reg, \
                               reg_prefix=ants_prefix_ext_abdomen_reg, use_syn=True, \
                               use_full_iters=True)

        reference_data_env.save()
        target_data_env.save()
    else:
        print "Extracted abdomenal part of the reference data is already registered to the extracted one of target"

    print "--Transfrom labels of reference fish's extracted abdomenal part into the target's one..."
    # Transforming labels of abdomenal part of reference fish to the abdomenal part of target one
    ref_image_path_ext_htr = target_image_path_ext_abdomen_reg
    transformation_path_ext_htr = ants_ext_abdomen_reg_paths['gen_affine']
    def_transformation_path_ext_htr = ants_ext_abdomen_reg_paths['warp']
    labels_image_path_ext_htr = abdomen_part_data_labels_volume_reference_niigz_path

    test_data_ext_htr = open_data(ref_image_path_ext_htr)
    transformation_output_ext_htr = \
            target_data_env.get_new_volume_niigz_path(test_data_ext_htr.shape, \
                        'zoomed_0p5_extracted_abdomen_part_labels_nondeformed')
    reg_prefix_ext_htr = 'extracted_abdomen_label_deforming'

    if True:
    #if not os.path.exists(transformation_output_ext_htr):
        apply_transform_fish(target_data_env, ref_image_path_ext_htr, \
                             transformation_path_ext_htr, labels_image_path_ext_htr, \
                             transformation_output_ext_htr, \
                             def_transformation_path=def_transformation_path_ext_htr, \
                             reg_prefix=reg_prefix_ext_htr)
        reference_data_env.save()
        target_data_env.save()
    else:
        print "Extracted abdomenal part data is already transformed"

    print "--Transfrom labels of reference fish's organs extracted abdomenal part into the target's one..."

    reference_extracted_abdomen_part_separated_organs_labels_dict = \
                reference_data_env.get_extracted_abdomen_part_separated_organs_labels()
    print 'reference_extracted_abdomen_part_separated_organs_labels_dict = %s'% \
            str(reference_extracted_abdomen_part_separated_organs_labels_dict)

    print str(target_data_env.envs)
    target_extracted_abdomen_part_separated_organs_labels_dict = \
            target_data_env.get_extracted_abdomen_part_separated_organs_labels()
    print 'target_extracted_abdomen_part_separated_organs_labels_dict = %s' % \
            str(target_extracted_abdomen_part_separated_organs_labels_dict)

    if True:
    #if not target_extracted_abdomen_part_separated_organs_labels_dict:
        for organ_name, organ_labels in reference_extracted_abdomen_part_separated_organs_labels_dict.iteritems():
            print '----Extracting abdomenal organ %s' % organ_name

            organ_ref_image_path_ext_htr = target_image_path_ext_abdomen_reg
            organ_transformation_path_ext_htr = ants_ext_abdomen_reg_paths['gen_affine']
            organ_def_transformation_path_ext_htr = ants_ext_abdomen_reg_paths['warp']
            organ_labels_image_path_ext_htr = organ_labels['zoomed']

            organ_test_data_ext_htr = open_data(organ_ref_image_path_ext_htr)
            organ_transformation_output_ext_htr = \
                    target_data_env.get_new_volume_niigz_path(organ_test_data_ext_htr.shape, \
                        'zoomed_0p5_extracted_abdomen_part_organ_%s_labels' % organ_name)

            organ_reg_prefix_ext_htr = 'extracted_abdomen_organ_%s_label_deforming' % organ_name

            if True:
            #if not os.path.exists(organ_transformation_output_ext_htr):
                apply_transform_fish(target_data_env, organ_ref_image_path_ext_htr, \
                                     organ_transformation_path_ext_htr, organ_labels_image_path_ext_htr, \
                                     organ_transformation_output_ext_htr, \
                                     def_transformation_path=organ_def_transformation_path_ext_htr, \
                                     reg_prefix=organ_reg_prefix_ext_htr)

                target_extracted_abdomen_part_separated_organs_labels_dict[organ_name] = \
                        { 'normal': None, 'zoomed': organ_transformation_output_ext_htr }
                target_data_env.set_extracted_abdomen_part_separated_organs_labels(target_extracted_abdomen_part_separated_organs_labels_dict)

                reference_data_env.save()
                target_data_env.save()
            else:
                print "Extracted abdomenal organ '%s' part data is already transformed" % organ_name

    reference_data_env.save()
    target_data_env.save()

    print 'reference_extracted_abdomen_part_separated_organs_labels_dict = %s' % str(reference_extracted_abdomen_part_separated_organs_labels_dict)
    print 'target_extracted_abdomen_part_separated_organs_labels_dict = %s' % str(target_extracted_abdomen_part_separated_organs_labels_dict)
    '''
    print "--Upscale the target fish's organs labels to the initial volume size..."
    target_organs_labels_dict = target_data_env.get_organs_labels()

    if True:
    #if target_organs_labels_dict:
        for organ_name, organ_labels in target_extracted_abdomen_part_separated_organs_labels_dict.iteritems():
            zoomed_extracted_organ_label_path = organ_labels['zoomed']

            zoomed_organ_labels = complete_brain_to_full_volume(abdomen_data_part_labels_target_niigz_path, \
                                                                head_data_part_labels_target_niigz_path, \
                                                                transformation_output_brain_tr, \
                                                                brain_data_volume_target_bbox, \
                                                                separation_overlap=20)

            original_aligned_data_path = target_data_env.get_input_align_data_path()
            original_aligned_data = open_data(original_aligned_data_path)

            upscaled_organ_label_niigz_path = target_data_env.get_new_volume_niigz_path(original_aligned_data.shape, 'extracted_organ_%s_labels' % organ_name, bits=8)

            if True:
            #if not os.path.exists(upscaled_organ_label_niigz_path):
                upscaled_organ_label_data = scale_to_size(original_aligned_data_path, \
                                                          zoomed_organ_label_path, \
                                                          scale=2.0, \
                                                          order=0)
                save_as_nifti(upscaled_organ_label_data, upscaled_organ_label_niigz_path)
                target_organs_labels_dict[organ_name]['normal'] = upscaled_organ_label_niigz_path

        target_data_env.set_organs_labels(target_organs_labels_dict)

        reference_data_env.save()
        target_data_env.save()
    else:
        print "The target fish's organs labels are already upscaled to the input volume size."

    print "--Complete the target fish's organs labels to full volume..."
    test_data_complete_vol_brain_target = open_data(target_image_path_sep)
    complete_vol_target_brain_labels_niigz_path = target_data_env.get_new_volume_niigz_path(test_data_complete_vol_brain_target.shape, 'zoomed_0p5_complete_volume_brain_labels', bits=8)

    # if True:
    if not os.path.exists(complete_vol_target_brain_labels_niigz_path):
        complete_vol_target_brain_labels = complete_brain_to_full_volume(abdomen_data_part_labels_target_niigz_path, \
                                                                         head_data_part_labels_target_niigz_path, \
                                                                         transformation_output_brain_tr, \
                                                                         brain_data_volume_target_bbox, \
                                                                         separation_overlap=20);
        save_as_nifti(complete_vol_target_brain_labels, complete_vol_target_brain_labels_niigz_path)
    else:
        print "The brain labels of the target data (target fish) is already transformed."
    '''
    print "--Extract the target fish's organs labels from the initial volume..."
    target_extracted_organs_dict = target_data_env.get_extracted_roi_abdomen_part_separated_organs()
    target_extracted_organs_labels_dict = target_data_env.get_extracted_roi_abdomen_part_separated_organs_labels()
    target_organs_labels_dict = target_data_env.get_extracted_abdomen_part_separated_organs_labels()

    bb_side_offset = 10

    if True:
    #if (not target_extracted_organs_labels_dict) and (not target_extracted_organs_dict):
        for organ_name, organ_labels in target_organs_labels_dict.iteritems():
            normal_organ_label_path = organ_labels['zoomed']
            normal_organ_label_data = open_data(normal_organ_label_path)

            original_aligned_data_path = target_image_path_ext_abdomen_reg
            original_aligned_data = open_data(original_aligned_data_path)

            extracted_organ, extracted_organ_bbox = extract_largest_volume_by_label(original_aligned_data, normal_organ_label_data, bb_side_offset=bb_side_offset)
            extracted_organ_label = normal_organ_label_data[extracted_organ_bbox]

            extracted_organ_niigz_path = \
                    target_data_env.get_new_volume_niigz_path(extracted_organ.shape, \
                        'zoomed_0p5_extracted_roi_organ_%s' % organ_name)
            extracted_organ_label_niigz_path = \
                    target_data_env.get_new_volume_niigz_path(extracted_organ_label.shape, \
                        'zoomed_0p5_eextracted_roi_organ_%s_label' % organ_name, bits=8)

            print extracted_organ_niigz_path
            print extracted_organ_label_niigz_path

            if True:
            #if not os.path.exists(extracted_organ_niigz_path):
                save_as_nifti(extracted_organ, extracted_organ_niigz_path)

            if True:
            #if not os.path.exists(extracted_organ_label_niigz_path):
                save_as_nifti(extracted_organ_label, extracted_organ_label_niigz_path)

            target_extracted_organs_dict[organ_name] = { 'normal': None, 'zoomed': extracted_organ_niigz_path }
            target_extracted_organs_labels_dict[organ_name] = { 'normal': None, 'zoomed': extracted_organ_label_niigz_path }

            target_data_env.set_extracted_roi_abdomen_part_separated_organs(target_extracted_organs_dict)
            target_data_env.set_extracted_roi_abdomen_part_separated_organs_labels(target_extracted_organs_labels_dict)

            reference_data_env.save()
            target_data_env.save()

    print 'target_data_env.get_extracted_organs = %s' % str(target_data_env.get_extracted_roi_abdomen_part_separated_organs())
    print 'target_data_env.get_extracted_organs_labels = %s' % str(target_data_env.get_extracted_roi_abdomen_part_separated_organs_labels())

    print "--Extract the reference fish's organs labels from the initial volume..."
    reference_extracted_organs_dict = reference_data_env.get_extracted_roi_abdomen_part_separated_organs()
    reference_extracted_organs_labels_dict = reference_data_env.get_extracted_roi_abdomen_part_separated_organs_labels()
    reference_organs_labels_dict = reference_data_env.get_extracted_abdomen_part_separated_organs_labels()

    if True:
    #if (not reference_extracted_organs_labels_dict) and (not reference_extracted_organs_dict):
        for organ_name, organ_labels in reference_organs_labels_dict.iteritems():
            normal_organ_label_path = organ_labels['zoomed']
            normal_organ_label_data = open_data(normal_organ_label_path)

            original_aligned_data_path = reference_image_path_ext_abdomen_reg
            original_aligned_data = open_data(original_aligned_data_path)

            extracted_organ, extracted_organ_bbox = extract_largest_volume_by_label(original_aligned_data, normal_organ_label_data, bb_side_offset=bb_side_offset)
            extracted_organ_label = normal_organ_label_data[extracted_organ_bbox]

            extracted_organ_niigz_path = reference_data_env.get_new_volume_niigz_path(extracted_organ.shape, 'zoomed_0p5_extracted_roi_organ_%s' % organ_name)
            extracted_organ_label_niigz_path = \
                reference_data_env.get_new_volume_niigz_path(extracted_organ_label.shape, \
                    'zoomed_0p5_extracted_roi_organ_%s_label' % organ_name, bits=8)

            if True:
            #if not os.path.exists(extracted_organ_niigz_path):
                save_as_nifti(extracted_organ, extracted_organ_niigz_path)

            if True:
            #if not os.path.exists(extracted_organ_label_niigz_path):
                save_as_nifti(extracted_organ_label, extracted_organ_label_niigz_path)

            reference_extracted_organs_dict[organ_name] = { 'normal': extracted_organ_niigz_path, 'zoomed': None }
            reference_extracted_organs_labels_dict[organ_name] = { 'normal': extracted_organ_label_niigz_path, 'zoomed': None }

            reference_data_env.set_extracted_roi_abdomen_part_separated_organs(reference_extracted_organs_dict)
            reference_data_env.set_extracted_roi_abdomen_part_separated_organs_labels(reference_extracted_organs_labels_dict)

            reference_data_env.save()
            target_data_env.save()

    print 'reference_data_env.get_extracted_organs = %s' % \
            str(reference_data_env.get_extracted_roi_abdomen_part_separated_organs())
    print 'reference_data_env.get_extracted_organs_labels = %s' % \
            str(reference_data_env.get_extracted_roi_abdomen_part_separated_organs_labels())

@timing
def brain_segmentation_ants(reference_data_env, target_data_env):
    bb_side_offset = 5
    separation_overlap = 10

    # bb_side_offset = 2
    # separation_overlap = 3

    reference_data_env.load()
    target_data_env.load()

    organs_labels = ['heart']

    # Crop the raw data
    print "--Aligning and volumes' extraction"
    reference_data_results = initialize_env(reference_data_env, zoom_level=2, min_zoom_level=2, organs_labels=organs_labels)
    moving_data_results = initialize_env(target_data_env, zoom_level=2, min_zoom_level=2)

    #sys.exit(1)

    reference_data_env.save()
    target_data_env.save()



    #sys.exit(1)


    #generate_stats(reference_data_env)
    #generate_stats(target_data_env)

    # print "--Pre-alignment of the target fish to the known one"
    # Pre-alignment fish1 to fish_aligned
    # ants_prefix_prealign = 'pre_alignment'
    # ants_prealign_paths = target_data_env.get_aligned_data_paths(ants_prefix_prealign)
    # ants_prealign_names = target_data_env.get_aligned_data_paths(ants_prefix_prealign, produce_paths=False)
    #
    # working_env_prealign = target_data_env
    # reference_image_path_prealign = reference_data_env.envs['zoomed_0p5_extracted_input_data_path_niigz']
    # reference_image_path_prealign_raw = reference_data_env.envs['zoomed_0p5_extracted_input_data_path']
    # target_image_path_prealign = target_data_env.envs['zoomed_0p5_extracted_input_data_path_niigz']
    # output_name_prealign = ants_prealign_names['out_name']
    # warped_path_prealign = ants_prealign_paths['warped']
    # iwarped_path_prealign = ants_prealign_paths['iwarp']
    #
    # print 'reference_image_path_prealign = %s' % reference_image_path_prealign
    # print 'target_image_path_prealign = %s' % target_image_path_prealign
    #
    # if not os.path.exists(warped_path_prealign):
    #     align_fish_simple_ants(working_env_prealign, reference_image_path_prealign,\
    #                            target_image_path_prealign, output_name_prealign, \
    #                            warped_path_prealign, iwarped_path_prealign, \
    #                            reg_prefix=ants_prefix_prealign, use_syn=False, \
    #                            use_full_iters=False)
    #
    #     reference_data_env.save()
    #     target_data_env.save()
    # else:
    #     print "Data is already prealigned"

    #print  "--FINITO LA COMEDIA!"
    #return

    print  "--Registration of the reference fish to the target one"
    # Registration of fish_aligned to fish1
    ants_prefix_sep = 'parts_separation'
    ants_separation_paths = reference_data_env.get_aligned_data_paths(ants_prefix_sep)
    working_env_sep = reference_data_env
    reference_image_path_sep = reference_data_env.envs['zoomed_0p5_extracted_input_data_path_niigz']
    target_image_path_sep = target_data_env.envs['zoomed_0p5_extracted_input_data_path_niigz']

    print "\033[1;31m reference_image_path_sep = %s \033[0m" % reference_image_path_sep
    print "\033[1;31m target_image_path_sep = %s \033[0m" % target_image_path_sep

    target_image_path_sep_raw = target_data_env.envs['zoomed_0p5_extracted_input_data_path']
    output_name_sep = ants_separation_paths['out_name']
    warped_path_sep = ants_separation_paths['warped']
    iwarped_path_sep = ants_separation_paths['iwarp']

    print reference_image_path_sep
    print target_image_path_sep
    print output_name_sep

    if True:
    #if not os.path.exists(warped_path_sep):
        align_fish_simple_ants(working_env_sep, target_image_path_sep, reference_image_path_sep, \
                               output_name_sep, warped_path_sep, iwarped_path_sep, \
                               reg_prefix=ants_prefix_sep, use_syn=True, use_full_iters=False)

        reference_data_env.save()
        target_data_env.save()
    else:
        print "Data is already registered for separation"

    print "--Transforming brain and abdomen labels of the reference fish to the target's one"
    # Transforming labels of fish_aligned to fish1
    wokring_env_tr = target_data_env
    ref_image_path_tr = target_image_path_sep
    affine_transformation_path_tr = ants_separation_paths['gen_affine']
    def_transformation_path_tr = ants_separation_paths['warp']
    labels_image_path_tr = reference_data_env.envs['zoomed_0p5_extracted_input_data_labels_path_niigz']

    print ref_image_path_tr
    print affine_transformation_path_tr
    print def_transformation_path_tr
    print labels_image_path_tr
    print target_image_path_sep_raw

    __, __, new_size, __ = parse_filename(target_image_path_sep_raw)

    transformation_output_tr = target_data_env.get_new_volume_niigz_path(new_size, 'zoomed_0p5_extracted_labels')
    reg_prefix_tr = 'label_deforming'

    print transformation_output_tr

    if True:
    #if not os.path.exists(transformation_output_tr):
        apply_transform_fish(wokring_env_tr, ref_image_path_tr, \
                             affine_transformation_path_tr, \
                             labels_image_path_tr, transformation_output_tr, \
                             def_transformation_path=def_transformation_path_tr, \
                             reg_prefix=reg_prefix_tr)

        reference_data_env.save()
        target_data_env.save()
    else:
        print "Abdomen and brain data is already transformed"

    #Separate head and tail of reference image
    print "--Fish separation (reference image)..."
    abdomen_data_part_reference_niigz_path = None
    head_data_part_reference_niigz_path = None
    abdomen_data_part_labels_reference_niigz_path = None
    head_data_part_labels_reference_niigz_path = None

    if True:
    #if not os.path.exists(reference_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']) or \
    #   not os.path.exists(reference_data_env.envs['zoomed_0p5_head_input_data_path_niigz']) or \
    #   not os.path.exists(reference_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']) or \
    #   not os.path.exists(reference_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']):
        aligned_data_reference = open_data(reference_image_path_sep)
        aligned_data_labels_reference = open_data(labels_image_path_tr)

        print 'labels_image_path_tr = %s' % labels_image_path_tr

        separation_pos_reference, abdomen_label_reference_full, head_label_reference_full = find_separation_pos(aligned_data_labels_reference)

        abdomen_data_part_reference, head_data_part_reference = split_fish_by_pos(aligned_data_reference, separation_pos_reference, overlap=separation_overlap)
        abdomen_label_reference, _ = split_fish_by_pos(abdomen_label_reference_full, separation_pos_reference, overlap=separation_overlap)
        _, head_label_reference = split_fish_by_pos(head_label_reference_full, separation_pos_reference, overlap=separation_overlap)

        abdomen_data_part_reference_niigz_path = reference_data_env.get_new_volume_niigz_path(abdomen_data_part_reference.shape, 'zoomed_0p5_abdomen')
        if True:
        #if not os.path.exists(abdomen_data_part_reference_niigz_path):
            save_as_nifti(abdomen_data_part_reference, abdomen_data_part_reference_niigz_path)

        head_data_part_reference_niigz_path = reference_data_env.get_new_volume_niigz_path(head_data_part_reference.shape, 'zoomed_0p5_head')
        if True:
        #if not os.path.exists(head_data_part_reference_niigz_path):
            save_as_nifti(head_data_part_reference, head_data_part_reference_niigz_path)

        abdomen_data_part_labels_reference_niigz_path = reference_data_env.get_new_volume_niigz_path(abdomen_label_reference.shape, 'zoomed_0p5_abdomen_labels')
        if True:
        #if not os.path.exists(abdomen_data_part_labels_reference_niigz_path):
            save_as_nifti(abdomen_label_reference, abdomen_data_part_labels_reference_niigz_path)

        head_data_part_labels_reference_niigz_path = reference_data_env.get_new_volume_niigz_path(head_label_reference.shape, 'zoomed_0p5_head_labels')
        if True:
        #if not os.path.exists(head_data_part_labels_reference_niigz_path):
            save_as_nifti(head_label_reference, head_data_part_labels_reference_niigz_path)

        print abdomen_data_part_labels_reference_niigz_path
        print head_data_part_labels_reference_niigz_path

        reference_data_env.save()
    else:
        abdomen_data_part_reference_niigz_path = reference_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']
        head_data_part_reference_niigz_path = reference_data_env.envs['zoomed_0p5_head_input_data_path_niigz']
        abdomen_data_part_labels_reference_niigz_path = reference_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']
        head_data_part_labels_reference_niigz_path = reference_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']

    print "--Fish-organs separation (reference image)..."
    abdomen_separated_organs_labels_dict = reference_data_env.get_abdomen_separated_organs_labels()

    print 'organs_labels_dict = %s' % str(reference_data_env.get_organs_labels())
    print 'abdomen_separated_organs_labels_dict = %s' % str(reference_data_env.get_abdomen_separated_organs_labels())

    aligned_data_labels_reference = open_data(labels_image_path_tr)
    organ_separation_pos_reference, _, _ = find_separation_pos(aligned_data_labels_reference)

    if not abdomen_separated_organs_labels_dict:
        organs_labels_dict = reference_data_env.get_organs_labels()

        for organ_name, organ_labels in organs_labels_dict.iteritems():
            zoomed_organ_label_path = organ_labels['zoomed']
            zoomed_organ_label_data = open_data(zoomed_organ_label_path)

            abdomen_data_organ_part_reference, _ = split_fish_by_pos(zoomed_organ_label_data, organ_separation_pos_reference, overlap=separation_overlap)

            abdomen_data_organ_part_reference_niigz_path = reference_data_env.get_new_volume_niigz_path(abdomen_data_organ_part_reference.shape, 'zoomed_0p5_abdomen_organ_%s_labels' % organ_name)
            if not os.path.exists(abdomen_data_organ_part_reference_niigz_path):
                save_as_nifti(abdomen_data_organ_part_reference, abdomen_data_organ_part_reference_niigz_path)

            abdomen_separated_organs_labels_dict[organ_name] = { 'normal': None, 'zoomed': abdomen_data_organ_part_reference_niigz_path }

    reference_data_env.set_abdomen_separated_organs_labels(abdomen_separated_organs_labels_dict)
    reference_data_env.save()

    print 'abdomen_separated_organs_labels = %s' % str(reference_data_env.get_abdomen_separated_organs_labels())

    #Separate head and tail of target image
    print "--Fish separation (target image)..."
    abdomen_data_part_target_niigz_path = None
    abdomen_data_part_labels_target_niigz_path = None
    head_data_part_target_niigz_path = None
    head_data_part_labels_target_niigz_path = None

    if True:
    #if not os.path.exists(target_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']) or \
    #   not os.path.exists(target_data_env.envs['zoomed_0p5_head_input_data_path_niigz']) or \
    #   not os.path.exists(target_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']) or \
    #   not os.path.exists(target_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']):

        aligned_data_target = open_data(target_image_path_sep)
        aligned_data_labels_target = open_data(transformation_output_tr)

        separation_pos_target, abdomen_label_target_full, head_label_target_full = find_separation_pos(aligned_data_labels_target)

        abdomen_data_part_target, head_data_part_target = split_fish_by_pos(aligned_data_target, separation_pos_target, overlap=separation_overlap)
        abdomen_label_target, _ = split_fish_by_pos(abdomen_label_target_full, separation_pos_target, overlap=separation_overlap)
        _, head_label_target = split_fish_by_pos(head_label_target_full, separation_pos_target, overlap=separation_overlap)

        abdomen_data_part_target_niigz_path = target_data_env.get_new_volume_niigz_path(abdomen_data_part_target.shape, 'zoomed_0p5_abdomen')
        #if not os.path.exists(abdomen_data_part_target_niigz_path):
        if True:
            save_as_nifti(abdomen_data_part_target, abdomen_data_part_target_niigz_path)

        head_data_part_target_niigz_path = target_data_env.get_new_volume_niigz_path(head_data_part_target.shape, 'zoomed_0p5_head')
        if True:
        #if not os.path.exists(head_data_part_target_niigz_path):
            save_as_nifti(head_data_part_target, head_data_part_target_niigz_path)

        abdomen_data_part_labels_target_niigz_path = target_data_env.get_new_volume_niigz_path(abdomen_label_target.shape, 'zoomed_0p5_abdomen_labels')
        if True:
        #if not os.path.exists(abdomen_data_part_labels_target_niigz_path):
            save_as_nifti(abdomen_label_target, abdomen_data_part_labels_target_niigz_path)

        head_data_part_labels_target_niigz_path = target_data_env.get_new_volume_niigz_path(head_label_target.shape, 'zoomed_0p5_head_labels')
        if True:
        #if not os.path.exists(head_data_part_labels_target_niigz_path):
            save_as_nifti(head_label_target, head_data_part_labels_target_niigz_path)

        print abdomen_data_part_labels_target_niigz_path
        print head_data_part_labels_target_niigz_path

        target_data_env.save()
    else:
        abdomen_data_part_target_niigz_path = target_data_env.envs['zoomed_0p5_abdomen_input_data_path_niigz']
        abdomen_data_part_labels_target_niigz_path = target_data_env.envs['zoomed_0p5_abdomen_labels_input_data_path_niigz']
        head_data_part_target_niigz_path = target_data_env.envs['zoomed_0p5_head_input_data_path_niigz']
        head_data_part_labels_target_niigz_path = target_data_env.envs['zoomed_0p5_head_labels_input_data_path_niigz']


    organs_segmentation_ants(reference_data_env, target_data_env)

    sys.exit(1)

    print "--Register reference fish's head to the target's one..."
    #Register reference head to target one
    ants_prefix_head_reg = 'head_registration'
    ants_head_reg_paths = reference_data_env.get_aligned_data_paths(ants_prefix_head_reg)
    working_env_head_reg = reference_data_env
    reference_image_path_head_reg = head_data_part_target_niigz_path
    target_image_path_head_reg = head_data_part_reference_niigz_path
    output_name_head_reg = ants_head_reg_paths['out_name']
    warped_path_head_reg = ants_head_reg_paths['warped']
    iwarped_path_head_reg = ants_head_reg_paths['iwarp']

    if not os.path.exists(warped_path_head_reg):
        align_fish_simple_ants(working_env_head_reg, reference_image_path_head_reg, \
                               target_image_path_head_reg, output_name_head_reg, \
                               warped_path_head_reg, iwarped_path_head_reg, \
                               reg_prefix=ants_prefix_head_reg, use_syn=True, \
                               use_full_iters=False)

        reference_data_env.save()
        target_data_env.save()
    else:
        print "Head of the reference data is already registered to the head of target one"

    print "--Transfrom labels of reference fish's head into the target's one..."
    # Transforming labels of head of reference fish to the head of target one
    wokring_env_htr = target_data_env
    ref_image_path_htr = head_data_part_target_niigz_path
    print ref_image_path_htr
    transformation_path_htr = ants_head_reg_paths['gen_affine']
    def_transformation_path_htr = ants_head_reg_paths['warp']
    labels_image_path_htr = head_data_part_labels_reference_niigz_path
    test_data_htr = open_data(ref_image_path_htr)
    transformation_output_htr = target_data_env.get_new_volume_niigz_path(test_data_htr.shape, 'zoomed_0p5_head_brain_labels', bits=8)
    reg_prefix_htr = 'head_label_deforming'

    if not os.path.exists(transformation_output_htr):
        apply_transform_fish(wokring_env_tr, ref_image_path_htr, \
                             transformation_path_htr, labels_image_path_htr, \
                             transformation_output_htr, \
                             def_transformation_path=def_transformation_path_htr, \
                             reg_prefix=reg_prefix_htr)
        reference_data_env.save()
        target_data_env.save()
    else:
        print "Head data is already transformed"

    print "--Extract the target fish's brain using transformed head labels..."
    # Extract target brain volume
    head_brain_label_target = open_data(transformation_output_htr)
    head_brain_data_target = open_data(ref_image_path_htr)
    brain_data_volume_target, brain_data_volume_target_bbox = extract_largest_volume_by_label(head_brain_data_target, head_brain_label_target, bb_side_offset=bb_side_offset)
    brain_data_volume_target_niigz_path = target_data_env.get_new_volume_niigz_path(brain_data_volume_target.shape, 'zoomed_0p5_head_extracted_brain')

    print brain_data_volume_target_niigz_path

    # if True:
    if not os.path.exists(brain_data_volume_target_niigz_path):
        save_as_nifti(brain_data_volume_target, brain_data_volume_target_niigz_path)

    print "--Extract the reference fish's brain using labels..."
    # Extract reference brain volume
    head_brain_label_reference = open_data(labels_image_path_htr)
    head_brain_data_reference = open_data(target_image_path_head_reg)
    brain_data_volume_reference, brain_reference_bbox = extract_largest_volume_by_label(head_brain_data_reference, head_brain_label_reference, bb_side_offset=bb_side_offset)
    brain_data_labels_volume_reference = head_brain_label_reference[brain_reference_bbox]

    #dilate mask
    brain_data_labels_volume_reference = binary_dilation(brain_data_labels_volume_reference, structure=generate_binary_structure(3, 1)).astype(brain_data_labels_volume_reference.dtype)

    brain_data_volume_reference_niigz_path = reference_data_env.get_new_volume_niigz_path(brain_data_volume_reference.shape, 'zoomed_0p5_head_extracted_brain')
    brain_data_labels_volume_reference_niigz_path = reference_data_env.get_new_volume_niigz_path(brain_data_labels_volume_reference.shape, 'zoomed_0p5_head_extracted_brain_labels')

    print brain_data_volume_reference_niigz_path
    print brain_data_labels_volume_reference_niigz_path

    # if True:
    if not os.path.exists(brain_data_volume_reference_niigz_path):
        save_as_nifti(brain_data_volume_reference, brain_data_volume_reference_niigz_path)

    # if True:
    if not os.path.exists(brain_data_labels_volume_reference_niigz_path):
        save_as_nifti(brain_data_labels_volume_reference, brain_data_labels_volume_reference_niigz_path)

    print "--Register the reference fish's brain to the target's one..."
    # Register the reference brain to the target one
    ants_prefix_head_brain_reg = 'head_brain_registration'
    ants_head_brain_reg_paths = reference_data_env.get_aligned_data_paths(ants_prefix_head_brain_reg)
    working_env_head_brain_reg = reference_data_env
    reference_image_path_head_brain_reg = brain_data_volume_target_niigz_path
    target_image_path_head_brain_reg = brain_data_volume_reference_niigz_path
    output_name_head_brain_reg = ants_head_brain_reg_paths['out_name']
    warped_path_head_brain_reg = ants_head_brain_reg_paths['warped']
    iwarped_path_head_brain_reg = ants_head_brain_reg_paths['iwarp']

    # if True:
    if not os.path.exists(warped_path_head_brain_reg):
        align_fish_simple_ants(working_env_head_brain_reg, \
                               reference_image_path_head_brain_reg, \
                               target_image_path_head_brain_reg, \
                               output_name_head_brain_reg, \
                               warped_path_head_brain_reg, \
                               iwarped_path_head_brain_reg, \
                               reg_prefix=ants_prefix_head_brain_reg, \
                               use_syn=True, \
                               use_full_iters=True)

        reference_data_env.save()
        target_data_env.save()
    else:
        print "The brain of the head of the reference data is already registered to the brain of the head of target one"


    print "--Transform the reference fish's brain labels into the target's one..."
    # Transforming labels of the brain of head of reference fish to the brain of the head of target one
    wokring_env_brain_tr = target_data_env
    ref_image_path_brain_tr = brain_data_volume_target_niigz_path
    transformation_path_brain_tr = ants_head_brain_reg_paths['gen_affine']
    labels_image_path_brain_tr = brain_data_labels_volume_reference_niigz_path
    test_data_brain_tr = open_data(ref_image_path_brain_tr)
    transformation_output_brain_tr = target_data_env.get_new_volume_niigz_path(test_data_brain_tr.shape, 'zoomed_0p5_head_extracted_brain_labels', bits=8)
    reg_prefix_brain_tr = 'head_brain_label_deforming'
    def_transformation_path_brain_tr = ants_head_brain_reg_paths['warp']

    # if True:
    if not os.path.exists(transformation_output_brain_tr):
        apply_transform_fish(wokring_env_brain_tr, ref_image_path_brain_tr, \
                             transformation_path_brain_tr, labels_image_path_brain_tr, \
                             transformation_output_brain_tr, \
                             def_transformation_path=def_transformation_path_brain_tr, \
                             reg_prefix=reg_prefix_brain_tr)


        reference_data_env.save()
        target_data_env.save()
    else:
        print "The brain of the reference head data is already transformed"

    print "--Segment zoomed target fish's brain data..."
    zoomed_target_head_extracted_brain_data_labels = open_data(transformation_output_brain_tr)
    zoomed_target_head_extracted_brain_data = open_data(brain_data_volume_target_niigz_path)

    masked_zoomed_target_head_extracted_brain_data = zoomed_target_head_extracted_brain_data * zoomed_target_head_extracted_brain_data_labels

    masked_zoomed_target_head_extracted_brain_data_niigz_path = \
                target_data_env.get_new_volume_niigz_path(zoomed_target_head_extracted_brain_data.shape, 'zoomed_0p5_segmented_head_extracted_brain')

    print masked_zoomed_target_head_extracted_brain_data_niigz_path

    # if True:
    if not os.path.exists(masked_zoomed_target_head_extracted_brain_data_niigz_path):
        save_as_nifti(masked_zoomed_target_head_extracted_brain_data, \
                        masked_zoomed_target_head_extracted_brain_data_niigz_path)

    print "--Segment zoomed reference fish's brain data..."
    zoomed_reference_head_extracted_brain_data_labels = open_data(brain_data_labels_volume_reference_niigz_path)
    zoomed_reference_head_extracted_brain_data = open_data(brain_data_volume_reference_niigz_path)

    masked_zoomed_reference_head_extracted_brain_data = zoomed_reference_head_extracted_brain_data * zoomed_reference_head_extracted_brain_data_labels

    masked_zoomed_reference_head_extracted_brain_data_niigz_path = \
                reference_data_env.get_new_volume_niigz_path(masked_zoomed_reference_head_extracted_brain_data.shape, 'zoomed_0p5_segmented_head_extracted_brain')

    print masked_zoomed_reference_head_extracted_brain_data_niigz_path

    # if True:
    if not os.path.exists(masked_zoomed_reference_head_extracted_brain_data_niigz_path):
        save_as_nifti(masked_zoomed_reference_head_extracted_brain_data, \
                        masked_zoomed_reference_head_extracted_brain_data_niigz_path)

    print "--Complete the target fish's brain labels to full volume..."
    test_data_complete_vol_brain_target = open_data(target_image_path_sep)
    complete_vol_target_brain_labels_niigz_path = target_data_env.get_new_volume_niigz_path(test_data_complete_vol_brain_target.shape, 'zoomed_0p5_complete_volume_brain_labels', bits=8)

    # if True:
    if not os.path.exists(complete_vol_target_brain_labels_niigz_path):
        complete_vol_target_brain_labels = complete_brain_to_full_volume(abdomen_data_part_labels_target_niigz_path, \
                                                                         head_data_part_labels_target_niigz_path, \
                                                                         transformation_output_brain_tr, \
                                                                         brain_data_volume_target_bbox, \
                                                                         separation_overlap=separation_overlap);
        save_as_nifti(complete_vol_target_brain_labels, complete_vol_target_brain_labels_niigz_path)
    else:
        print "The brain labels of the target data (target fish) is already transformed."

    # print "--Inverse transfrom the completed target fish's brain labels to the initial alignment..."
    # wokring_env_brain_labels_inverse_tr = target_data_env
    # ref_image_space_path_brain_labels_inverse_tr = target_image_path_prealign
    # affine_transformation_path_brain_labels_inverse_tr = ants_prealign_paths['gen_affine']
    # labels_image_path_brain_inverse_tr = complete_vol_target_brain_labels_niigz_path
    # test_data_brain_inverse_tr = open_data(ref_image_space_path_brain_labels_inverse_tr)
    # transformation_output_brain_labels_inverse_tr = target_data_env.get_new_volume_niigz_path(test_data_brain_inverse_tr.shape, 'zoomed_0p5_complete_volume_brain_labels_initial_alignment', bits=8)
    # reg_prefix_brain_labels_inverse_tr = 'complete_volume_brain_labels_deforming_to_initial_alignment'
    #
    # if not os.path.exists(transformation_output_brain_labels_inverse_tr):
    #     apply_inverse_transform_fish(wokring_env_brain_labels_inverse_tr, \
    #                                  ref_image_space_path_brain_labels_inverse_tr, \
    #                                  affine_transformation_path_brain_labels_inverse_tr, \
    #                                  labels_image_path_brain_inverse_tr, \
    #                                  transformation_output_brain_labels_inverse_tr, \
    #                                  reg_prefix=reg_prefix_brain_labels_inverse_tr)
    #
    #     reference_data_env.save()
    #     target_data_env.save()
    # else:
    #     print "The completed target fish's brain labels is already transformed to the initial alignment."

    print "--Upscale the initial aligned completed target fish's brain labels to the input volume size..."
    scaled_initally_aligned_data_brain_labels_path = complete_vol_target_brain_labels_niigz_path
    upscaled_initially_aligned_complete_vol_target_brain_labels_niigz_path = target_data_env.get_new_volume_niigz_path(test_data_complete_vol_brain_target.shape, 'complete_volume_brain_labels_initial_alignment', bits=8)
    # zoomed_volume_bbox = target_data_env.get_zoomed_effective_volume_bbox()
    original_aligned_data_path = target_data_env.get_input_align_data_path()

    # if True:
    if not os.path.exists(upscaled_initially_aligned_complete_vol_target_brain_labels_niigz_path):
        upscaled_initally_aligned_data_brain_labels = scale_to_size(original_aligned_data_path, \
                                                                    scaled_initally_aligned_data_brain_labels_path, \
                                                                    scale=2.0, \
                                                                    order=0)
        save_as_nifti(upscaled_initally_aligned_data_brain_labels, \
                      upscaled_initially_aligned_complete_vol_target_brain_labels_niigz_path)
    else:
        print "The initially aligned completed target fish's brain labels is already upscaled to the input volume size."

    print "--Extract the target fish's brain labels from the upscaled initial volume..."
    upscaled_aligned_complete_target_data_brain_labels = open_data(upscaled_initially_aligned_complete_vol_target_brain_labels_niigz_path)
    original_aligned_data = open_data(original_aligned_data_path)

    complete_aligned_brain_data_target, complete_aligned_brain_data_target_bbox = extract_largest_volume_by_label(original_aligned_data, upscaled_aligned_complete_target_data_brain_labels, bb_side_offset=bb_side_offset*2)
    complete_aligned_brain_data_target_labels = upscaled_aligned_complete_target_data_brain_labels[complete_aligned_brain_data_target_bbox]

    extracted_brain_aligned_complete_volume_niigz_path = target_data_env.get_new_volume_niigz_path(complete_aligned_brain_data_target.shape, 'extracted_brain_aligned_complete_volume')
    extracted_brain_aligned_complete_volume_labels_niigz_path = target_data_env.get_new_volume_niigz_path(complete_aligned_brain_data_target_labels.shape, 'extracted_brain_aligned_complete_volume_labels', bits=8)

    print extracted_brain_aligned_complete_volume_niigz_path
    print extracted_brain_aligned_complete_volume_labels_niigz_path

    # if True:
    if not os.path.exists(extracted_brain_aligned_complete_volume_niigz_path):
        save_as_nifti(complete_aligned_brain_data_target, extracted_brain_aligned_complete_volume_niigz_path)

    # if True:
    if not os.path.exists(extracted_brain_aligned_complete_volume_labels_niigz_path):
        save_as_nifti(complete_aligned_brain_data_target_labels, extracted_brain_aligned_complete_volume_labels_niigz_path)

    print "--Extract the reference fish's brain labels from the original volume..."
    reference_original_aligned_input_data_labels_niigz_path = reference_data_env.get_input_aligned_data_labels_path()
    reference_original_aligned_input_data_labels = open_data(reference_original_aligned_input_data_labels_niigz_path)
    reference_original_aligned_input_data_brain_labels = extract_label_by_name(reference_original_aligned_input_data_labels, label_name='brain')

    reference_original_aligned_input_data_niigz_path = reference_data_env.get_input_align_data_path()
    reference_original_aligned_input_data = open_data(reference_original_aligned_input_data_niigz_path)

    extracted_aligned_brain_data_reference, extracted_aligned_brain_data_reference_bbox = \
            extract_largest_volume_by_label(reference_original_aligned_input_data, \
                        reference_original_aligned_input_data_brain_labels, bb_side_offset=bb_side_offset*2)
    extracted_aligned_brain_data_reference_labels = \
            reference_original_aligned_input_data_brain_labels[extracted_aligned_brain_data_reference_bbox]

    extracted_aligned_brain_data_reference_niigz_path = \
            reference_data_env.get_new_volume_niigz_path(extracted_aligned_brain_data_reference.shape, 'extracted_brain_aligned_complete_volume')
    extracted_aligned_brain_data_reference_labels_niigz_path = \
            reference_data_env.get_new_volume_niigz_path(extracted_aligned_brain_data_reference_labels.shape, 'extracted_brain_aligned_complete_volume_labels', bits=8)

    print extracted_aligned_brain_data_reference_niigz_path
    print extracted_aligned_brain_data_reference_labels_niigz_path

    # if True:
    if not os.path.exists(extracted_aligned_brain_data_reference_niigz_path):
        save_as_nifti(extracted_aligned_brain_data_reference, extracted_aligned_brain_data_reference_niigz_path)

    # if True:
    if not os.path.exists(extracted_aligned_brain_data_reference_labels_niigz_path):
        save_as_nifti(extracted_aligned_brain_data_reference_labels, extracted_aligned_brain_data_reference_labels_niigz_path)

    print "--Segment original target fish's brain data..."
    target_head_extracted_brain_data = open_data(extracted_brain_aligned_complete_volume_niigz_path)
    target_head_extracted_brain_data_labels = open_data(extracted_brain_aligned_complete_volume_labels_niigz_path)

    masked_target_head_extracted_brain_data = target_head_extracted_brain_data * target_head_extracted_brain_data_labels

    masked_target_head_extracted_brain_data_niigz_path = \
                target_data_env.get_new_volume_niigz_path(masked_target_head_extracted_brain_data.shape, 'extracted_segmented_brain_aligned_complete_volume')

    print masked_target_head_extracted_brain_data_niigz_path

    # if True:
    if not os.path.exists(masked_target_head_extracted_brain_data_niigz_path):
        save_as_nifti(masked_target_head_extracted_brain_data, \
                        masked_target_head_extracted_brain_data_niigz_path)

    print "--Segment original reference fish's brain data..."
    reference_head_extracted_brain_data = open_data(extracted_aligned_brain_data_reference_niigz_path)
    reference_head_extracted_brain_data_labels = open_data(extracted_aligned_brain_data_reference_labels_niigz_path)

    masked_reference_head_extracted_brain_data = reference_head_extracted_brain_data * reference_head_extracted_brain_data_labels

    masked_reference_head_extracted_brain_data_niigz_path = \
                reference_data_env.get_new_volume_niigz_path(masked_reference_head_extracted_brain_data.shape, 'extracted_segmented_brain_aligned_complete_volume')

    print masked_reference_head_extracted_brain_data_niigz_path

    # if True:
    if not os.path.exists(masked_reference_head_extracted_brain_data_niigz_path):
        save_as_nifti(masked_reference_head_extracted_brain_data, \
                        masked_reference_head_extracted_brain_data_niigz_path)

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

def scale_to_size_old2(target_data_path, extracted_scaled_data_path, \
                  extracted_scaled_data_bbox, scale=2.0, order=0):
    target_data = open_data(target_data_path)
    extracted_scaled_data = open_data(extracted_scaled_data_path)

    print 'scale_to_size (scale) = %f' % scale
    print 'scale_to_size (extracted_scaled_data_bbox) = %s' % str(extracted_scaled_data_bbox)

    rescaled_extracted_data = zoom(extracted_scaled_data, scale, order=order)
    rescaled_extracted_bbox = _zoom_bbox(extracted_scaled_data_bbox, scale)

    print 'scale_to_size (rescaled_extracted_bbox) = %s' % str(rescaled_extracted_bbox)
    print 'scale_to_size (extracted_scaled_data.shape) = %s' % str(extracted_scaled_data.shape)

    print 'scale_to_size (target_data.shape) = %s' % str(target_data.shape)

    complete_scaled_data = complete_volume_to_full_volume(target_data.shape, rescaled_extracted_data, rescaled_extracted_bbox)

    print 'complete_scaled_data = %s' % str(complete_scaled_data.shape)

    return complete_scaled_data

def scale_to_size_old3(original_aligned_data_path, extracted_scaled_data_path, \
                  extracted_scaled_data_bbox, scale=2.0, order=0):
    target_aligned_data = open_data(original_aligned_data_path)
    extracted_scaled_data = open_data(extracted_scaled_data_path)

    print 'scale_to_size (scale) = %f' % scale
    print 'scale_to_size (extracted_scaled_data_bbox) = %s' % str(extracted_scaled_data_bbox)

    rescaled_extracted_data = zoom(extracted_scaled_data, scale, order=order)
    rescaled_extracted_bbox = _zoom_bbox(extracted_scaled_data_bbox, scale)

    print 'scale_to_size (rescaled_extracted_bbox) = %s' % str(rescaled_extracted_bbox)
    print 'scale_to_size (extracted_scaled_data.shape) = %s' % str(extracted_scaled_data.shape)

    print 'scale_to_size (target_aligned_data.shape) = %s' % str(target_aligned_data.shape)

    complete_scaled_data = complete_volume_to_full_volume(target_aligned_data.shape, rescaled_extracted_data, rescaled_extracted_bbox)

    print 'complete_scaled_data = %s' % str(complete_scaled_data.shape)

    return complete_scaled_data

def scale_to_size(original_aligned_data_path, extracted_scaled_data_path, scale=2.0, order=0):
    target_aligned_data = open_data(original_aligned_data_path)
    extracted_scaled_data = open_data(extracted_scaled_data_path)

    print 'scale_to_size (target_aligned_data.shape) = %s' % str(target_aligned_data.shape)
    print 'scale_to_size (extracted_scaled_data.shape) = %s' % str(extracted_scaled_data.shape)

    rescaled_extracted_data = zoom(extracted_scaled_data, scale, order=order)

    print 'scale_to_size (SCALED rescaled_extracted_data.shape) = %s' % str(rescaled_extracted_data.shape)

    return rescaled_extracted_data

def complete_brain_to_full_volume(abdomed_part_path, head_part_path, extracted_brain_volume_path, extracted_brain_volume_bbox, separation_overlap=1):
    abdomed_part = open_data(abdomed_part_path)
    head_part = open_data(head_part_path)
    extracted_brain_volume = open_data(extracted_brain_volume_path)

    mask_volume_abdomed = np.zeros_like(abdomed_part, dtype=np.uint8)

    mask_volume_head = np.zeros_like(head_part, dtype=np.uint8)
    print 'mask_volume_head.shape = %s' % str(mask_volume_head.shape)
    print 'extracted_brain_volume.shape = %s' % str(extracted_brain_volume.shape)
    print 'extracted_brain_volume_bbox = %s' % str(extracted_brain_volume_bbox)
    mask_volume_head[extracted_brain_volume_bbox] = extracted_brain_volume

    # separation_overlap*2 - 1 because two parts overlap at some point and share it
    mask_full_volume = np.concatenate((mask_volume_abdomed[:-(separation_overlap*2 - 1),:,:], mask_volume_head), axis=0)

    return mask_full_volume

def complete_data_to_full_volume(abdomed_part_path, \
                                 head_part_path, \
                                 extracted_organ_volume_path, \
                                 extracted_organ_volume_bbox, \
                                 abdomen_local_part_path=None, \
                                 abdomen_local_part_bbox=None, \
                                 separation_overlap=1, \
                                 body_part='head'):
    abdomed_part = open_data(abdomed_part_path)
    mask_volume_abdomed = np.zeros_like(abdomed_part, dtype=np.uint8)

    head_part = open_data(head_part_path)
    mask_volume_head = np.zeros_like(head_part, dtype=np.uint8)

    extracted_organ_volume = open_data(extracted_organ_volume_path)

    mask_volume_abdomed_local = None
    if abdomen_local_part_path:
        abdomed_part_local = open_data(abdomen_local_part_path)
        mask_volume_abdomed_local = np.zeros_like(abdomed_part_local, dtype=np.uint8)

    if body_part == 'head':
        mask_volume_head[extracted_organ_volume_bbox] = extracted_organ_volume
    else:
        if mask_volume_abdomed_local is not None:
            mask_volume_abdomed_local[extracted_organ_volume_bbox] = extracted_organ_volume
            mask_volume_abdomed[abdomen_local_part_bbox] = mask_volume_abdomed_local
        else:
            mask_volume_abdomed[extracted_organ_volume_bbox] = extracted_organ_volume

    # separation_overlap*2 - 1 because two parts overlap at some point and share it
    mask_full_volume = np.concatenate((mask_volume_abdomed[:-(separation_overlap*2 - 1),:,:], mask_volume_head), axis=0)

    return mask_full_volume

def complete_volume_to_full_volume(target_data_shape, extracted_data, extracted_data_bbox):
    print 'target_data_shape = %s' % str(target_data_shape)
    print 'extracted_data.shape = %s' % str(extracted_data.shape)
    print 'extracted_data_bbox = %s' % str(extracted_data_bbox)

    completed_data = np.zeros(shape=target_data_shape, dtype=extracted_data.dtype)
    completed_data[extracted_data_bbox] = extracted_data

    return completed_data

def generate_stats(data_env):
    data_env.load()

    glob_stats = 'input_data_global_statistics'
    eyes_stats = 'input_data_eyes_statistics'

    if True:
    #if not data_env.is_entry_exists(glob_stats):
        input_data = open_data(data_env.envs['extracted_input_data_path'])

        print 'Global statistics...'
        t = Timer()

        binarized_stack, _, _ = binarizator(input_data)
        stack_statistics, thresholded_stack = object_counter(binarized_stack)

        #stack_statistics, thresholded_stack = gather_statistics(input_data)
        global_stats_path = data_env.get_statistic_path('global')
        stack_statistics.to_csv(global_stats_path)

        thresholded_stack_niigz_path = data_env.get_new_volume_niigz_path(thresholded_stack.shape, 'thresholded_stack')
        save_as_nifti(thresholded_stack, thresholded_stack_niigz_path)

        t.elapsed('Gathering statistics')

        data_env.save()
    else:
        print "Global statistics is already gathered: %s" % data_env.envs[glob_stats]

    if not data_env.is_entry_exists(eyes_stats):
        input_data = open_data(data_env.envs['extracted_input_data_path'])

        print 'Filtering eyes\' statistics...'
        t = Timer()

        eye_stack_statistics, _ = gather_statistics(input_data)

        eyes_stats = eyes_statistics(eye_stack_statistics)
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
        app = 'antsRegistrationSyN.sh -d 3 -f {fixedImagePath} ' \
                '-m {movingImagePath} -o {out_name} -n {num_threads} -t a -p f'.format(**args_fmt)
    else:
        if use_full_iters:
            app = 'antsRegistrationSyN.sh -d 3 -f {fixedImagePath} ' \
                    '-m {movingImagePath} -o {out_name} -n {num_threads} -t b -p f'.format(**args_fmt)
        else:
            app = 'antsRegistrationSyNQuick.sh -d 3 -f {fixedImagePath} ' \
                    '-m {movingImagePath} -o {out_name} -n {num_threads} -t b -p f'.format(**args_fmt)

    app = os.path.join(path_ants_scripts_fmt, app)
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
                'newSegmentationImage': transformation_output, \
                'defTransformation': def_transformation_path}

    cmd_template = 'antsApplyTransforms -d 3 -r {refImage} -n NearestNeighbor ' \
                   '-i {labelImage} -o {newSegmentationImage}'

    if def_transformation_path is None:
        cmd_template = cmd_template + ' -t {affineTransformation}'
    else:
        cmd_template = cmd_template + ' -t {defTransformation} -t {affineTransformation}'

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
    #MASK PARTS AND ANALYSE SEPSARATELY
    head_labels = (stack_labels == 1).astype(np.uint8)
    abdomen_labels = (stack_labels == 2).astype(np.uint8)
    #gills_labels = (stack_labels == 3).astype(np.uint8)

    head_objects_stats, _ = object_counter(head_labels)
    abdomen_objects_stats, _ = object_counter(abdomen_labels)

    abdomen_part_z = abdomen_objects_stats.loc[0, 'bb_z'] + abdomen_objects_stats.loc[0, 'bb_depth']

    #abdomen_label = (labels == objects_stats.loc[0, 'label']).astype(np.uint8)
    #head_label = ((labels != 0) & (labels != objects_stats.loc[0, 'label'])).astype(np.uint8)
    #abdomen_label = (labels == 2).astype(np.uint8)
    #head_label = (labels == (objects_stats.shape[0]-1)).astype(np.uint8)

    return int(abdomen_part_z / scale_factor), abdomen_labels, head_labels

def split_fish_by_pos(stack_data, separation_pos, overlap=1):
    separation_overlap = stack_data.shape[0] * overlap/100.
    abdomen_data_part = stack_data[:separation_pos + overlap,:,:]
    head_data_part = stack_data[(separation_pos - overlap + 1):,:,:]
    print 'separation_pos + overlap = %d' % (separation_pos + overlap)
    print 'separation_pos - overlap + 1 = %d' % (separation_pos - overlap + 1)

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
    return tuple([slice(int(round(0 if v.start is None else v.start * scale)), \
                        int(round(0 if v.stop is None else v.stop * scale)), \
                        int(round(0 if v.step is None else v.step * scale)) if v.step else None) \
                                    for v in bbox])

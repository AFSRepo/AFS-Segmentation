import os
import subprocess as subpr
import numpy as np
from modules.segmentation.eyes import eyes_statistics, eyes_zrange
from modules.tools.morphology import object_counter, gather_statistics, extract_largest_area_data
from modules.tools.io import get_filename, open_data, save_as_nifti, check_files
from modules.tools.processing import binarizator
from modules.tools.env import DataEnvironment
from modules.tools.misc import Timer
from scipy.ndimage.interpolation import zoom, rotate

def produce_cropped_data(data_env):
    phase_name = 'zoomed_0p5_extracted_input_data_path_niigz'

    data_env.load()

    if not data_env.is_entry_exists(phase_name):
        t = Timer()

        input_data = open_data(data_env.get_input_path())

        ext_volume = extract_effective_volume(input_data, bb_side_offset=30)
        zoomed_0p5_ext_volume = zoom(ext_volume, 0.5, order=3)

        ext_volume_path = data_env.get_new_volume_path(ext_volume.shape, 'extracted')
        ext_volume_niigz_path = data_env.get_new_volume_niigz_path(ext_volume.shape, 'extracted')

        zoomed_ext_volume_path = data_env.get_new_volume_path(zoomed_0p5_ext_volume.shape, 'zoomed_0p5_extracted')
        zoomed_ext_volume_niigz_path = data_env.get_new_volume_niigz_path(zoomed_0p5_ext_volume.shape, 'zoomed_0p5_extracted')

        save_as_nifti(ext_volume, ext_volume_niigz_path)
        save_as_nifti(zoomed_0p5_ext_volume, zoomed_ext_volume_niigz_path)

        ext_volume.tofile(ext_volume_path)
        zoomed_0p5_ext_volume.tofile(zoomed_ext_volume_path)

        data_env.save()

        t.elapsed('Data cropping is finished')
    else:
        print 'Files of \'%s\' phase are already in working directory: %s' % (phase_name, data_env.get_working_path())

def crop_align_data(fixed_data_env, moving_data_env):
    fixed_data_env.load()
    moving_data_env.load()

    produce_cropped_data(fixed_data_env)
    produce_cropped_data(moving_data_env)

    fixed_data_env.save()
    moving_data_env.save()

    moving_data_env.set_target_data_path(fixed_data_env.get_input_path())
    moving_data_env.save()

    generate_stats(fixed_data_env)
    generate_stats(moving_data_env)

    align_fish(fixed_data_env, moving_data_env)




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

def align_fish(fixed_env, moving_env):
    working_path = moving_env.get_working_path()
    os.environ["ANTSPATH"] = moving_env.ANTSPATH

    fPath = fixed_env.envs['zoomed_0p5_extracted_input_data_path_niigz']
    mPath = moving_env.envs['zoomed_0p5_extracted_input_data_path_niigz']

    ants_paths = moving_env.get_aligned_data_paths("parts_separation")
    ants_names = moving_env.get_aligned_data_paths("parts_separation", produce_paths=False)


    app2 = ["bash","antsRegistrationSyNQuick.sh","-d", "3", "-f", fPath, "-m", mPath, "-o", ants_names['out_name'], "-t", "r", "-n", "4"]

    args_fmt = {'out_name': ants_names['out_name'], 'warped_path': ants_names['warped'], 'iwarped_path': ants_names['iwarp'], 'fixedImagePath': fPath, 'movingImagePath': mPath}
    app3 = 'antsRegistration --dimensionality 3 --float 0 --output [{out_name},{warped_path},{iwarped_path}] --interpolation BSpline --use-histogram-matching 0 --initial-moving-transform [{fixedImagePath},{movingImagePath},1] --transform Rigid[0.01] --metric MI[{fixedImagePath},{movingImagePath},1,32,Regular,0.25] --convergence [1000x500x250x10,1e-8,10] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox'.format(**args_fmt)

    process = subpr.Popen(app3, cwd=working_path)
    streamdata = process.communicate()[0]
    rc = process.returncode
    print "antsRegistration = %d" % rc

    moving_env.save()

def apply_transform_fish(fixed_env, moving_env):
    working_path = moving_env.get_working_path()
    os.environ["ANTSPATH"] = moving_env.ANTSPATH

    fPath = fixed_env.envs['zoomed_0p5_extracted_input_data_path_niigz']
    mPath = moving_env.envs['zoomed_0p5_extracted_input_data_path_niigz']

    ants_paths = moving_env.get_aligned_data_paths("parts_separation")
    ants_names = moving_env.get_aligned_data_paths("parts_separation", produce_paths=False)


    app2 = ["bash","antsRegistrationSyNQuick.sh","-d", "3", "-f", fPath, "-m", mPath, "-o", ants_names['out_name'], "-t", "r", "-n", "4"]

    args_fmt = {'out_name': ants_names['out_name'], 'warped_path': ants_names['warped'], 'iwarped_path': ants_names['iwarp'], 'fixedImagePath': fPath, 'movingImagePath': mPath}
    app3 = 'antsRegistration --dimensionality 3 --float 0 --output [{out_name},{warped_path},{iwarped_path}] --interpolation BSpline --use-histogram-matching 0 --initial-moving-transform [{fixedImagePath},{movingImagePath},1] --transform Rigid[0.01] --metric MI[{fixedImagePath},{movingImagePath},1,32,Regular,0.25] --convergence [1000x500x250x10,1e-8,10] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox'.format(**args_fmt)

    process = subpr.Popen(app3, cwd=working_path)
    streamdata = process.communicate()[0]
    rc = process.returncode
    print "%s = %d" % (app, rc)

    moving_env.save()

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

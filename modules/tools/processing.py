import os
import numpy as np
import pandas as pd
from operator import attrgetter
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import median_filter
from scipy.ndimage.morphology import binary_dilation, binary_closing, binary_fill_holes, binary_opening
from scipy.ndimage.measurements import label, find_objects
from skimage.filters import threshold_otsu, threshold_li
from skimage.morphology import disk, white_tophat
from skimage.measure import regionprops
from ..segmentation.eyes import eyes_statistics, eyes_zrange
from .morphology import object_counter, gather_statistics, extract_largest_area_data, stats_at_slice, rotate_stats, flip_stats
from .misc import timing

def binarizator(stack_data, eyes_stats=None, filter_size=6,
        non_zeros_ratio=0.5, tolerance=50, preserve_big_objects=True, verbose=False):
    num_slices = stack_data.shape[0]

    print 'Binarizing - Eye range estimating...'
    eyes_range = eyes_zrange(eyes_stats)
    if eyes_range.size:
        print 'Eyes range: %d-%d' % (np.min(eyes_range), np.max(eyes_range))

    #otsu thresholding
    thresholded_stack = np.empty_like(stack_data, dtype=np.uint8)

    print 'Binarizing - Thresholding...'
    for slice_idx in np.arange(num_slices):
        threshold_global_otsu = threshold_otsu(stack_data[slice_idx])
        thresholded_stack[slice_idx] = stack_data[slice_idx] >= threshold_global_otsu

    print 'Binarizing - Nonzeros, Closing, Filling, Medianing...'
    for slice_idx in np.arange(num_slices):
        #apply more hard thresholding to eyes region

        if eyes_range.size and slice_idx in eyes_range:
            threshold_global_li = threshold_li(stack_data[slice_idx])
            thresholded_stack[slice_idx] = stack_data[slice_idx] >= threshold_global_li

        if _check_slice_position(slice_idx, num_slices):
            #check artifacts
            dilated_slice = binary_dilation(thresholded_stack[slice_idx],
                                            iterations=5).flatten()
            num_nonzeros = np.count_nonzero(dilated_slice)

            if num_nonzeros > (dilated_slice.size - num_nonzeros) * non_zeros_ratio:
                thresholded_stack[slice_idx] = np.zeros_like(thresholded_stack[slice_idx])
                continue;

        thresholded_stack[slice_idx] = binary_closing(thresholded_stack[slice_idx],
                                                      iterations=4)
        thresholded_stack[slice_idx] = binary_fill_holes(thresholded_stack[slice_idx])
        thresholded_stack[slice_idx] = median_filter(thresholded_stack[slice_idx],
                                                     size=(filter_size,filter_size))

    print 'Binarizing - Filling along Y...'
    for slice_idx in np.arange(thresholded_stack.shape[1]):
        thresholded_stack[:,slice_idx,:] = binary_fill_holes(thresholded_stack[:,slice_idx,:])

    print 'Binarizing - Filling along X...'
    for slice_idx in np.arange(thresholded_stack.shape[2]):
        thresholded_stack[:,:,slice_idx] = binary_fill_holes(thresholded_stack[:,:,slice_idx])

    print 'Binarizing - Big object preservation...'
    if preserve_big_objects:
        for slice_idx in np.arange(num_slices):
            if np.count_nonzero(thresholded_stack[slice_idx]):
                labeled_slice, num_labels = label(thresholded_stack[slice_idx])
                max_area_label = max(regionprops(labeled_slice), key=attrgetter('area')).label
                labeled_slice[ labeled_slice != max_area_label] = 0
                labeled_slice[ labeled_slice == max_area_label] = 1
                thresholded_stack[slice_idx] = labeled_slice

    print 'Binarizing - Remove streak artifacts...'
    for slice_idx in np.arange(1, num_slices):
        prev_slice_labels, _ = label(thresholded_stack[slice_idx - 1])
        curr_slice_labels, _ = label(thresholded_stack[slice_idx])

        prev_props = regionprops(prev_slice_labels)
        curr_props = regionprops(curr_slice_labels)

        if len(prev_props) and len(curr_props):
            prev_slice_bbox = max(prev_props, key=attrgetter('area')).bbox
            curr_slice_bbox = max(curr_props, key=attrgetter('area')).bbox

            if _detect_streaks(prev_slice_bbox, curr_slice_bbox, tolerance):
                if verbose:
                    print 'Slice #%d' % slice_idx
                filtered_streaks = white_tophat(thresholded_stack[slice_idx], \
                                                selem=disk(15))
                removed_streaks = thresholded_stack[slice_idx] - filtered_streaks
                thresholded_stack[slice_idx] = \
                        median_filter(removed_streaks, size=(filter_size,filter_size))

    bbox = find_objects(thresholded_stack)
    return thresholded_stack, bbox, eyes_stats

def _detect_streaks(prev_bbox, curr_bbox, tolerance):
    p_width, p_height = prev_bbox[3] - prev_bbox[1], prev_bbox[2] - prev_bbox[0]
    c_width, c_height = curr_bbox[3] - curr_bbox[1], curr_bbox[2] - curr_bbox[0]

    return True if abs(p_width - c_width) > tolerance or \
                   abs(p_height - c_height) > tolerance else False


def _check_slice_position(slice_idx, num_slices, slice_threshold_ratio=0.2):
    num_slices_threshold = np.ceil(num_slices * slice_threshold_ratio)

    if slice_idx in np.arange(num_slices_threshold) or \
            slice_idx in np.arange(num_slices - num_slices_threshold, num_slices):
        return True

    return False

def interp3(x, y, z, v, xi, yi, zi, **kwargs):
    """Sample a 3D array "v" with pixel corner locations at "x","y","z" at the
    points in "xi", "yi", "zi" using linear interpolation. Additional kwargs
    are passed on to ``scipy.ndimage.map_coordinates``."""
    def index_coords(corner_locs, interp_locs):
        index = np.arange(len(corner_locs))
        print "index_coords %d %s" % (len(corner_locs), str(index.dtype))
        if np.all(np.diff(corner_locs) < 0):
            corner_locs, index = corner_locs[::-1], index[::-1]
        return np.interp(interp_locs, corner_locs, index)

    print "stage 1"
    print_available_ram()

    orig_shape = np.asarray(xi).shape
    xi, yi, zi = np.atleast_1d(xi, yi, zi)
    print xi.dtype, yi.dtype, zi.dtype

    print "stage 2"
    print_available_ram()

    for arr in [xi, yi, zi]:
        arr.shape = -1

    print "stage 3"
    print_available_ram()

    output = np.empty(xi.shape, dtype=np.float32)

    print "stage 4"
    print_available_ram()

    coords = np.array([index_coords(*item) for item in zip([x, y, z], [xi, yi, zi])])
    print "coords = %f,  %s" % (coords[0][2000], coords[0].dtype)

    print "stage 5"
    print_available_ram()

    map_coordinates(v, coords, output=output, **kwargs)

    print "stage 6"
    print_available_ram()

    return output.reshape(orig_shape)

def _get_rot_matrix_arbitrary_axis(ux, uy, uz, theta):
    mat = np.matrix([[np.cos(theta) + ux*ux*(1.-np.cos(theta)), \
                     ux*uy*(1.-np.cos(theta))-uz*np.sin(theta), \
                     ux*uz*(1.-np.cos(theta))+uy*np.sin(theta)],\

                    [uy*ux*(1.-np.cos(theta))+uz*np.sin(theta), \
                     np.cos(theta)+uy*uy*(1.-np.cos(theta)), \
                     uy*uz*(1.-np.cos(theta))-ux*np.sin(theta)], \

                    [uz*ux*(1.-np.cos(theta))-uy*np.sin(theta), \
                     uz*uy*(1.-np.cos(theta))+ux*np.sin(theta), \
                     np.cos(theta)+uz*uz*(1.-np.cos(theta))]], dtype=np.float32)

    return mat

def rotate_around_vector(data, origin_point, rot_axis, angle, interp_order=3):
    dims = data.shape

    R = _get_rot_matrix_arbitrary_axis(rot_axis[0], rot_axis[1], rot_axis[2], angle)

    print_available_ram()
    print 'Rotation around vector - Coords shifting...'
    zv, yv, xv = np.meshgrid(np.arange(dims[0], dtype=np.int32), \
                             np.arange(dims[1], dtype=np.int32), \
                             np.arange(dims[2], dtype=np.int32), indexing='ij')
    print zv[0].dtype
    print "%s = %d = %f MB" % (str(zv.shape), zv.shape[0]*zv.shape[1]*zv.shape[2], (zv.shape[0]*zv.shape[1]*zv.shape[2] * 4 / 1024. / 1024. / 1024.))
    print "%s = %d = %f MB" % (str(yv.shape), yv.shape[0]*yv.shape[1]*yv.shape[2], (yv.shape[0]*yv.shape[1]*yv.shape[2] * 4 / 1024. / 1024. / 1024.))
    print "%s = %d = %f MB" % (str(xv.shape), xv.shape[0]*xv.shape[1]*xv.shape[2], (xv.shape[0]*xv.shape[1]*xv.shape[2] * 4 / 1024. / 1024. / 1024.))

    print_available_ram()
    print "origin_point = %s" % str(origin_point)
    coordinates_translated = np.array([np.ravel(xv - origin_point[0]).T, \
                                       np.ravel(yv - origin_point[1]).T, \
                                       np.ravel(zv - origin_point[2]).T]).astype(np.int32)
    print coordinates_translated[0][0].dtype
    print_available_ram()
    del zv, yv, xv
    print_available_ram()

    print 'Rotation around vector - Rotating...'
    coordinates_rotated = np.array(R * coordinates_translated, dtype=np.float32)

    del coordinates_translated

    print "coordinates_rotated = %f, %s" % (coordinates_rotated[0][0], coordinates_rotated[0][0].dtype)
    print_available_ram()

    print 'Rotation around vector - Coords back shifting...'
    coordinates_rotated[0] = coordinates_rotated[0] + origin_point[0]
    coordinates_rotated[1] = coordinates_rotated[1] + origin_point[1]
    coordinates_rotated[2] = coordinates_rotated[2] + origin_point[2]
    print_available_ram()

    print 'Rotation around vector - Data reshaping...'
    x_coordinates = np.reshape(coordinates_rotated[0,:], dims)
    y_coordinates = np.reshape(coordinates_rotated[1,:], dims)
    z_coordinates = np.reshape(coordinates_rotated[2,:], dims)
    del coordinates_rotated
    print_available_ram()

    #get the values for your new coordinates
    print 'Rotation around vector - Interpolation in 3D...'
    interp_data = interp3(np.arange(dims[0], dtype=np.uint32), np.arange(dims[1], dtype=np.uint32), np.arange(dims[2], dtype=np.uint32), data, \
                          z_coordinates, y_coordinates, x_coordinates, order=interp_order)
    print_available_ram()

    return interp_data, R, origin_point

def _get_reference_landmark_orientation(ref_data, bb_side_offset=20):
    '''
    Returns the index of landmark slice in fraction, the offset form
    eyes in fraction; centroid of landmark slice forms vector directing
    from eyes' center which should be rotated by theta (rads) around
    the vector obtained as a cross product of this vector and z-axis.
    '''
    binarized_stack, bbox, eyes_stats = binarizator(ref_data)
    binary_stack_stats, _ = object_counter(binarized_stack)
    largest_volume_region, largest_volume_region_bbox = extract_largest_area_data(input_data, binary_stack_stats, bb_side_offset=bb_side_offset, force_bbox_fit=False, pad_data=True)

    stack_statistics, _ = gather_statistics(largest_volume_region)
    eyes_stats = eyes_statistics(stack_statistics)

    eye_c = np.round([eyes_stats['com_x'].mean(), eyes_stats['com_y'].mean(), eyes_stats['com_z'].mean()]).astype(np.int32)

    #let's use 530 slice as tail landmark
    lmt_slice_idx = 530
    ext_vol_len = largest_volume_region.shape[0]
    eye1_idx_frac, eye2_idx_frac = eyes_stats['com_z'].values[0] / float(ext_vol_len),\
                                   eyes_stats['com_z'].values[1] / float(ext_vol_len)
    landmark_tail_idx_frac = lmt_slice_idx / float(ext_vol_len)
    landmark_tail_idx_eyes_offset_frac = (eye1_idx_frac + eye2_idx_frac)/2.0 - landmark_tail_idx_frac

    landmark_tail_idx = int(float(eye1_idx_frac * ext_vol_len + eye2_idx_frac * ext_vol_len)/2.0 - landmark_tail_idx_eyes_offset_frac * ext_vol_len)

    #z -> y
    #y -> x
    lm_slice_stats, lm_labeled_slice = stats_at_slice(largest_volume_region, landmark_tail_idx)

    #(z,y,x)
    z_axis_vec = np.array([0., 0., -1.])
    spinal_vec = np.array([0, lm_slice_stats['com_z'] - eye_c[1], landmark_tail_idx - eye_c[2]])

    tail_com_y, tail_com_z = lm_slice_stats['com_z'].values[0] - eye_c[1], landmark_tail_idx - eye_c[2]

    rot_axis = np.cross(z_axis_vec, spinal_vec)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    theta = np.arccos(z_axis_vec.dot(spinal_vec)/(np.sqrt(z_axis_vec.dot(z_axis_vec)) * np.sqrt(spinal_vec.dot(spinal_vec))))

    return theta, rot_axis, landmark_tail_idx_frac, landmark_tail_idx_eyes_offset_frac

def _flip_z(stack_data, eyes_stats, is_tail_fisrt=True):
    data_shape = stack_data.shape

    eyes_z_pos = -1

    if isinstance(eyes_stats, list) or isinstance(eyes_stats, np.ndarray):
        eyes_z_pos = (eyes_stats[0][2] + eyes_stats[1][2]) / 2.
    else:
        eyes_z_pos = eyes_stats['com_z'].mean()

    #print 'data_shape[0] = %d' % data_shape[0]
    #print 'eyes_z_pos = %s' % str(eyes_z_pos)

    eyes_in_beginning = (data_shape[0]/2.0 > eyes_z_pos)

    if is_tail_fisrt:
        return (stack_data[::-1,:,:], True) if eyes_in_beginning else (stack_data, False)
    else:
        return (stack_data, False) if eyes_in_beginning else (stack_data[::-1,:,:], True)

def _flip_y(data, head_slice_idx):
    #(y -> x, z -> y)
    head_slice_stats, head_labeled_slice = stats_at_slice(data, head_slice_idx)
    central_head_part = head_labeled_slice[head_slice_stats['bb_z']:head_slice_stats['bb_z']+head_slice_stats['bb_depth'],\
                                           head_slice_stats['bb_y']:head_slice_stats['bb_y']+head_slice_stats['bb_height']]
    top_part, bottom_part = central_head_part[:central_head_part.shape[0]/2,:], central_head_part[central_head_part.shape[0]/2+1:,:]

    top_part, bottom_part = binary_opening(top_part, iterations=2), binary_opening(bottom_part, iterations=2)
    non_zeros_top, non_zeros_bottom = np.count_nonzero(top_part), np.count_nonzero(bottom_part)

    print 'Vectical orientation correction - Counting non-zeros (top=%d, bottom=%d)...' % (non_zeros_top, non_zeros_bottom)
    print 'Vectical orientation correction - Reversing y-direction if needed...'

    return (data[:,::-1,:], True) if non_zeros_top < non_zeros_bottom else (data, False)

def _calc_approx_eyes_params(data_shape):
    min_area, min_sphericity = -1, 1.0

    if sum(data_shape) > 2800:
        min_area = 20000
    elif sum(data_shape) < 2100:
        min_area = 8000
    else:
        min_area = 12000

    return min_area, min_sphericity

def check_depth_orientation(data, is_tail_fisrt=True):
    print_available_ram()

    print 'Depth orientation correction - Finding eyes...'
    stack_statistics, threshoded_data = gather_statistics(data)
    min_area, min_sphericity = _calc_approx_eyes_params(data.shape)
    eyes_stats = eyes_statistics(stack_statistics, min_area=min_area, min_sphericity=min_sphericity)

    print_available_ram()

    print 'Depth orientation correction - Reversing data and stats z-direction if needed...'
    data, flipped = _flip_z(data, eyes_stats, is_tail_fisrt=is_tail_fisrt)

    print_available_ram()

    if flipped:
        eyes_stats = flip_stats(eyes_stats, data.shape, axes=(0,))

    print_available_ram()

    return data, eyes_stats, threshoded_data

def check_vertical_orientation(data, eyes_stats=None):
    if eyes_stats is None:
        print 'Vectical orientation correction - Finding eyes...'
        stack_statistics, _ = gather_statistics(data)
        min_area, min_sphericity = _calc_approx_eyes_params(data.shape)
        eyes_stats = eyes_statistics(stack_statistics, min_area=min_area, min_sphericity=min_sphericity)

    eye_c = np.round([eyes_stats['com_x'].mean(), eyes_stats['com_y'].mean(), eyes_stats['com_z'].mean()]).astype(np.int32)
    avg_eye_size = np.round(np.mean([eyes_stats['bb_width'].mean(), eyes_stats['bb_height'].mean(), eyes_stats['bb_depth'].mean()]))

    #shift from eye's to tail direction
    z_shift = avg_eye_size * 2
    head_slice_idx = eye_c[2] - z_shift

    print 'Vectical orientation correction - Analyzing head region slice #%d...' % head_slice_idx
    data, flipped = _flip_y(data, head_slice_idx)

    if flipped:
        eyes_stats = flip_stats(eyes_stats, data.shape, axes=(1,))

    return data, eyes_stats

def _align_by_eyes_centroids(data, centroids, interp_order=3):
    dims = data.shape
    eye_l, eye_r = centroids

    eye_c = np.array([(lv + rv)/2. for lv,rv in zip(eye_l, eye_r)])
    p = np.array([eye_l, eye_r])
    p_s = p - eye_c

    theta = np.arccos(p_s[0][0]/np.sqrt(p_s[0].dot(p_s[0])))

    x_axis_vec = np.array([1.,0.,0.])
    rot_axis = np.cross(p_s[0], x_axis_vec)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    interp_data, R, origin_point = rotate_around_vector(data, eye_c, rot_axis, -theta, interp_order=interp_order)

    return interp_data, R, origin_point

def align_eyes_centroids(data, eyes_stats=None):
    print_available_ram()

    if eyes_stats is None:
        print 'Aligning of eyes\' centroids - Finding eyes...'
        stack_statistics, _ = gather_statistics(data)
        min_area, min_sphericity = _calc_approx_eyes_params(data.shape)
        eyes_stats = eyes_statistics(stack_statistics, min_area=min_area, min_sphericity=min_sphericity)

    print_available_ram()

    print 'Aligning of eyes\' centroids - Stats extracting...'
    eye1_com, eye2_com = np.array([eyes_stats['com_x'].values[0], eyes_stats['com_y'].values[0], eyes_stats['com_z'].values[0]]), \
                             np.array([eyes_stats['com_x'].values[1], eyes_stats['com_y'].values[1], eyes_stats['com_z'].values[1]])
    eyes_coms = [eye1_com, eye2_com]

    print_available_ram()

    eye1_sbbox, eye2_sbbox = np.array([eyes_stats['bb_width'].values[0], eyes_stats['bb_height'].values[0], eyes_stats['bb_depth'].values[0]]), \
                                 np.array([eyes_stats['bb_width'].values[1], eyes_stats['bb_height'].values[1], eyes_stats['bb_depth'].values[1]])
    eyes_sizes = [eye1_sbbox, eye2_sbbox]

    print_available_ram()

    print 'Aligning of eyes\' centroids - Aligning along x- and y-axes...'
    aligned_data, R, origin_point  = _align_by_eyes_centroids(data, eyes_coms)

    print_available_ram()

    aligned_eyes_stats = rotate_stats(eyes_stats, R, origin_point)

    print_available_ram()

    return aligned_data, aligned_eyes_stats

def align_tail_part(input_data, landmark_tail_idx_frac=0.56383, spinal_angle=0.12347, interp_order=3):
    print 'Aligning of tail part - Extracting z-bounded data...'
    binarized_stack, bbox, eyes_stats = binarizator(input_data)
    binary_stack_stats, _ = object_counter(binarized_stack)
    largest_volume_region, _ = \
                    extract_largest_area_data(input_data, binary_stack_stats, bb_side_offset=50, \
                                              force_bbox_fit=False, pad_data=True, extact_axes=(0,), \
                                              force_positiveness=False)

    print 'Aligning of tail part - Finding eyes...'
    stack_statistics, _ = gather_statistics(largest_volume_region)
    min_area, min_sphericity = _calc_approx_eyes_params(input_data.shape)
    eyes_stats = eyes_statistics(stack_statistics, min_area=min_area, min_sphericity=min_sphericity)

    print 'Aligning of tail part - Calculating eyes\' center, landmark point...'
    eye_c = np.round([eyes_stats['com_x'].mean(), eyes_stats['com_y'].mean(), eyes_stats['com_z'].mean()]).astype(np.int32)

    ext_vol_len = largest_volume_region.shape[0]
    eye1_idx_frac, eye2_idx_frac = eyes_stats['com_z'].values[0] / float(ext_vol_len),\
                                   eyes_stats['com_z'].values[1] / float(ext_vol_len)

    landmark_tail_idx = int(ext_vol_len * landmark_tail_idx_frac)
    lm_slice_stats, lm_labeled_slice = stats_at_slice(largest_volume_region, landmark_tail_idx)

    z_axis_vec = np.array([0., 0., -1.])
    tail_com_y, tail_com_z = lm_slice_stats['com_z'].values[0] - eye_c[1], landmark_tail_idx - eye_c[2]
    spinal_vec = np.array([0, tail_com_y, tail_com_z])

    rot_axis = np.cross(z_axis_vec, spinal_vec)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    theta = np.arccos(z_axis_vec.dot(spinal_vec)/(np.sqrt(z_axis_vec.dot(z_axis_vec)) * np.sqrt(spinal_vec.dot(spinal_vec))))

    if tail_com_y < 0:
        theta = (theta - spinal_angle) if theta > spinal_angle else -(spinal_angle - theta)
    else:
        theta = theta + spinal_angle

    print 'Aligning of tail part - Rotating data around %s vector by %f degree...' % (str(rot_axis), np.rad2deg(-theta))
    data_rotated, _, _ = rotate_around_vector(largest_volume_region, eye_c, rot_axis, theta, interp_order=interp_order)

    print 'Aligning of tail part - Extracting aligned data...'
    data_rotated_binarized_stack, _, _ = binarizator(data_rotated)
    data_rotated_binary_stack_stats, _ = object_counter(data_rotated_binarized_stack)
    largest_data_rotated_region, _ = extract_largest_area_data(data_rotated, data_rotated_binary_stack_stats, \
                                                               bb_side_offset=50, force_bbox_fit=False, pad_data=True, \
                                                               force_positiveness=False)

    return largest_data_rotated_region

TMP_PATH = "C:\\Users\\Administrator\\Documents\\tmp"
#filepath = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish200\\fish200_rotated_32bit_286x286x1235.raw"
#from modules.tools.io import create_filename_with_shape
#filled_y_data.astype(np.float32).tofile(os.path.join(TMP_PATH, create_filename_with_shape(filepath, filled_y_data.shape, prefix='stage3')))
#aligned_data.astype(np.float32).tofile(os.path.join(TMP_PATH, create_filename_with_shape(filepath, aligned_data.shape, prefix='stage2')))
#tail_aligned_data.astype(np.float32).tofile(os.path.join(TMP_PATH, create_filename_with_shape(filepath, tail_aligned_data.shape, prefix='stage4')))
filepath = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish215\\fish215_32bit_640x640x2478.raw"
#filepath = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish215\\fish215_32bit_320x320x1239.raw"
from .io import open_data, create_filename_with_shape
from .misc import print_available_ram
import matplotlib.pyplot as plt

@timing
def align_fish_by_eyes_tail___old(input_data, landmark_tail_idx_frac=0.56383, spinal_angle=0.12347, interp_order=3):
    flipped_z_data, flipped_z_eyes_stats = None, None
    curpath_data = os.path.join(TMP_PATH, create_filename_with_shape(filepath, tuple([1239,320,320]), prefix='flippedzdata'))
    curpath_stats = os.path.join(TMP_PATH, "flipped_z_eyes_stats_fish215_32bit_320x320x1239.csv")
    if os.path.exists(curpath_data) and os.path.exists(curpath_stats):
        flipped_z_data = open_data(curpath_data)
        flipped_z_eyes_stats = pd.read_csv(curpath_stats)
    else:
        flipped_z_data, flipped_z_eyes_stats, threshoded_data = check_depth_orientation(input_data)
        flipped_z_data.astype(np.float32).tofile(curpath_data)
        flipped_z_eyes_stats.to_csv(curpath_stats)


    aligned_data, aligned_eyes_stats = None, None
    curpath_data = os.path.join(TMP_PATH, create_filename_with_shape(filepath, tuple([1239,320,320]), prefix='aligneddata'))
    curpath_stats = os.path.join(TMP_PATH, "aligned_eyes_stats_fish215_32bit_320x320x1239.csv")
    if os.path.exists(curpath_data) and os.path.exists(curpath_stats):
        aligned_data = open_data(curpath_data)
        aligned_eyes_stats = pd.read_csv(curpath_stats)
    else:
        aligned_data, aligned_eyes_stats = align_eyes_centroids(flipped_z_data, eyes_stats=flipped_z_eyes_stats)
        aligned_data.astype(np.float32).tofile(curpath_data)
        aligned_eyes_stats.to_csv(curpath_stats)

    flipped_y_data, flipped_y_eyes_stats = None, None
    curpath_data = os.path.join(TMP_PATH, create_filename_with_shape(filepath, tuple([1239,320,320]), prefix='flippedydata'))
    curpath_stats = os.path.join(TMP_PATH, "flipped_y_eyes_stats_fish215_32bit_320x320x1239.csv")
    if os.path.exists(curpath_data) and os.path.exists(curpath_stats):
        flipped_y_data = open_data(curpath_data)
        flipped_y_eyes_stats = pd.read_csv(curpath_stats)
    else:
        flipped_y_data, flipped_y_eyes_stats = check_vertical_orientation(aligned_data, eyes_stats=aligned_eyes_stats)
        flipped_y_data.astype(np.float32).tofile(curpath_data)
        flipped_y_eyes_stats.to_csv(curpath_stats)

    tail_aligned_data = align_tail_part(flipped_y_data,
                                        landmark_tail_idx_frac=landmark_tail_idx_frac, \
                                        spinal_angle=spinal_angle,
                                        interp_order=interp_order)

    return tail_aligned_data

@timing
def align_fish_by_eyes_tail(input_data, landmark_tail_idx_frac=0.56383, spinal_angle=0.12347, interp_order=3):
    flipped_z_data, flipped_z_eyes_stats = None, None
    curpath_data = os.path.join(TMP_PATH, create_filename_with_shape(filepath, tuple([2478,640,640]), prefix='flippedzdata'))
    curpath_stats = os.path.join(TMP_PATH, "flipped_z_eyes_stats_fish215_32bit_640x640x2478.csv")
    if os.path.exists(curpath_data) and os.path.exists(curpath_stats):
        flipped_z_data = open_data(curpath_data)
        print "flipped_z_data = %s" % str(flipped_z_data.shape)
        plt.imshow(flipped_z_data[1500], cmap='gray')
        plt.show()
        flipped_z_eyes_stats = pd.read_csv(curpath_stats)
        print flipped_z_eyes_stats
    else:
        flipped_z_data, flipped_z_eyes_stats, threshoded_data = check_depth_orientation(input_data)
        flipped_z_data.astype(np.float32).tofile(curpath_data)
        flipped_z_eyes_stats.to_csv(curpath_stats)


    aligned_data, aligned_eyes_stats = None, None
    curpath_data = os.path.join(TMP_PATH, create_filename_with_shape(filepath, tuple([2478,640,640]), prefix='aligneddata'))
    curpath_stats = os.path.join(TMP_PATH, "aligned_eyes_stats_fish215_32bit_640x640x2478.csv")
    if os.path.exists(curpath_data) and os.path.exists(curpath_stats):
        aligned_data = open_data(curpath_data)
        print "aligned_data = %s" % str(aligned_data.shape)
        plt.imshow(aligned_data[1500], cmap='gray')
        plt.show()
        aligned_eyes_stats = pd.read_csv(curpath_stats)
        print aligned_eyes_stats
    else:
        aligned_data, aligned_eyes_stats = align_eyes_centroids(flipped_z_data, eyes_stats=flipped_z_eyes_stats)
        aligned_data.astype(np.float32).tofile(curpath_data)
        aligned_eyes_stats.to_csv(curpath_stats)

    flipped_y_data, flipped_y_eyes_stats = None, None
    curpath_data = os.path.join(TMP_PATH, create_filename_with_shape(filepath, tuple([2478,640,640]), prefix='flippedydata'))
    curpath_stats = os.path.join(TMP_PATH, "flipped_y_eyes_stats_fish215_32bit_640x640x2478.csv")
    if os.path.exists(curpath_data) and os.path.exists(curpath_stats):
        flipped_y_data = open_data(curpath_data)
        print "flipped_y_data = %s" % str(flipped_y_data.shape)
        plt.imshow(flipped_y_data[1500], cmap='gray')
        plt.show()
        flipped_y_eyes_stats = pd.read_csv(curpath_stats)
        print flipped_y_eyes_stats
    else:
        flipped_y_data, flipped_y_eyes_stats = check_vertical_orientation(aligned_data, eyes_stats=aligned_eyes_stats)
        flipped_y_data.astype(np.float32).tofile(curpath_data)
        flipped_y_eyes_stats.to_csv(curpath_stats)

    tail_aligned_data = align_tail_part(flipped_y_data,
                                        landmark_tail_idx_frac=landmark_tail_idx_frac, \
                                        spinal_angle=spinal_angle,
                                        interp_order=interp_order)

    return tail_aligned_data

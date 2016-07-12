import os
import gc
import numpy as np
import pandas as pd
from operator import attrgetter
from multiprocessing import Process, Pool
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import median_filter
from scipy.ndimage.morphology import binary_dilation, binary_closing, binary_fill_holes, binary_opening
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.interpolation import zoom, rotate
from skimage.filters import threshold_otsu, threshold_li
from skimage.morphology import disk, white_tophat
from skimage.measure import regionprops
from ..segmentation.eyes import eyes_statistics, eyes_zrange
from .morphology import object_counter, gather_statistics, extract_largest_area_data, stats_at_slice, rotate_stats, flip_stats
from .misc import timing, Timer
from .io import get_path_by_name, INPUT_DIR, OUTPUT_DIR, LSDF_DIR, create_filename_with_shape, parse_filename, create_raw_stack

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
    x, y, z = np.arange(dims[0], dtype=np.int32), np.arange(dims[1], dtype=np.int32), np.arange(dims[2], dtype=np.int32)
    interp_data = interp3(x, y, z, data, z_coordinates, y_coordinates, x_coordinates, order=interp_order)

    print_available_ram()
    del data, x, y, z, x_coordinates, y_coordinates, z_coordinates
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
    largest_volume_region, largest_volume_region_bbox, _ = extract_largest_area_data(input_data, binary_stack_stats, bb_side_offset=bb_side_offset, force_bbox_fit=False, pad_data=True)

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
        min_area = 4000
    else:
        min_area = 12000

    return min_area, min_sphericity

def check_depth_orientation(data, data_label=None, is_tail_fisrt=True):
    print_available_ram()

    print 'Depth orientation correction - Finding eyes...'
    stack_statistics, thresholded_data = gather_statistics(data)
    min_area, min_sphericity = _calc_approx_eyes_params(data.shape)
    eyes_stats = eyes_statistics(stack_statistics, min_area=min_area, min_sphericity=min_sphericity)

    print_available_ram()

    print 'Depth orientation correction - Reversing data and stats z-direction if needed...'
    flipped_data, flipped = _flip_z(data, eyes_stats, is_tail_fisrt=is_tail_fisrt)

    flipped_data_label, flipped_label = None, None
    if data_label is not None:
        flipped_data_label, flipped_label = _flip_z(data_label, eyes_stats, is_tail_fisrt=is_tail_fisrt)

    del thresholded_data, data

    if data_label is not None:
        del data_label

    print_available_ram()

    if flipped:
        eyes_stats = flip_stats(eyes_stats, flipped_data.shape, axes=(0,))

    print_available_ram()

    return flipped_data, flipped_data_label, eyes_stats


def check_vertical_orientation(data, data_label=None, eyes_stats=None):
    if eyes_stats is None:
        print 'Vectical orientation correction - Finding eyes...'
        stack_statistics, thresholded_data = gather_statistics(data)
        min_area, min_sphericity = _calc_approx_eyes_params(data.shape)
        eyes_stats = eyes_statistics(stack_statistics, min_area=min_area, min_sphericity=min_sphericity)
        del thresholded_data

    eye_c = np.round([eyes_stats['com_x'].mean(), eyes_stats['com_y'].mean(), eyes_stats['com_z'].mean()]).astype(np.int32)
    avg_eye_size = np.round(np.mean([eyes_stats['bb_width'].mean(), eyes_stats['bb_height'].mean(), eyes_stats['bb_depth'].mean()]))

    #shift from eye's to tail direction
    z_shift = avg_eye_size * 2
    head_slice_idx = eye_c[2] - z_shift

    print 'Vectical orientation correction - Analyzing head region slice #%d...' % head_slice_idx
    flipped_data, flipped = _flip_y(data, head_slice_idx)

    flipped_data_label = None
    if data_label is not None:
        flipped_data_label = data_label[:,::-1,:] if flipped else data_label

    del data
    if data_label is not None:
        del data_label

    if flipped:
        eyes_stats = flip_stats(eyes_stats, flipped_data.shape, axes=(1,))

    return flipped_data, flipped_data_label, eyes_stats

def _align_by_eyes_centroids(data, centroids, data_label=None, interp_order=3):
    dims = data.shape
    eye_l, eye_r = centroids

    eye_c = np.array([(lv + rv)/2. for lv,rv in zip(eye_l, eye_r)])
    p = np.array([eye_l, eye_r])
    p_s = p - eye_c

    #theta = np.arccos(p_s[0][0]/np.sqrt(p_s[0].dot(p_s[0])))
    x_axis_vec = np.array([float(np.sign(p_s[0][0])),0.,0.])
    theta = np.arccos(p_s[0].dot(x_axis_vec)/np.sqrt(x_axis_vec.dot(x_axis_vec) * p_s[0].dot(p_s[0])))

    rot_axis = np.cross(p_s[0], x_axis_vec)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    print 'theta eyes = %f' % np.rad2deg(theta)
    print 'rot_axis eyes = %s' % str(rot_axis)
    print 'x_axis_vec eyes = %s' % str(x_axis_vec)
    print 'vec eyes = %s' % str(p_s[0])

    interp_data, R, origin_point = rotate_around_vector(data, eye_c, rot_axis, -theta, interp_order=interp_order)

    interp_data_label = None
    if data_label is not None:
        interp_data_label, _, _ = rotate_around_vector(data_label, eye_c, rot_axis, -theta, interp_order=0)

    return interp_data, interp_data_label, R, origin_point

def align_eyes_centroids(data, data_label=None, eyes_stats=None):
    print_available_ram()

    if eyes_stats is None:
        print 'Aligning of eyes\' centroids - Finding eyes...'
        stack_statistics, thresholded_data = gather_statistics(data)
        min_area, min_sphericity = _calc_approx_eyes_params(data.shape)
        eyes_stats = eyes_statistics(stack_statistics, min_area=min_area, min_sphericity=min_sphericity)
        del thresholded_data

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
    aligned_data, aligned_data_label, R, origin_point  = \
                                            _align_by_eyes_centroids(data, \
                                                                     eyes_coms, \
                                                                     data_label=data_label)

    del data

    if data_label is not None:
        del data_label

    print_available_ram()

    aligned_eyes_stats = rotate_stats(eyes_stats, R, origin_point)

    del R

    print_available_ram()

    return aligned_data, aligned_data_label, aligned_eyes_stats

def align_tail_part(input_data, input_data_label=None, landmark_tail_idx_frac=0.56383, spinal_angle=0.12347, interp_order=3, bb_side_offset=20):
    print 'Aligning of tail part - Extracting z-bounded data...'
    data_type = input_data.dtype

    data_label_type = None
    if input_data_label is not None:
        data_label_type = input_data_label.dtype

    binarized_stack, bbox, eyes_stats = binarizator(input_data)
    binary_stack_stats, thresholded_data = object_counter(binarized_stack)
    largest_volume_region, _, _ = \
                    extract_largest_area_data(input_data, binary_stack_stats, bb_side_offset=bb_side_offset, \
                                              force_bbox_fit=False, pad_data=True, extract_axes=(0,), \
                                              force_positiveness=False)

    del thresholded_data, binarized_stack

    largest_volume_region_label = None
    if input_data_label is not None:
        largest_volume_region_label, _, _ = \
                        extract_largest_area_data(input_data_label, binary_stack_stats, bb_side_offset=bb_side_offset, \
                                                  force_bbox_fit=False, pad_data=True, extract_axes=(0,), \
                                                  force_positiveness=False)
        del input_data_label

    print 'Aligning of tail part - Finding eyes...'
    stack_statistics, inversed_thresholded_data = gather_statistics(largest_volume_region)
    min_area, min_sphericity = _calc_approx_eyes_params(input_data.shape)
    eyes_stats = eyes_statistics(stack_statistics, min_area=min_area, min_sphericity=min_sphericity)

    del inversed_thresholded_data

    print 'Aligning of tail part - Calculating eyes\' center, landmark point...'
    eye_c = np.round([eyes_stats['com_x'].mean(), eyes_stats['com_y'].mean(), eyes_stats['com_z'].mean()]).astype(np.int32)

    ext_vol_len = largest_volume_region.shape[0]
    eye1_idx_frac, eye2_idx_frac = eyes_stats['com_z'].values[0] / float(ext_vol_len),\
                                   eyes_stats['com_z'].values[1] / float(ext_vol_len)

    #Approx. position of landmark on the tail part
    landmark_tail_idx = int(ext_vol_len * landmark_tail_idx_frac)
    lm_slice_stats, lm_labeled_slice = stats_at_slice(largest_volume_region, landmark_tail_idx)

    #z-axis
    z_axis_vec = np.array([0., 0., -1.])

    #vector from eyes center along tail to the landmark point
    tail_com_y, tail_com_z = lm_slice_stats['com_z'].values[0] - eye_c[1], landmark_tail_idx - eye_c[2]
    spinal_vec = np.array([0, tail_com_y, tail_com_z])

    #cross product of z-axis and the vector to produce orthogonal vector of rotation
    rot_axis = np.cross(z_axis_vec, spinal_vec)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    #angle between the vector and z-axis
    theta = np.arccos(z_axis_vec.dot(spinal_vec)/(np.sqrt(z_axis_vec.dot(z_axis_vec)) * np.sqrt(spinal_vec.dot(spinal_vec))))

    #calculate an angle value to aling the vector with the reference vector of spinal_angle rad with z-axis
    if tail_com_y < 0:
        theta = (theta - spinal_angle) if theta > spinal_angle else -(spinal_angle - theta)
    else:
        theta = theta + spinal_angle

    print 'Aligning of tail part - Rotating data around %s vector by %f degree...' % (str(rot_axis), np.rad2deg(-theta))
    data_rotated, _, _ = rotate_around_vector(largest_volume_region, eye_c, \
            rot_axis, theta, interp_order=interp_order)

    del largest_volume_region

    data_label_rotated = None
    if largest_volume_region_label is not None:
        data_label_rotated, _, _ = rotate_around_vector(largest_volume_region_label, eye_c, \
            rot_axis, theta, interp_order=0)

        del largest_volume_region_label


    print 'Aligning of tail part - Extracting aligned data...'
    data_rotated_binarized_stack, _, _ = binarizator(data_rotated)
    data_rotated_binary_stack_stats, thresholded_stack = object_counter(data_rotated_binarized_stack)

    largest_data_rotated_region, _, extration_bbox  = \
                extract_largest_area_data(data_rotated, \
                                          data_rotated_binary_stack_stats, \
                                          bb_side_offset=bb_side_offset, \
                                          force_bbox_fit=False, \
                                          pad_data=True, \
                                          force_positiveness=False)

    del data_rotated, thresholded_stack, data_rotated_binarized_stack

    largest_data_label_rotated_region = None
    if data_label_rotated is not None:
        largest_data_label_rotated_region, _, _  = \
                        extract_largest_area_data(data_label_rotated, \
                                                  data_rotated_binary_stack_stats, \
                                                  bb_side_offset=bb_side_offset, \
                                                  force_bbox_fit=False, \
                                                  pad_data=True, \
                                                  force_positiveness=False)

        del data_label_rotated

    largest_data_rotated_region = largest_data_rotated_region.astype(data_type)

    if largest_data_label_rotated_region is not None:
        largest_data_label_rotated_region = largest_data_label_rotated_region.astype(data_label_type)

    return largest_data_rotated_region, largest_data_label_rotated_region, extration_bbox

TMP_PATH = "C:\\Users\\Administrator\\Documents\\tmp"
#filepath = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish200\\fish200_rotated_32bit_286x286x1235.raw"
#from modules.tools.io import create_filename_with_shape
#filled_y_data.astype(np.float32).tofile(os.path.join(TMP_PATH, create_filename_with_shape(filepath, filled_y_data.shape, prefix='stage3')))
#aligned_data.astype(np.float32).tofile(os.path.join(TMP_PATH, create_filename_with_shape(filepath, aligned_data.shape, prefix='stage2')))
#tail_aligned_data.astype(np.float32).tofile(os.path.join(TMP_PATH, create_filename_with_shape(filepath, tail_aligned_data.shape, prefix='stage4')))
filepath = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish204\\fish204_32bit_631x631x1992.raw"
#filepath = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish204\\fish204_32bit_315x315x996.raw"
#filepath = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish243\\fish243_32bit_320x320x996.raw"
#filepath = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish215\\fish215_32bit_320x320x1239.raw"
#filepath = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish200\\fish200_rotated_32bit_286x286x1235.raw"
from .io import open_data, create_filename_with_shape, get_filename
from .misc import print_available_ram
#import matplotlib.pyplot as plt

@timing
def align_fish_by_eyes_tail___old(input_data, landmark_tail_idx_frac=0.56383, spinal_angle=0.12347, interp_order=3):
    flipped_z_data, flipped_z_eyes_stats = None, None
    curpath_data = os.path.join(TMP_PATH, create_filename_with_shape(filepath, tuple([1239,320,320]), prefix='flippedzdata'))
    curpath_stats = os.path.join(TMP_PATH, "flipped_z_eyes_stats_fish215_32bit_320x320x1239.csv")
    if os.path.exists(curpath_data) and os.path.exists(curpath_stats):
        flipped_z_data = open_data(curpath_data)
        flipped_z_eyes_stats = pd.read_csv(curpath_stats)
    else:
        flipped_z_data, flipped_z_eyes_stats = check_depth_orientation(input_data)
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

    del aligned_data, aligned_eyes_stats

    tail_aligned_data = align_tail_part(flipped_y_data,
                                        landmark_tail_idx_frac=landmark_tail_idx_frac, \
                                        spinal_angle=spinal_angle,
                                        interp_order=interp_order)

    del flipped_y_data, flipped_y_eyes_stats

    return tail_aligned_data

@timing
def align_fish_by_eyes_tail___(input_data, landmark_tail_idx_frac=0.56383, spinal_angle=0.12347, interp_order=3):
    shape = tuple([1992,631,631])
    #shape = tuple([996,315,315])
    #shape = tuple([996,320,320])
    #shape = tuple([1239,320,320])
    #shape = tuple([1235,286,286])

    fn = get_filename(filepath)

    flipped_z_data, flipped_z_eyes_stats = None, None
    curpath_data = os.path.join(TMP_PATH, create_filename_with_shape(filepath, shape, prefix='flippedzdata'))
    curpath_stats = os.path.join(TMP_PATH, "flipped_z_eyes_stats_%s.csv" % fn)
    if os.path.exists(curpath_data) and os.path.exists(curpath_stats):
        flipped_z_data = open_data(curpath_data)
        print "flipped_z_data = %s" % str(flipped_z_data.shape)
        plt.imshow(flipped_z_data[400], cmap='gray')
        plt.show()
        flipped_z_eyes_stats = pd.read_csv(curpath_stats)
        print flipped_z_eyes_stats
    else:
        flipped_z_data, flipped_z_eyes_stats = check_depth_orientation(input_data)
        flipped_z_data.astype(np.float32).tofile(curpath_data)
        flipped_z_eyes_stats.to_csv(curpath_stats)


    aligned_data, aligned_eyes_stats = None, None
    curpath_data = os.path.join(TMP_PATH, create_filename_with_shape(filepath, shape, prefix='aligneddata'))
    curpath_stats = os.path.join(TMP_PATH, "aligned_eyes_stats_%s.csv" % fn)
    if os.path.exists(curpath_data) and os.path.exists(curpath_stats):
        aligned_data = open_data(curpath_data)
        print "aligned_data = %s" % str(aligned_data.shape)
        plt.imshow(aligned_data[400], cmap='gray')
        plt.show()
        aligned_eyes_stats = pd.read_csv(curpath_stats)
        print aligned_eyes_stats
    else:
        aligned_data, aligned_eyes_stats = align_eyes_centroids(flipped_z_data, eyes_stats=flipped_z_eyes_stats)
        aligned_data.astype(np.float32).tofile(curpath_data)
        aligned_eyes_stats.to_csv(curpath_stats)

    del flipped_z_data, flipped_z_eyes_stats

    flipped_y_data, flipped_y_eyes_stats = None, None
    curpath_data = os.path.join(TMP_PATH, create_filename_with_shape(filepath, shape, prefix='flippedydata'))
    curpath_stats = os.path.join(TMP_PATH, "flipped_y_eyes_stats_%s.csv" % fn)
    if os.path.exists(curpath_data) and os.path.exists(curpath_stats):
        flipped_y_data = open_data(curpath_data)
        print "flipped_y_data = %s" % str(flipped_y_data.shape)
        plt.imshow(flipped_y_data[400], cmap='gray')
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

@timing
def align_fish_by_eyes_tail(input_data, input_data_label=None, landmark_tail_idx_frac=0.56383, spinal_angle=0.12347, interp_order=3, bb_side_offset=20):
    flipped_z_data, flipped_z_data_label, flipped_z_eyes_stats = \
        check_depth_orientation(input_data, data_label=input_data_label)


    print 'flipped_z_data_label is None = %s' % str(flipped_z_data_label is None)

    aligned_data, aligned_data_label, aligned_eyes_stats = \
        align_eyes_centroids(flipped_z_data, data_label=flipped_z_data_label, eyes_stats=flipped_z_eyes_stats)

    print 'aligned_data_label is None = %s' % str(aligned_data_label is None)

    del flipped_z_data, flipped_z_eyes_stats

    if flipped_z_data_label is not None:
        del flipped_z_data_label

    flipped_y_data, flipped_y_data_label, flipped_y_eyes_stats = \
                                            check_vertical_orientation(aligned_data, \
                                                                       data_label=aligned_data_label, \
                                                                       eyes_stats=aligned_eyes_stats)

    print 'flipped_y_data_label is None = %s' % str(flipped_y_data_label is None)

    del aligned_data, aligned_eyes_stats

    if aligned_data_label is not None:
        del aligned_data_label

    tail_aligned_data, tail_aligned_data_label, extration_bbox = \
                                align_tail_part(flipped_y_data, \
                                                input_data_label=flipped_y_data_label, \
                                                landmark_tail_idx_frac=landmark_tail_idx_frac, \
                                                spinal_angle=spinal_angle, \
                                                interp_order=interp_order, \
                                                bb_side_offset=bb_side_offset)

    print 'tail_aligned_data_label is None = %s' % str(tail_aligned_data_label is None)

    del flipped_y_data, flipped_y_eyes_stats

    if flipped_y_data_label is not None:
        del flipped_y_data_label

    return tail_aligned_data, tail_aligned_data_label, extration_bbox

def convert_fish(fish_number):
    print '----Converting fish #%d' % fish_number
    dir_path = os.path.join(LSDF_DIR, 'grif', 'ANKA_data', '2014', 'XRegioMay2014', 'tomography', 'Phenotyping', 'Recon', 'fish%d', 'Complete', 'Corr')
    dir_path = dir_path % fish_number
    out_path = os.path.join(LSDF_DIR, 'grif', 'Phenotype_medaka', 'Originals', '%s')

    raw_data_stack = create_raw_stack(dir_path, "fish%d_proj_" % fish_number)

    print '----Raw stack creating fish #%d' % fish_number
    raw_data_stack.tofile(out_path % ("fish%d_32bit_%dx%dx%d.raw" % \
            (fish_number, raw_data_stack.shape[2], raw_data_stack.shape[1], raw_data_stack.shape[0])))
    del raw_data_stack

def convert_fish_in_parallel(fish_num_array, core=4):
    t = Timer()

    p = Pool(core)
    p.map(convert_fish, fish_num_array)

    t.elapsed('Fish converting')

def zoom_chunk_fishes(args):
    for arg in args:
        zoom_rotate(arg)

def zoom_in_parallel(fish_num_array, input_dir, output_dir, core=2):
    t = Timer()

    args = []
    for fish_num in fish_num_array:
        args.append(tuple([get_path_by_name(fish_num, input_dir), output_dir]))

    processes = [Process(target=zoom_rotate, args=(ip,op,)) for ip,op in args]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    t.elapsed('Zooming global')

def align_fishes(fish_num_array, input_dir, output_dir):
    for fish_num in fish_num_array:
        t = Timer()

        print 'Aligning started fish%d...' % fish_num

        input_path = get_path_by_name(fish_num, input_dir)

        print "Input: %s" % input_path

        input_data = open_data(input_path)
        aligned_data, aligned_data_label, extration_bbox  = align_fish_by_eyes_tail(input_data)

        name, bits, size, ext = parse_filename(input_path)
        output_file = create_filename_with_shape(input_path, aligned_data.shape, prefix="aligned")

        output_path = os.path.join(output_dir, output_file)
        print 'Output: %s' % output_path

        aligned_data.astype('float%d' % bits).tofile(output_path)

        del input_data, aligned_data

        t.elapsed('Aligning')

def zoom_fishes(fish_num_array, input_dir, output_dir):
    for fish_num in fish_num_array:

        t = Timer()

        print 'Zooming started fish%d...' % fish_num

        input_path = get_path_by_name(fish_num, input_dir)

        print "Input: %s" % input_path

        input_data = open_data(input_path)
        zoomed_data = zoom(input_data, 0.5, order=3)

        name, bits, size, ext = parse_filename(input_path)
        output_file = create_filename_with_shape(input_path, zoomed_data.shape)

        output_path = os.path.join(output_dir, output_file)

        print 'Output: %s' % output_path

        zoomed_data.astype('float%d' % bits).tofile(output_path)

        t.elapsed('Zooming')

def zoom_rotate(input_path, output_path, rotate_angle=0, rot_axis='z', in_folder=False):
    t = Timer()

    print "Input: %s" % input_path
    print "Output: %s" % output_path

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

    if in_folder:
        output_path = os.path.join(output_path, name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    output_path = os.path.join(output_path, output_file)
    print 'Output will be: %s' % output_path

    rotated_data.tofile(output_path)

    t.elapsed('Zoom and rotation: %s' % input_path)

def get_fish_folder(fish_num, zoom_level=2):
    return os.path.join(INPUT_DIR, 'fish%d' % fish_num, '@%d' % zoom_level)

def get_aligned_fish_folder(fish_num, zoom_level=2):
    return os.path.join(OUTPUT_DIR, 'Aligned', 'fish%d' % fish_num, '@%d' % zoom_level)

def get_fish_project_folder(fish_num):
    return os.path.join(OUTPUT_DIR, 'Segmentation', 'fish%d' % fish_num)

def get_fish_path(fish_num, zoom_level=2, isLabel=False):
    req_path = get_path_by_name(fish_num, os.path.join(INPUT_DIR, 'fish%d' % fish_num, '@%d' % zoom_level), isFindLabels=isLabel)

    if req_path is None:
        downsampled_data_path = None

        for zl in [i for i in [1,2,4,8] if i < zoom_level]: #[1,2,4,8] - zoom levels
            prev_zoom_level_data_path = get_path_by_name(fish_num, \
                                                  get_fish_folder(fish_num, zoom_level=zl), \
                                                  isFindLabels=isLabel)
            if prev_zoom_level_data_path is not None:
                downsampled_data_path = downsample_data(prev_zoom_level_data_path, \
                                                        get_fish_folder(fish_num, zoom_level=zoom_level), \
                                                        zoom_in_level=zoom_level/zl, \
                                                        order=0 if isLabel else 3)
                break

        if downsampled_data_path is not None:
            return downsampled_data_path
        else:
            return None
    else:
        return req_path

def get_aligned_fish_paths(fish_num, zoom_level=2, min_zoom_level=2):
    data_req_path = get_path_by_name(fish_num, \
                                    get_aligned_fish_folder(fish_num, zoom_level=zoom_level))
    data_label_req_path = get_path_by_name(fish_num, \
                                    get_aligned_fish_folder(fish_num, zoom_level=zoom_level), \
                                    isFindLabels=True)

    print '###########data_req_path = %s' % data_req_path
    print '###########data_label_req_path = %s' % data_label_req_path

    if data_req_path is None:
        downsampled_data_path, downsampled_data_label_path = None, None

        for zl in [i for i in [1,2,4,8] if i <= zoom_level]:
            prev_zoom_level_data_path = get_path_by_name(fish_num, \
                                                         get_aligned_fish_folder(fish_num, zoom_level=zl))

            prev_zoom_level_data_label_path = get_path_by_name(fish_num, \
                                                                get_aligned_fish_folder(fish_num, zoom_level=zl), \
                                                                isFindLabels=True)

            print 'prev_zoom_level_data_path = %s' % prev_zoom_level_data_path
            print 'prev_zoom_level_data_label_path = %s' % prev_zoom_level_data_label_path


            if prev_zoom_level_data_path is not None:
                downsampled_data_path = downsample_data(prev_zoom_level_data_path, \
                                                        get_aligned_fish_folder(fish_num, zoom_level=zoom_level), \
                                                        zoom_in_level=zoom_level/zl, \
                                                        order=3)
                print 'downsampled_data_path = %s' % downsampled_data_path

                if prev_zoom_level_data_label_path is not None:
                    downsampled_data_label_path = downsample_data(prev_zoom_level_data_label_path, \
                                                                  get_aligned_fish_folder(fish_num, zoom_level=zoom_level), \
                                                                  zoom_in_level=zoom_level/zl, \
                                                                  order=0)
                    print 'downsampled_data_label_path = %s' % downsampled_data_label_path

                break

        if downsampled_data_path is not None:
            return downsampled_data_path, downsampled_data_label_path
        else:
            produce_aligned_fish(fish_num, min_zoom_level=min_zoom_level)
            return get_aligned_fish_paths(fish_num, zoom_level=zoom_level, min_zoom_level=min_zoom_level)
    else:
        return data_req_path, data_label_req_path

def get_aligned_fish_path(fish_num, zoom_level=2, isLabel=False, min_zoom_level=2):
    req_path = get_path_by_name(fish_num, \
                                get_aligned_fish_folder(fish_num, zoom_level=zoom_level), \
                                isFindLabels=isLabel)
    if req_path is None:
        downsampled_data_path = None
        for zl in [i for i in [1,2,4,8] if i <= zoom_level]:
            prev_zoom_level_data_path = get_path_by_name(fish_num, \
                                                         get_aligned_fish_folder(fish_num, zoom_level=zl), \
                                                         isFindLabels=isLabel)
            if prev_zoom_level_data_path is not None:
                downsampled_data_path = downsample_data(prev_zoom_level_data_path, \
                                                        get_aligned_fish_folder(fish_num, zoom_level=zoom_level), \
                                                        zoom_in_level=zoom_level/zl, \
                                                        order=0 if isLabel else 3)
                break

        if downsampled_data_path is not None:
            return downsampled_data_path
        else:
            produce_aligned_fish(fish_num, min_zoom_level=min_zoom_level)
            get_aligned_fish_path(fish_num, zoom_level=zoom_level, isLabel=isLabel)
    else:
        return req_path

def downsample_data(input_path, output_path, zoom_in_level=2, order=3):
    t = Timer()

    print "Input: %s" % input_path
    print "Output: %s" % output_path

    input_data = open_data(input_path)

    print 'Zooming started...'
    zoomed_data = zoom(input_data, 1./zoom_in_level, order=order)
    print 'Normal data shape = %s, Zoomed data shape = %s' % (str(input_data.shape), str(zoomed_data.shape))

    name, bits, size, ext = parse_filename(input_path)
    output_file = create_filename_with_shape(input_path, zoomed_data.shape)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path = os.path.join(output_path, output_file)
    print 'Output will be: %s' % output_path

    zoomed_data.tofile(output_path)

    t.elapsed('Zoom by %f: %s' % (1./zoom_in_level, input_path))

    return output_path

def scaling_aligning():
    fish_num_array = np.array([200, 204, 215, 223, 226, 228, 230, 231, 233, 238, 243])

    input_dir = os.path.join(LSDF_DIR, 'grif', 'Phenotype_medaka', 'Misc', 'Originals')
    output_zoom_dir = os.path.join(LSDF_DIR, 'grif', 'Phenotype_medaka', 'Misc', 'Originals_scaled')
    output_align_dir = os.path.join(LSDF_DIR, 'grif', 'Phenotype_medaka', 'Misc', 'Originals_aligned')

    zoom_fishes(fish_num_array, input_dir, output_zoom_dir)

    fish_num_array = np.array([230, 231, 233, 238, 243])
    align_fishes(fish_num_array, output_zoom_dir, output_align_dir)

def produce_aligned_fish(fish_num, min_zoom_level=2):
    print "--Produce aligned fish%d" % fish_num
    non_aligned_data_path, non_aligned_data_label_path = \
                    get_fish_path(fish_num, zoom_level=min_zoom_level), \
                    get_fish_path(fish_num, zoom_level=min_zoom_level, isLabel=True)

    non_aligned_input_data = open_data(non_aligned_data_path)

    non_aligned_input_data_label = None
    if non_aligned_data_label_path is not None:
        non_aligned_input_data_label = open_data(non_aligned_data_label_path)

    aligned_data, aligned_data_label, extration_bbox = \
                align_fish_by_eyes_tail(non_aligned_input_data, \
                                        input_data_label=non_aligned_input_data_label)

    name, bits, size, ext = parse_filename(non_aligned_data_path)
    output_file = create_filename_with_shape(non_aligned_data_path, \
                                             aligned_data.shape, \
                                             prefix="aligned")

    aligned_data = aligned_data.astype('float%d' % bits)

    output_label_file = None
    if non_aligned_data_label_path is not None:
        name_label, bits_label, size_label, ext_label = parse_filename(non_aligned_data_label_path)
        output_label_file = create_filename_with_shape(non_aligned_data_label_path, \
                                                       aligned_data_label.shape, \
                                                       prefix="aligned")
        if bits_label != 8:
            raise ValueError('Label data should be 8-bit.')

        aligned_data_label = aligned_data_label.astype('uint%d' % bits_label)

    output_path = get_aligned_fish_folder(fish_num, zoom_level=min_zoom_level)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_data_path = os.path.join(output_path, output_file)
    aligned_data.tofile(output_data_path)

    if output_label_file is not None:
        output_label_path = os.path.join(output_path, output_label_file)
        aligned_data_label.tofile(output_label_path)

def get_aligned_fish(fish_num, zoom_level=2, min_zoom_level=2):
    input_aligned_data_path, input_aligned_data_label_path = get_aligned_fish_paths(fish_num, zoom_level=zoom_level, min_zoom_level=min_zoom_level)

    print 'zoom_level = %d' % zoom_level
    print 'min_zoom_level = %d' % min_zoom_level

    print 'input_aligned_data_path = %s' % str(input_aligned_data_path)
    print 'input_aligned_data_label_path = %s' % str(input_aligned_data_label_path)

    input_aligned_data = open_data(input_aligned_data_path)

    input_aligned_data_label = None
    if input_aligned_data_label_path is not None:
        input_aligned_data_label = open_data(input_aligned_data_label_path)

    return input_aligned_data, input_aligned_data_label

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
from .morphology import object_counter, gather_statistics, extract_largest_area_data, stats_at_slice
from .misc import timing

def binarizator(stack_data, eyes_stats=None, filter_size=6,
        non_zeros_ratio=0.5, tolerance=50, preserve_big_objects=True):
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
                print 'Slice #%d' % slice_idx
                filtered_streaks = white_tophat(thresholded_stack[slice_idx], \
                                                selem=disk(15))
                removed_streaks = thresholded_stack[slice_idx] - filtered_streaks
                thresholded_stack[slice_idx] = \
                        median_filter(removed_streaks, size=(filter_size,filter_size))

    bbox = find_objects(thresholded_stack)

    print "thresholded_stack = %s" % str(thresholded_stack.shape)

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
        if np.all(np.diff(corner_locs) < 0):
            corner_locs, index = corner_locs[::-1], index[::-1]
        return np.interp(interp_locs, corner_locs, index)

    orig_shape = np.asarray(xi).shape
    xi, yi, zi = np.atleast_1d(xi, yi, zi)
    for arr in [xi, yi, zi]:
        arr.shape = -1

    output = np.empty(xi.shape, dtype=float)
    coords = [index_coords(*item) for item in zip([x, y, z], [xi, yi, zi])]

    map_coordinates(v, coords, output=output, **kwargs)

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
                     np.cos(theta)+uz*uz*(1.-np.cos(theta))]])

    return mat

def rotate_around_vector(data, origin_point, rot_axis, angle, interp_order=3):
    dims = data.shape

    R = _get_rot_matrix_arbitrary_axis(rot_axis[0], rot_axis[1], rot_axis[2], angle)

    print 'Rotation around vector - Coords shifting...'
    zv, yv, xv = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing='ij')
    coordinates_translated = np.array([np.ravel(xv - origin_point[0]).T, \
                                       np.ravel(yv - origin_point[1]).T, \
                                       np.ravel(zv - origin_point[2]).T])

    print 'Rotation around vector - Rotating...'
    coordinates_rotated = np.array(R * coordinates_translated)

    print 'Rotation around vector - Coords back shifting...'
    coordinates_rotated[0] = coordinates_rotated[0] + origin_point[0]
    coordinates_rotated[1] = coordinates_rotated[1] + origin_point[1]
    coordinates_rotated[2] = coordinates_rotated[2] + origin_point[2]

    print 'Rotation around vector - Data reshaping...'
    x_coordinates = np.reshape(coordinates_rotated[0,:], dims)
    y_coordinates = np.reshape(coordinates_rotated[1,:], dims)
    z_coordinates = np.reshape(coordinates_rotated[2,:], dims)

    #get the values for your new coordinates
    print 'Rotation around vector - Interpolation in 3D...'
    interp_data = interp3(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), data, \
                          z_coordinates, y_coordinates, x_coordinates, order=interp_order)

    return interp_data

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

    if (data_shape[0]/2.0 > eyes_z_pos):
        return stack_data[::-1,:,:] if is_tail_fisrt else stack_data
    else:
        return stack_data if not is_tail_fisrt else stack_data[::-1,:,:]

def check_depth_orientation(data):
    print 'Depth orientation correction - Finding eyes...'
    stack_statistics, _ = gather_statistics(data)
    eyes_stats = eyes_statistics(stack_statistics)

    print 'Depth orientation correction - Reversing z-direction if needed...'
    data = _flip_z(data, eyes_stats)

    return data

def check_vertical_orientation(data):
    print 'Vectical orientation correction - Finding eyes...'
    stack_statistics, _ = gather_statistics(data)
    eyes_stats = eyes_statistics(stack_statistics)

    eye_c = np.round([eyes_stats['com_x'].mean(), eyes_stats['com_y'].mean(), eyes_stats['com_z'].mean()]).astype(np.int32)
    avg_eye_size = np.round(np.mean([eyes_stats['bb_width'].mean(), eyes_stats['bb_height'].mean(), eyes_stats['bb_depth'].mean()]))

    #shift from eye's to tail direction
    z_shift = avg_eye_size * 2
    head_slice_idx = eye_c[2] - z_shift

    print 'Vectical orientation correction - Analyzing head region slice #%d...' % head_slice_idx
    #z -> y
    #y -> x
    head_slice_stats, head_labeled_slice = stats_at_slice(data, head_slice_idx)

    central_head_part = head_labeled_slice[head_slice_stats['bb_z']:head_slice_stats['bb_z']+head_slice_stats['bb_depth'],\
                                           head_slice_stats['bb_y']:head_slice_stats['bb_y']+head_slice_stats['bb_height']]
    top_part, bottom_part = central_head_part[:central_head_part.shape[0]/2,:], central_head_part[central_head_part.shape[0]/2+1:,:]
    top_part, bottom_part = binary_opening(top_part, iterations=2), binary_opening(bottom_part, iterations=2)
    non_zeros_top, non_zeros_bottom = np.count_nonzero(top_part), np.count_nonzero(bottom_part)
    print 'Vectical orientation correction - Counting non-zeros (top=%d, bottom=%d)...' % (non_zeros_top, non_zeros_bottom)

    print 'Vectical orientation correction - Reversing y-direction if needed...'
    if non_zeros_top < non_zeros_bottom:
        data = data[:,::-1,:]

    return data

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

    interp_data = rotate_around_vector(data, eye_c, rot_axis, -theta, interp_order=interp_order)

    return interp_data

def align_eyes_centroids(data):
    print 'Aligning of eyes\' centroids - Finding eyes...'
    stack_statistics, _ = gather_statistics(data)
    eyes_stats = eyes_statistics(stack_statistics)

    print 'Aligning of eyes\' centroids - Stats extracting...'
    eye1_com, eye2_com = np.array([eyes_stats['com_x'].values[0], eyes_stats['com_y'].values[0], eyes_stats['com_z'].values[0]]), \
                             np.array([eyes_stats['com_x'].values[1], eyes_stats['com_y'].values[1], eyes_stats['com_z'].values[1]])
    eyes_coms = [eye1_com, eye2_com]

    eye1_sbbox, eye2_sbbox = np.array([eyes_stats['bb_width'].values[0], eyes_stats['bb_height'].values[0], eyes_stats['bb_depth'].values[0]]), \
                                 np.array([eyes_stats['bb_width'].values[1], eyes_stats['bb_height'].values[1], eyes_stats['bb_depth'].values[1]])
    eyes_sizes = [eye1_sbbox, eye2_sbbox]

    print 'Aligning of eyes\' centroids - Aligning along x- and y-axes...'
    aligned_data = _align_by_eyes_centroids(data, eyes_coms)

    return aligned_data

def align_tail_part(input_data, landmark_tail_idx_frac=0.56383, spinal_angle=0.12347, interp_order=3):
    print 'Aligning of tail part - Extracting z-bounded data...'
    binarized_stack, bbox, eyes_stats = binarizator(input_data)
    binary_stack_stats, _ = object_counter(binarized_stack)
    largest_volume_region, largest_volume_region_bbox = extract_largest_area_data(input_data, binary_stack_stats, bb_side_offset=50, force_bbox_fit=False, pad_data=True, extact_axes=(0,))

    print 'Aligning of tail part - Finding eyes...'
    stack_statistics, _ = gather_statistics(largest_volume_region)
    eyes_stats = eyes_statistics(stack_statistics)

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
    data_rotated = rotate_around_vector(largest_volume_region, eye_c, rot_axis, theta, interp_order=interp_order)

    print 'Aligning of tail part - Extracting aligned data...'
    data_rotated_binarized_stack, _, _ = binarizator(data_rotated)
    data_rotated_binary_stack_stats, _ = object_counter(data_rotated_binarized_stack)
    largest_data_rotated_region, _ = extract_largest_area_data(data_rotated, data_rotated_binary_stack_stats, bb_side_offset=50, force_bbox_fit=False, pad_data=True)

    return largest_data_rotated_region

@timing
def align_fish_by_eyes_tail(input_data, landmark_tail_idx_frac=0.56383, spinal_angle=0.12347, interp_order=3):
    input_data = check_depth_orientation(input_data)
    input_data = align_eyes_centroids(input_data)
    input_data = check_vertical_orientation(input_data)
    aligned_data = align_tail_part(input_data,
                                   landmark_tail_idx_frac=landmark_tail_idx_frac, \
                                   spinal_angle=spinal_angle,
                                   interp_order=interp_order)
    return aligned_data

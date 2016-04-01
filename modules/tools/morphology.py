import numpy as np
import pandas as pd
import multiprocessing as mp
from operator import attrgetter

from skimage.filters import threshold_otsu, threshold_li
from scipy.ndimage.morphology import binary_closing, binary_fill_holes, generate_binary_structure, binary_dilation
from scipy.ndimage.measurements import label, find_objects, center_of_mass
from scipy.ndimage.filters import median_filter
from skimage.measure import regionprops
from misc import BBox, timing, Timer


_MEASUREMENTS = {
    'Label': 'label',
    'Area': 'area',
    'Perimeter': 'perimeter'
}

_MEASUREMENTS_EXTRA = {
    'Sphericity': 'sphericity',
    'Bounding box X': 'bb_x',
    'Bounding box Y': 'bb_y',
    'Bounding box Z': 'bb_z',
    'Bounding box Width': 'bb_width',
    'Bounding box Height': 'bb_height',
    'Bounding box Depth': 'bb_depth',
    'Center of mass X': 'com_x',
    'Center of mass Y': 'com_y',
    'Center of mass Z': 'com_z'
}

_MEASUREMENTS_EXTRA_2D = {
    'Slice Index': 'slice_idx',
    'Circularity': 'circularity',
    'Center of mass Y': 'com_y',
    'Center of mass Z': 'com_z',
    'Bounding box Y': 'bb_y',
    'Bounding box Z': 'bb_z',
    'Bounding box Height': 'bb_height',
    'Bounding box Depth': 'bb_depth'
}

_MEASUREMENTS_VALS = _MEASUREMENTS.values()

_MEASUREMENTS_EXTRA_VALS = _MEASUREMENTS_EXTRA.values()
_MEASUREMENTS_EXTRA_VALS_2D = _MEASUREMENTS_EXTRA_2D.values()

@timing
def gather_statistics(stack_data, filter_size=1, non_zeros_ratio=0.5, \
                        tolerance=50, is_inverse=True, verbose=False):

    #duplication!!!!!!!!!!!!!!!!!!!!!!!!!!!! FOROM PROCESSING MODULE
    def check_slice_position(slice_idx, num_slices, slice_threshold_ratio=0.2):
        num_slices_threshold = np.ceil(num_slices * slice_threshold_ratio)

        if slice_idx in np.arange(num_slices_threshold) or \
                slice_idx in np.arange(num_slices - num_slices_threshold, num_slices):
            return True

        return False

    thresholded_stack = np.empty_like(stack_data, dtype=np.uint8)
    num_slices = stack_data.shape[0]

    print 'Gathering statistics...'

    for slice_idx in np.arange(num_slices):
        threshold_val = threshold_otsu(stack_data[slice_idx])
        thresholded_stack[slice_idx] = stack_data[slice_idx] < threshold_val if is_inverse \
                                       else stack_data[slice_idx] >= threshold_val
        thresholded_stack[slice_idx] = median_filter(thresholded_stack[slice_idx], size=(1,1))

        if check_slice_position(slice_idx, num_slices):
            #check artifacts
            th_val = threshold_otsu(stack_data[slice_idx])
            checked_slice = stack_data[slice_idx] >= th_val
            dilated_slice = binary_dilation(checked_slice, iterations=5).flatten()
            num_nonzeros = np.count_nonzero(dilated_slice)

            if num_nonzeros > (dilated_slice.size - num_nonzeros) * non_zeros_ratio:
                thresholded_stack[slice_idx] = np.zeros_like(thresholded_stack[slice_idx])
                continue

        if verbose:
            if slice_idx % 100 == 0 or slice_idx == num_slices-1:
                print 'Slice #%d' % slice_idx

    stack_statistics, _ = object_counter(thresholded_stack)

    return stack_statistics, thresholded_stack

def stats_at_slice(stack_data, slice_idx, preserve_big_objects=True):
    threshold_val = threshold_otsu(stack_data[slice_idx])
    thresholded_slice = stack_data[slice_idx] >= threshold_val
    thresholded_slice = median_filter(thresholded_slice, size=(1,1))

    if preserve_big_objects:
        labeled_slice, num_labels = label(thresholded_slice)
        max_area_label = max(regionprops(labeled_slice), key=attrgetter('area')).label
        labeled_slice[labeled_slice != max_area_label] = 0
        labeled_slice[labeled_slice == max_area_label] = 1
        thresholded_slice = labeled_slice

    stack_statistics, labeled_slice = cell_counter(thresholded_slice, slice_index=slice_idx)

    return stack_statistics, labeled_slice

def flip_stats(stats, data_shape, axes=(0,)):
    coms = ['com_z', 'com_y', 'com_x']
    bbox_start = ['bb_z', 'bb_y', 'bb_x']

    for axis in axes:
        axis_max_idx = data_shape[axis] - 1
        stats[[bbox_start[axis], coms[axis]]] = \
                stats[[bbox_start[axis], coms[axis]]].apply(lambda val: axis_max_idx - val)

    return stats

def rotate_stats(stats, rot_mat, rot_point):
    for i, row in stats.iterrows():
        com = np.array([row['com_x'], row['com_y'], row['com_z']])
        com_s = np.matrix(com - rot_point)
        com_rot = np.asarray((rot_mat * com_s.T).T + rot_point)

        bbox_start = np.array([row['bb_x'], row['bb_y'], row['bb_z']])
        bbox_start_s = np.matrix(bbox_start - rot_point)
        bbox_start_rot = np.asarray((rot_mat * bbox_start_s.T).T + rot_point)

        row['com_x'], row['com_y'], row['com_z'] = np.round(com_rot[0]).astype(np.int32)
        row['bb_x'], row['bb_y'], row['bb_z'] =  np.round(bbox_start_rot[0]).astype(np.int32)

    return stats

def _collect_stats(labeled_stack):
    objects_stats = pd.DataFrame(columns=_MEASUREMENTS_VALS)
    for slice_idx in np.arange(labeled_stack.shape[0]):
        for region in regionprops(labeled_stack[slice_idx]):
            objects_stats = objects_stats.append({_measure: region[_measure] \
                                                    for _measure in _MEASUREMENTS_VALS}, \
                                                        ignore_index=True)
    objects_stats = objects_stats.groupby('label', as_index=False).sum()

    return objects_stats

def _collect_callback(result):
    TOTAL_OBJECT_STATS = TOTAL_OBJECT_STATS.append(result)

@timing
def object_counter(stack_binary_data):
    t = Timer()
    print 'Object counting - Labeling...'
    labeled_stack, num_labels = label(stack_binary_data)
    t.elapsed('Object counting - Labeling...')

    t = Timer()
    print 'Object counting - BBoxing...'
    bboxes_labels = [BBox(bb_obj) for bb_obj in find_objects(labeled_stack)]
    t.elapsed('Object counting - BBoxing...')

    t = Timer()
    print 'Object counting - Centers of masses...'
    center_of_mass_labels = center_of_mass(stack_binary_data, labeled_stack, np.arange(1, num_labels+1))
    t.elapsed('Object counting - Centers of masses...')

    objects_stats = pd.DataFrame(columns=_MEASUREMENTS_VALS)
    t = Timer()
    print 'Object counting - Stats gathering...'
    for slice_idx in np.arange(labeled_stack.shape[0]):
        for region in regionprops(labeled_stack[slice_idx]):
            objects_stats = objects_stats.append({_measure: region[_measure] \
                                        for _measure in _MEASUREMENTS_VALS}, \
                                            ignore_index=True)
    t.elapsed('Object counting - Stats gathering...')

    # t = Timer()
    # print 'Object counting - Stats gathering...'
    # processes=2
    # per_chunk = 100
    # chunks = int(np.ceil(labeled_stack.shape[0] / float(per_chunk)))
    # index_sets = np.array_split(np.arange(labeled_stack.shape[0]), chunks)
    # data_sets = np.array_split(labeled_stack, chunks, axis=0)
    #
    # pool = mp.Pool(processes=processes)
    # results = pool.map(_collect_stats, data_sets)
    # print results
    # t.elapsed('Object counting - Stats gathering...')
    #
    # objects_stats = pd.DataFrame(columns=_MEASUREMENTS_VALS)
    # for df in results:
    #     objects_stats = objects_stats.append(df)

    t = Timer()
    print objects_stats.shape
    objects_stats = objects_stats.groupby('label', as_index=False).sum()
    t.elapsed('objects_stats.groupby(\'label\', as_index=False).sum()')

    t = Timer()
    print 'Object counting - Extra stats gathering...'
    for _measure_extra in _MEASUREMENTS_EXTRA_VALS:
        if _measure_extra == 'sphericity':
            objects_stats[_measure_extra] = objects_stats.apply(lambda row: \
                0.0 if row['perimeter'] == 0 else
                    _calc_sphericity(row['area'], row['perimeter']), axis=1)
        elif _measure_extra == 'bb_x':
            objects_stats[_measure_extra] = objects_stats.apply(lambda row: \
                bboxes_labels[int(row['label']) - 1].x, axis=1)
        elif _measure_extra == 'bb_width':
            objects_stats[_measure_extra] = objects_stats.apply(lambda row: \
                bboxes_labels[int(row['label']) - 1].width, axis=1)
        elif _measure_extra == 'bb_y':
            objects_stats[_measure_extra] = objects_stats.apply(lambda row: \
                bboxes_labels[int(row['label']) - 1].y, axis=1)
        elif _measure_extra == 'bb_height':
            objects_stats[_measure_extra] = objects_stats.apply(lambda row: \
                bboxes_labels[int(row['label']) - 1].height, axis=1)
        elif _measure_extra == 'bb_z':
            objects_stats[_measure_extra] = objects_stats.apply(lambda row: \
                bboxes_labels[int(row['label']) - 1].z, axis=1)
        elif _measure_extra == 'bb_depth':
            objects_stats[_measure_extra] = objects_stats.apply(lambda row: \
                bboxes_labels[int(row['label']) - 1].depth, axis=1)
        elif _measure_extra == 'com_x':
            objects_stats[_measure_extra] = objects_stats.apply(lambda row: \
                center_of_mass_labels[int(row['label']) - 1][2], axis=1)
        elif _measure_extra == 'com_y':
            objects_stats[_measure_extra] = objects_stats.apply(lambda row: \
                center_of_mass_labels[int(row['label']) - 1][1], axis=1)
        elif _measure_extra == 'com_z':
            objects_stats[_measure_extra] = objects_stats.apply(lambda row: \
                center_of_mass_labels[int(row['label']) - 1][0], axis=1)

    t.elapsed('Object counting - Extra stats gathering...')

    return objects_stats, labeled_stack

@timing
def cell_counter(slice_binary_data, min_area=0.0, min_circularity=0.0, slice_index=-1):
    print 'Object counting - Labeling...'
    labeled_data, num_labels = label(slice_binary_data)

    if not num_labels:
        return pd.DataFrame(), np.zeros(slice_binary_data.shape)

    print 'Object counting - BBoxing...'
    bboxes_labels = [BBox(bb_obj) for bb_obj in find_objects(labeled_data)]

    print 'Object counting - Centers of masses...'
    center_of_mass_labels = center_of_mass(slice_binary_data, labeled_data, np.arange(1, num_labels+1))

    objects_stats = pd.DataFrame(columns=_MEASUREMENTS_VALS)

    for region in regionprops(labeled_data):
        objects_stats = objects_stats.append({_measure: region[_measure] \
                            for _measure in _MEASUREMENTS_VALS}, \
                                ignore_index=True)

    print 'Object counting - Extra stats gathering...'
    for _measure_extra in _MEASUREMENTS_EXTRA_VALS_2D:
        if _measure_extra == 'circularity':
            objects_stats[_measure_extra] = objects_stats.apply(lambda row: \
                0.0 if row['perimeter'] == 0 else
                    _calc_circularity(row['area'], row['perimeter']), axis=1)
        elif _measure_extra == 'com_y':
            objects_stats[_measure_extra] = objects_stats.apply(lambda row: \
                center_of_mass_labels[int(row['label']) - 1][1], axis=1)
        elif _measure_extra == 'com_z':
            objects_stats[_measure_extra] = objects_stats.apply(lambda row: \
                center_of_mass_labels[int(row['label']) - 1][0], axis=1)
        elif _measure_extra == 'bb_y':
            objects_stats[_measure_extra] = objects_stats.apply(lambda row: \
                bboxes_labels[int(row['label']) - 1].y, axis=1)
        elif _measure_extra == 'bb_height':
            objects_stats[_measure_extra] = objects_stats.apply(lambda row: \
                bboxes_labels[int(row['label']) - 1].height, axis=1)
        elif _measure_extra == 'bb_z':
            objects_stats[_measure_extra] = objects_stats.apply(lambda row: \
                bboxes_labels[int(row['label']) - 1].z, axis=1)
        elif _measure_extra == 'bb_depth':
            objects_stats[_measure_extra] = objects_stats.apply(lambda row: \
                bboxes_labels[int(row['label']) - 1].depth, axis=1)
        elif _measure_extra == 'slice_idx':
            objects_stats[_measure_extra] = slice_index

    filtered_stats = objects_stats

    return filtered_stats, labeled_data

def extract_data_by_label(stack_data, stack_stats, label, bb_side_offset=0, \
                          force_bbox_fit=True, pad_data=False, \
                          extact_axes=(0,1,2), force_positiveness=True):
    filtered_stats = stack_stats[stack_stats['label'] == label].head(1)
    bbox = BBox(filtered_stats.to_dict('records')[0])
    print "extracted_data_by_label = %s" % str(filtered_stats.to_dict('records')[0])

    tuple_bbox = None

    if force_bbox_fit:
        tuple_bbox = bbox.create_tuple(offset=bb_side_offset, \
                                       max_ranges=stack_data.shape, \
                                       force_positiveness=force_positiveness)
    else:
        tuple_bbox = bbox.create_tuple(offset=bb_side_offset, \
                                       force_positiveness=force_positiveness)
    #print 'DATA SHAPE: %s' % str(stack_data.shape)
    if pad_data:
        z_begin_pad, z_end_pad = np.abs(tuple_bbox[0].start) if tuple_bbox[0].start < 0 else 0, \
                                 tuple_bbox[0].stop - stack_data.shape[0] \
                                        if tuple_bbox[0].stop > stack_data.shape[0] else 0
        y_begin_pad, y_end_pad = np.abs(tuple_bbox[1].start) if tuple_bbox[1].start < 0 else 0, \
                                 tuple_bbox[1].stop - stack_data.shape[1] \
                                        if tuple_bbox[1].stop > stack_data.shape[1] else 0
        x_begin_pad, x_end_pad = np.abs(tuple_bbox[2].start) if tuple_bbox[2].start < 0 else 0, \
                                 tuple_bbox[2].stop - stack_data.shape[2] \
                                        if tuple_bbox[2].stop > stack_data.shape[2] else 0
        padding_sides = tuple([tuple([z_begin_pad, z_end_pad]), \
                               tuple([y_begin_pad, y_end_pad]), \
                               tuple([x_begin_pad, x_end_pad])])
        print 'Padding: %s' % str(padding_sides)
        stack_data = np.pad(stack_data, padding_sides, mode='constant')

    def shift_negative_idx(idx):
        start, stop = 0 if idx.start < 0 else idx.start, (idx.stop - idx.start) if idx.start < 0 else idx.stop
        return np.s_[start:stop]

    #print 'PADDED DATA SHAPE: %s' % str(stack_data.shape)
    tuple_bbox = tuple([shift_negative_idx(s) for s in tuple_bbox])
    extration_bbox = tuple([s if i in extact_axes else np.s_[:] for i,s in enumerate(tuple_bbox)])

    #print 'BBOX: %s' % str(tuple_bbox)
    #print 'EXTRACTION BBOX: %s' % str(extration_bbox)

    return stack_data[extration_bbox], tuple_bbox

def extract_largest_area_data(stack_data, stack_stats, bb_side_offset=0, \
                              force_bbox_fit=True, pad_data=False, \
                              extact_axes=(0,1,2), force_positiveness=True):
    filtered_stats = stack_stats.sort(['area'], ascending=False).head(1)

    return extract_data_by_label(stack_data, stack_stats, \
            filtered_stats['label'].values[0], bb_side_offset=bb_side_offset, \
                force_bbox_fit=force_bbox_fit, pad_data=pad_data, \
                    extact_axes=extact_axes, force_positiveness=force_positiveness)

def _calc_sphericity(area, perimeter):
    r = ((3.0 * area) / (4.0 * np.pi)) ** (1.0/3.0)
    return (4.0 * np.pi * (r*r)) / perimeter

def _calc_circularity(area, perimeter):
    return 4.0 * np.pi * area / (perimeter * perimeter)

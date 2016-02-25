import numpy as np
import pandas as pd
from operator import attrgetter

from skimage.filters import threshold_otsu, threshold_li
from scipy.ndimage.morphology import binary_closing, binary_fill_holes, generate_binary_structure
from scipy.ndimage.measurements import label, find_objects, center_of_mass
from scipy.ndimage.filters import median_filter
from skimage.measure import regionprops
from misc import BBox


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

def gather_statistics(stack_data, is_inverse=True):
    thresholded_stack = np.empty_like(stack_data, dtype=np.uint8)

    print 'Gathering statistics...'

    for slice_idx in np.arange(stack_data.shape[0]):
        threshold_val = threshold_otsu(stack_data[slice_idx])
        thresholded_stack[slice_idx] = stack_data[slice_idx] < threshold_val if is_inverse \
                                       else stack_data[slice_idx] >= threshold_val
        thresholded_stack[slice_idx] = median_filter(thresholded_stack[slice_idx], size=(1,1))

        if slice_idx % 100 == 0 or slice_idx == stack_data.shape[0]-1:
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

def object_counter(stack_binary_data):
    #labeled_stack, num_labels = label(stack_binary_data, \
    #                                  structure=generate_binary_structure(3,3))\
    print 'Object counting - Labeling...'
    labeled_stack, num_labels = label(stack_binary_data)

    print 'Object counting - BBoxing...'
    bboxes_labels = [BBox(bb_obj) for bb_obj in find_objects(labeled_stack)]

    print 'Object counting - Centers of masses...'
    center_of_mass_labels = center_of_mass(stack_binary_data, labeled_stack, np.arange(1, num_labels+1))

    objects_stats = pd.DataFrame(columns=_MEASUREMENTS_VALS)

    print 'Object counting - Stats gathering...'
    for slice_idx in np.arange(labeled_stack.shape[0]):
        for region in regionprops(labeled_stack[slice_idx]):
            objects_stats = objects_stats.append({_measure: region[_measure] \
                                        for _measure in _MEASUREMENTS_VALS}, \
                                            ignore_index=True)

    objects_stats = objects_stats.groupby('label', as_index=False).sum()

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

    return objects_stats, labeled_stack

def cell_counter(slice_binary_data, min_area=0.0, min_circularity=0.0, slice_index=-1):
    print 'Object counting - Labeling...'
    labeled_data, num_labels = label(slice_binary_data)

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

def extract_data_by_label(stack_data, stack_stats, label, bb_side_offset=0):
    filtered_stats = stack_stats[stack_stats['label'] == label].head(1)
    bbox = BBox(filtered_stats.to_dict('records')[0])
    print "extracted_data_by_label = %s" % str(filtered_stats.to_dict('records')[0])
    tuple_bbox = bbox.create_tuple(offset=bb_side_offset, max_ranges=stack_data.shape)

    return stack_data[bbox.create_tuple(offset=bb_side_offset)], tuple_bbox

def extract_largest_area_data(stack_data, stack_stats, bb_side_offset=0):
    filtered_stats = stack_stats.sort(['area'], ascending=False).head(1)

    return extract_data_by_label(stack_data, stack_stats, \
            filtered_stats['label'].values[0], bb_side_offset=bb_side_offset)

def _calc_sphericity(area, perimeter):
    r = ((3.0 * area) / (4.0 * np.pi)) ** (1.0/3.0)

    return (4.0 * np.pi * (r*r)) / perimeter

def _calc_circularity(area, perimeter):
    return 4.0 * np.pi * area / (perimeter * perimeter)

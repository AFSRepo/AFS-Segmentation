import numpy as np
import pandas as pd

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

_MEASUREMENTS_VALS = _MEASUREMENTS.values()
_MEASUREMENTS_EXTRA_VALS = _MEASUREMENTS_EXTRA.values()

def gather_statistics(stack_data, is_inverse=True):
    thresholded_stack = np.empty_like(stack_data, dtype=np.uint8)

    for slice_idx in np.arange(stack_data.shape[0]):
        threshold_val = threshold_otsu(stack_data[slice_idx])
        thresholded_stack[slice_idx] = stack_data[slice_idx] < threshold_val if is_inverse \
                                       else stack_data[slice_idx] >= threshold_val
        thresholded_stack[slice_idx] = median_filter(thresholded_stack[slice_idx], size=(2,2))

    stack_statistics = object_counter(thresholded_stack)

    return stack_statistics, thresholded_stack

def object_counter(stack_binary_data):
    #labeled_stack, num_labels = label(stack_binary_data, \
    #                                  structure=generate_binary_structure(3,3))
    print 'Object counting - Labeling...'
    labeled_stack, num_labels = label(stack_binary_data)

    print 'Object counting - BBoxing...'
    bboxes_labels = [BBox(bb_obj) for bb_obj in find_objects(labeled_stack)]

    print 'Object counting - Centers of masses...'
    center_of_mass_labels = center_of_mass(stack_binary_data, labeled_stack, np.arange(1, num_labels+1))

    objects_stats = pd.DataFrame(columns=_MEASUREMENTS_VALS)

    print 'Object counting - Stats gathering...'
    for slice_idx in np.arange(labeled_stack.shape[0]):
        #print "Stats for slice# %d" % slice_idx

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

    return objects_stats

def extract_data_by_label(stack_data, stack_stats, label):
    filtered_stats = stack_stats[stack_stats['label'] == label].head(1)
    bbox = BBox(filtered_stats.to_dict('records')[0])
    return stack_data[bbox.create_tuple()]

def extract_largest_area_data(stack_data, stack_stats):
    filtered_stats = stack_stats.sort(['area'], ascending=False).head(1)
    return extract_data_by_label(stack_data, stack_stats, filtered_stats['label'].values[0])

def _calc_sphericity(area, perimeter):
    r = ((3.0 * area) / (4.0 * np.pi)) ** (1.0/3.0)

    return (4.0 * np.pi * (r*r)) / perimeter

from operator import attrgetter
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu, threshold_li
from scipy.ndimage.filters import median_filter
from scipy.ndimage.morphology import binary_dilation, binary_closing, binary_fill_holes
from skimage.morphology import disk, white_tophat
from scipy.ndimage.measurements import label, find_objects
from skimage.measure import regionprops
from ..segmentation.eyes import eyes_statistics, eyes_zrange
from .morphology import gather_statistics

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

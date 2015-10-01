import numpy as np
import pandas as pd
import os
from skimage.filters import threshold_adaptive
from scipy.ndimage.morphology import binary_opening, binary_fill_holes, binary_closing, generate_binary_structure
from modules.tools.morphology import cell_counter, gather_statistics, extract_largest_area_data
from scipy.ndimage.measurements import label, find_objects, center_of_mass
from scipy.ndimage.filters import median_filter
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from mpl_toolkits.mplot3d import Axes3D

def gen_ranges(s,arr):
    l = len(arr)
    st = np.ceil(float(l)/s)
    print l, st
    out = []
    for i in np.arange(st):
        if (i * s + s) <= l:
            out.append(np.arange(i * s, i * s + s))
        else:
            out.append(np.arange(i * s, i * s + (i * s + s - l - 1)))

    return out

def get_empty_indices(binary_spine_data):
    print 'Find empty slices...'
    empty_indices = np.array([])

    for i in np.arange(binary_spine_data.shape[0]):
        print 'Slice %d/%d...' % (i, binary_spine_data.shape[0] - 1)
        slice_data = binary_spine_data[i]

        if not np.count_nonzero(slice_data):
            empty_indices = np.append(empty_indices, i)

    return empty_indices

def get_empty_ranges(empty_indices):
    print 'Get empty ranges...'
    start_idx, stop_idx = empty_indices[0], None
    curr_item, next_item = None, None
    output = []

    for i, item in enumerate(empty_indices):
        print 'Item %d/%d...' % (i, len(empty_indices) - 1)

        if i != (len(empty_indices) - 1):
            curr_item, next_item = item, empty_indices[i + 1]
        else:
            curr_item, next_item = item, item

        if i != (len(empty_indices) - 1):
            if (curr_item + 1) == next_item:
                continue
            else:
                stop_idx = empty_indices[i]
                output.append(np.arange(start_idx, stop_idx + 1))
                start_idx = empty_indices[i + 1]
        else:
            stop_idx = empty_indices[i]
            output.append(np.arange(start_idx, stop_idx + 1))

    return output

def fill_emptiness(empty_ranges, binary_spine_data):
    print 'Fill empty ranges...'
    slice_size = binary_spine_data[0].shape
    output = binary_spine_data.copy()

    def check_range(rng):
        if rng[0] == 0:
            return 1
        elif rng[-1] == (binary_spine_data.shape[0] - 1):
            return -1
        else:
            return 0

    for rng in empty_ranges:
        print 'Range [%d:%d]...' % (rng[0], rng[-1])

        if check_range(rng) == 0:
            if len(rng) > 1:
                l_arr,r_arr = np.array_split(rng, 2)
                output[l_arr.astype(np.int32)] = output[l_arr[0] - 1]
                output[r_arr.astype(np.int32)] = output[r_arr[-1] + 1]
            else:
                output[rng.astype(np.int32)] = output[rng[-1] + 1]
        elif check_range(rng) == -1:
            output[rng.astype(np.int32)] = output[rng[0] - 1]
        elif check_range(rng) == 1:
            output[rng.astype(np.int32)] = output[rng[-1] + 1]
        else:
            print 'Range can\'t be checked!'

    return output

def get_centroids_border_labels(empty_ranges, binary_spine_data):
    print 'Get centroinds of border labels...'

    def check_range(rng):
        if rng[0] == 0:
            return 1
        elif rng[-1] == (binary_spine_data.shape[0] - 1):
            return -1
        else:
            return 0

    for rng in empty_ranges:
        print 'Range [%d:%d]...' % (rng[0], rng[-1])

        if check_range(rng) == 0:
            if len(rng) > 1:
                l_arr,r_arr = np.array_split(rng, 2)
                output[l_arr.astype(np.int32)] = output[l_arr[0] - 1]
                output[r_arr.astype(np.int32)] = output[r_arr[-1] + 1]
            else:
                output[rng.astype(np.int32)] = output[rng[-1] + 1]
        elif check_range(rng) == -1:
            output[rng.astype(np.int32)] = output[rng[0] - 1]
        elif check_range(rng) == 1:
            output[rng.astype(np.int32)] = output[rng[-1] + 1]
        else:
            print 'Range can\'t be checked!'


def fit_spine(x, y, z, z_new):
    coefs = poly.polyfit(z, zip(x,y), 3)
    y_fitted = poly.polyval(z_new, coefs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(y_fitted[0], y_fitted[1], z_new, label='spine')
    ax.scatter(x, y, z)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

def segment_spine(spine_data, min_area=200.0, min_circularity=0.4, num_splits=10, tolerance=25):
    spine_data_thresholded = np.zeros_like(spine_data, dtype=np.uint8)
    spine_data_labeled = np.zeros_like(spine_data)
    spine_data_labeled_final = np.zeros_like(spine_data)
    total_stats = pd.DataFrame()

    stats_filepath = 'C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels.csv'
    labels_filepath = 'C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels_8bit_179x229x1667.raw'

    test_range = np.arange(spine_data.shape[0])

    for slice_idx in test_range:
    #for slice_idx in np.arange(spine_data.shape[0]):
        print 'Slice %d/%d' % (slice_idx + 1, spine_data.shape[0])
        spine_data_thresholded[slice_idx] = threshold_adaptive(spine_data[slice_idx], block_size=40)
        spine_data_thresholded[slice_idx] = binary_closing(spine_data_thresholded[slice_idx], iterations=1)
        spine_data_thresholded[slice_idx] = median_filter(spine_data_thresholded[slice_idx], size=(3,3))

        stats, labels = cell_counter(spine_data_thresholded[slice_idx], slice_index=slice_idx)
        spine_data_labeled[slice_idx] = labels
        total_stats = total_stats.append(stats, ignore_index=True)

    print "Stats filtering..."
    filtered_stats_neg = total_stats[(total_stats['area'] <= min_area) | (total_stats['circularity'] <= min_circularity)]
    filtered_stats = total_stats[(total_stats['area'] > min_area) & (total_stats['circularity'] > min_circularity)]

    for idx, row in filtered_stats_neg.iterrows():
        temp_slice = spine_data_labeled[row['slice_idx']]
        temp_slice[temp_slice == row['label']] = 0.0
        spine_data_labeled[row['slice_idx']] = temp_slice

    spine_data_labeled.tofile('C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels_8bit_179x229x1667.raw')

    filtered_stats_grp = filtered_stats.groupby('slice_idx', as_index=False).mean()

    ranges = np.array_split(test_range, num_splits)

    print "Range filtering..."
    for rng in ranges:
        print "Range [%d:%d]..." % (rng[0], rng[-1])
        stats_rng = filtered_stats_grp[filtered_stats_grp['slice_idx'].isin(rng)]
        stats_local = filtered_stats[filtered_stats['slice_idx'].isin(rng)]

        if not stats_rng.empty:
            stats_rng_mean = stats_rng.mean()

            stats_local.loc[:,'com_sum'] = stats_local.apply(lambda row: np.abs(row['com_y'] - stats_rng_mean['com_y']) + \
                                                        np.abs(row['com_z'] - stats_rng_mean['com_z']), axis=1)

            stats_final = stats_local[stats_local['com_sum'] <= tolerance].sort(['com_sum']).groupby('slice_idx', as_index=False).first()

            print stats_final

            for idx, row in stats_final.iterrows():
                tmp_labels = spine_data_labeled[row['slice_idx']]
                tmp_labels[tmp_labels != row['label']] = 0.0
                tmp_labels[tmp_labels != 0] = 1.0
                spine_data_labeled_final[row['slice_idx']] = tmp_labels

    spine_data_thresholded.tofile('C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_threshod_8bit_179x229x1667.raw')
    spine_data_labeled_final.tofile('C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels_final_8bit_179x229x1667.raw')
    total_stats.to_csv('C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels.csv')


def main():
    spine_path = 'C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_32bit_179x229x1667.raw'
    spine_data = np.memmap(spine_path, dtype=np.float32, mode='r', shape=(1667,229,179))

    #segment_spine(spine_data, min_area=200.0, min_circularity=0.6)

    segmeneted_spine_brain_path = 'C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels_final_8bit_179x229x1667.raw'
    segmeneted_spine_brain_data = np.memmap(segmeneted_spine_brain_path, dtype=np.float32, mode='r', shape=(1667,229,179))

    empty_indices = get_empty_indices(segmeneted_spine_brain_data)
    print "empty_indices = " + str(empty_indices)
    empty_ranges = get_empty_ranges(empty_indices)
    print "empty_ranges = " + str(empty_ranges)
    filled_brain_seg = fill_emptiness(empty_ranges, segmeneted_spine_brain_data)

    filled_brain_seg.tofile('C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels_final_filled_8bit_179x229x1667.raw')

if __name__ == "__main__":
    main()

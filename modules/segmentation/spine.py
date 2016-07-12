import numpy as np
import pandas as pd
import os
import sys
from skimage.filters import threshold_adaptive, threshold_otsu, threshold_li
from skimage.restoration import denoise_bilateral
from skimage.draw import circle
from scipy.ndimage.morphology import binary_opening, binary_fill_holes, binary_closing, generate_binary_structure
from modules.tools.morphology import cell_counter, gather_statistics, extract_largest_area_data
from scipy.ndimage.measurements import label, find_objects, center_of_mass
from scipy.ndimage.filters import median_filter
#import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from mpl_toolkits.mplot3d import Axes3D
from modules.tools.io import open_data
from modules.tools.misc import timing

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

def get_centroids_bboxes_border_labels(empty_ranges, binary_spine_data):
    print 'Get centroinds of border labels...'

    output_centroids, output_bboxes = [], []

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
                first_idx, last_idx = rng[0] - 1, rng[-1] + 1
                slice_stats_first, __ = cell_counter(binary_spine_data[first_idx])
                print slice_stats_first
                slice_stats_last, __ = cell_counter(binary_spine_data[last_idx])
                print slice_stats_last

                output_centroids.append(np.array([[slice_stats_first['com_z'][0], slice_stats_first['com_y'][0], first_idx], \
                                                  [slice_stats_last['com_z'][0], slice_stats_last['com_y'][0], last_idx]]))
                first_bbox = np.array([[slice_stats_first['bb_z'][0], slice_stats_first['bb_y'][0]], [slice_stats_first['bb_depth'][0], slice_stats_first['bb_height'][0]]])
                last_bbox = np.array([[slice_stats_last['bb_z'][0], slice_stats_last['bb_y'][0]], [slice_stats_last['bb_depth'][0], slice_stats_last['bb_height'][0]]])
                output_bboxes.append(np.array([first_bbox, last_bbox]))
            else:
                nearest_idx = rng[-1] + 1
                slice_stats, __ = cell_counter(binary_spine_data[nearest_idx])
                output_centroids.append(np.array([[slice_stats['com_z'][0], slice_stats['com_y'][0], nearest_idx]]))
                output_bboxes.append(np.array([[slice_stats['bb_z'][0], slice_stats['bb_y'][0]], [slice_stats['bb_depth'][0], slice_stats['bb_height'][0]]]))
        else:
            nearest_idx = (rng[0] - 1) if (check_range(rng) == -1) else (rng[-1] + 1)
            slice_stats, __ = cell_counter(binary_spine_data[nearest_idx])
            output_centroids.append(np.array([[slice_stats['com_z'][0], slice_stats['com_y'][0], nearest_idx]]))
            output_bboxes.append(np.array([[slice_stats['bb_z'][0], slice_stats['bb_y'][0]], [slice_stats['bb_depth'][0], slice_stats['bb_height'][0]]]))

    return output_centroids, output_bboxes

def fit_fill_spine(empty_ranges, centroid_coords, label_bboxes, binary_spine_data):
    print 'Fit and fill spine...'

    output = binary_spine_data.copy()

    def extract_label(bbox, slice_idx, stack_data):
        y0, x0 = bbox[0][0], bbox[0][1]
        height, width = bbox[1][0], bbox[1][1]

        ext_label = stack_data[slice_idx, y0:(y0 + height), x0:(x0 + width)]

        return ext_label

    def get_label_ranges(label, coord_com):
        label_shape = label.shape

        r_range = np.arange(label_shape[0]) - label_shape[0]/2
        c_range = np.arange(label_shape[1]) - label_shape[1]/2

        return r_range + coord_com[0], c_range + coord_com[1]

    if len(empty_ranges) != len(centroid_coords):
        sys.exit("Empty ranges and centriods arrays are not equal!")

    for rng, coord, bboxes in zip(empty_ranges, centroid_coords, label_bboxes):
        print 'Range [%d:%d]...' % (rng[0], rng[-1])

        if len(coord) > 1 and len(rng) > 1:
            x = coord[:,0]
            y = coord[:,1]
            z = coord[:,2]

            coefs = poly.polyfit(z, zip(x,y), 3)
            xy_fitted = np.round(poly.polyval(rng, coefs))

            first_bbox = bboxes[0]
            last_bbox = bboxes[1]

            first_label = extract_label(first_bbox, z[0], binary_spine_data)
            last_label = extract_label(last_bbox, z[1], binary_spine_data)

            f_rng, l_rng = np.array_split(rng, 2)
            f_fit_rng, l_fit_rng = np.array_split(np.arange(len(xy_fitted[1])), 2)

            for index, z_val in zip(f_fit_rng, f_rng):
                rows, cols = get_label_ranges(first_label, (xy_fitted[0][index], xy_fitted[1][index]))
                output[z_val, rows[:, np.newaxis].astype(np.int32), cols.astype(np.int32)] = first_label

            for index, z_val in zip(l_fit_rng, l_rng):
                rows, cols = get_label_ranges(last_label, (xy_fitted[0][index], xy_fitted[1][index]))
                output[z_val, rows[:, np.newaxis].astype(np.int32), cols.astype(np.int32)] = last_label

        else:
            output[rng.astype(np.int32)] = output[coord[0][2]]

    return output

def fit_fill_spine_trajectory(empty_ranges, centroid_coords, label_bboxes, binary_spine_data, traj_color=5):
    print 'Fit and fill spine trajectory...'

    output = binary_spine_data.copy()

    def extract_label(bbox, slice_idx, stack_data):
        y0, x0 = bbox[0][0], bbox[0][1]
        height, width = bbox[1][0], bbox[1][1]

        ext_label = stack_data[slice_idx, y0:(y0 + height), x0:(x0 + width)]

        return ext_label

    def get_label_ranges(label, coord_com):
        label_shape = label.shape

        r_range = np.arange(label_shape[0]) - label_shape[0]/2
        c_range = np.arange(label_shape[1]) - label_shape[1]/2

        return r_range + coord_com[0], c_range + coord_com[1]

    if len(empty_ranges) != len(centroid_coords):
        sys.exit("Empty ranges and centriods arrays are not equal!")

    for rng, coord, bboxes in zip(empty_ranges, centroid_coords, label_bboxes):
        print 'Range [%d:%d]...' % (rng[0], rng[-1])

        if len(coord) > 1 and len(rng) > 1:
            x = coord[:,0]
            y = coord[:,1]
            z = coord[:,2]

            coefs = poly.polyfit(z, zip(x,y), 3)
            xy_fitted = np.round(poly.polyval(rng, coefs))

            for i, z_val in enumerate(rng):
                output[z_val, xy_fitted[0][i], xy_fitted[1][i]] = traj_color
        else:
            output[rng.astype(np.int32)] = output[coord[0][2]]

    return output

def get_circle_roi_spine(binary_spine_data):
    output = np.zeros_like(binary_spine_data, dtype=np.uint8)

    for idx in np.arange(binary_spine_data.shape[0]):
        print 'Slice %d/%d' % (idx + 1, binary_spine_data.shape[0])

        slice_stats, __ = cell_counter(binary_spine_data[idx])
        row, col = np.round(slice_stats['com_z'][0]).astype(np.int32), np.round(slice_stats['com_y'][0]).astype(np.int32)
        rr, cc = circle(row, col, 30)
        output[idx, rr, cc] = 1

    return output

def fit_spine(x, y, z, z_new):
    coefs = poly.polyfit(z, zip(x,y), 3)
    y_fitted = poly.polyval(z_new, coefs)

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot(y_fitted[0], y_fitted[1], z_new, label='spine')
    #ax.scatter(x, y, z)
    #ax.set_xlabel('X axis')
    #ax.set_ylabel('Y axis')
    #ax.set_zlabel('Z axis')
    #plt.show()

def _segment_cirlces(rng, spine_data):
    total_stats = pd.DataFrame()
    spine_data_thresholded = np.zeros_like(spine_data)
    spine_data_labeled = np.zeros_like(spine_data)

    for slice_idx in rng:
        print 'Slice %d/%d' % (slice_idx + 1, spine_data.shape[0])
        spine_data_thresholded[slice_idx] = spine_data[slice_idx] >= threshold_otsu(spine_data[slice_idx])
        spine_data_thresholded[slice_idx] = binary_closing(spine_data_thresholded[slice_idx], iterations=1)
        #spine_data_thresholded[slice_idx] = denoise_bilateral(spine_data[slice_idx].astype(np.uint8), sigma_range=0.05, sigma_spatial=15)
        #spine_data_thresholded[slice_idx] = threshold_adaptive(spine_data_thresholded[slice_idx], block_size=19)

        #spine_data_thresholded[slice_idx] = binary_closing(spine_data_thresholded[slice_idx], iterations=1)
        #spine_data_thresholded[slice_idx] = median_filter(spine_data_thresholded[slice_idx], size=(3,3))

        stats, labels = cell_counter(spine_data_thresholded[slice_idx], slice_index=slice_idx)
        spine_data_labeled[slice_idx] = labels

        total_stats = total_stats.append(stats, ignore_index=True)

    return spine_data_labeled, total_stats

def segment_spine_v2(spine_data, min_area=200.0, min_circularity=0.6, tolerance=5):
    spine_data_thresholded, total_stats, init_stats = None, None, None
    spine_data_labeled, spine_data_labeled_final = None, None

    stats_filepath = 'C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels.csv'
    init_stats_filepath = 'C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_init_labels.csv'
    labels_filepath = 'C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels_8bit_%dx%dx%d.raw' % (60, 207, 1220)
    labels_nof_filepath = 'C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels_not_filtered_32bit_%dx%dx%d.raw' % (60, 207, 1220)

    rng = np.arange(spine_data.shape[0])

    if not os.path.exists(init_stats_filepath):
        _, init_stats = _segment_cirlces(rng[-2:], spine_data)
        init_stats.to_csv(init_stats_filepath)
    else:
        init_stats = pd.read_csv(init_stats_filepath)

    if not os.path.exists(stats_filepath):
        spine_data_labeled, total_stats = _segment_cirlces(rng, spine_data)
        spine_data_labeled.tofile(labels_nof_filepath)
        total_stats.to_csv(stats_filepath)
    else:
        total_stats = pd.read_csv(stats_filepath)
        spine_data_labeled = open_data(labels_nof_filepath)

    print "Stats filtering..."
    filtered_stats = total_stats[(total_stats['area'] > min_area) & (total_stats['circularity'] > min_circularity)]
    filtered_stats_neg = total_stats[(total_stats['area'] <= min_area) | (total_stats['circularity'] <= min_circularity)]

    filtered_init_stats = init_stats[(init_stats['area'] > min_area) & (init_stats['circularity'] > min_circularity)]

    print "filtered_stats = %d" % filtered_stats.shape[0]
    print "filtered_stats_neg = %d" % filtered_stats_neg.shape[0]
    print "filtered_init_stats = %d" % filtered_init_stats.shape[0]

    print "Find spine in %d rows..." % filtered_init_stats.shape[0]
    spines = []
    for idx, row in filtered_init_stats.iterrows():
        spines.append(find_by_com(filtered_stats, spine_data.shape[0], row['com_y'], row['com_z']))

    print "Create labels..."
    for i, spine_stats in enumerate(spines):
        print "-----#%d %d circles are found...; dist(mean) = %f" % (i, spine_stats.shape[0], spine_stats['distance'].mean())
        spine_data_labeled_final = np.zeros_like(spine_data_labeled)

        for idx, row in spine_stats.iterrows():
            tmp_labels = spine_data_labeled[row['slice_idx']]
            tmp_labels[tmp_labels != row['label']] = 0
            tmp_labels[tmp_labels == row['label']] = 1
            spine_data_labeled_final[row['slice_idx']] = tmp_labels

        spine_tmlp = 'C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_19Wlabels_%d_8bit_%dx%dx%d.raw' #fish204_aligned_32bit_60x207x1220
        spine_data_labeled_final.tofile(spine_tmlp % (i, 60, 207, 1220))
        init_stats.to_csv('C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_stats%d.csv' % i)

@timing
def find_by_com(stats, slices_num, init_com_y, init_com_z, tolerance=5):
    print '#######Init (%f,%f)' % (init_com_y, init_com_z)
    spine_stats = pd.DataFrame()

    #for slice_idx in np.arange(slices_num)[::-1]:
    for slice_idx in np.arange(slices_num)[::-1]:
        print 'Slice %d/%d' % (slice_idx, slices_num)

        slice_stats = stats[stats['slice_idx'] == slice_idx]

        if slice_stats.shape[0] == 0:
            continue

        slice_stats.loc[:,'distance'] = slice_stats.apply(lambda row: np.sqrt((row['com_y'] - init_com_y)*(row['com_y'] - init_com_y) + \
                                                                            (row['com_z'] - init_com_z)*(row['com_z'] - init_com_z)), axis=1)

        slice_stats = slice_stats[slice_stats['distance'] < tolerance]

        if slice_stats.shape[0] == 0:
            continue

        #print "$$$$$$$$$$$$$$$$$$  SLICE STATS  $$$$$$$$$$$$$$$$$$$$$$"
        #print slice_stats

        min_dist_stat = slice_stats.ix[slice_stats['distance'].idxmin()]

        #print "$$$$$$$$$$$$$$$$$$  MIN DIST STATS  $$$$$$$$$$$$$$$$$$$$$$"
        #print min_dist_stat

        spine_stats = spine_stats.append(min_dist_stat, ignore_index=True)
        init_com_y, init_com_z = min_dist_stat['com_y'], min_dist_stat['com_z']
        print 'New init com (%f,%f)' % (init_com_y, init_com_z)

    return spine_stats


def segment_spine(spine_data, min_area=200.0, min_circularity=0.4, num_splits=10, tolerance=25):
    spine_data_thresholded = np.zeros_like(spine_data, dtype=np.uint8)
    spine_data_labeled = None
    spine_data_labeled_final = np.zeros_like(spine_data)
    total_stats = pd.DataFrame()

    stats_filepath = 'C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels.csv'
    labels_filepath = 'C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels_8bit_360x396x1220.raw'

    test_range = np.arange(spine_data.shape[0])

    for slice_idx in test_range:
        print 'Slice %d/%d' % (slice_idx + 1, spine_data.shape[0])
        spine_data_thresholded[slice_idx] = threshold_adaptive(spine_data[slice_idx], block_size=40)
        spine_data_thresholded[slice_idx] = binary_closing(spine_data_thresholded[slice_idx], iterations=1)
        spine_data_thresholded[slice_idx] = median_filter(spine_data_thresholded[slice_idx], size=(3,3))

        stats, labels = cell_counter(spine_data_thresholded[slice_idx], slice_index=slice_idx)
        spine_data_labeled[slice_idx] = labels
        total_stats = total_stats.append(stats, ignore_index=True)

    spine_data_labeled.tofile('C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels_not_filtered_8bit_360x396x1220.raw')

    print "Stats filtering..."
    filtered_stats_neg = total_stats[(total_stats['area'] <= min_area) | (total_stats['circularity'] <= min_circularity)]
    filtered_stats = total_stats[(total_stats['area'] > min_area) & (total_stats['circularity'] > min_circularity)]

    for idx, row in filtered_stats_neg.iterrows():
        temp_slice = spine_data_labeled[row['slice_idx']]
        temp_slice[temp_slice == row['label']] = 0.0
        spine_data_labeled[row['slice_idx']] = temp_slice

    spine_data_labeled.tofile('C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels_8bit_360x396x1220.raw')

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

            for idx, row in stats_final.iterrows():
                tmp_labels = spine_data_labeled[row['slice_idx']]
                tmp_labels[tmp_labels != row['label']] = 0.0
                tmp_labels[tmp_labels != 0] = 1.0
                spine_data_labeled_final[row['slice_idx']] = tmp_labels
                spine_data_labeled_final[row['slice_idx']] = binary_fill_holes(spine_data_labeled_final[row['slice_idx']])

    spine_data_thresholded.tofile('C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_threshod_8bit_360x396x1220.raw')
    spine_data_labeled_final.tofile('C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels_final_8bit_360x396x1220.raw')
    total_stats.to_csv('C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels.csv')

    return spine_data_labeled_final

def plot_spine_example():
    data_spine = open_data('C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\Old2\\fish204_spine_19Wlabels_0_32bit_60x207x1220.raw')
    data_spine = data_spine.astype(np.uint8)

    rng = np.arange(data_spine.shape[0])
    total_stats = pd.DataFrame()

    for slice_idx in rng:
        print 'Slice %d/%d' % (slice_idx + 1, data_spine.shape[0])
        if data_spine[slice_idx].any():
            stats, labels = cell_counter(data_spine[slice_idx], slice_index=slice_idx)
            total_stats = total_stats.append(stats, ignore_index=True)

    print total_stats

    z = total_stats['slice_idx'].values
    x, y = total_stats['com_y'].values, total_stats['com_z'].values
    z_new = np.setdiff1d(np.arange(data_spine.shape[0]), z)

    coefs = poly.polyfit(z, zip(x,y), 3)
    xy_fitted = np.round(poly.polyval(z_new, coefs))

    data_spine_new = data_spine.copy()
    for i, slice_idx in enumerate(z_new):
        data_spine_new[slice_idx, xy_fitted[1][i], xy_fitted[0][i]] = 3

    data_spine_new.tofile('C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_FIT_8bit_60x207x1220.raw')

def plot_spine_example2():
    segmeneted_spine_brain_data = open_data('C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\Old2\\fish204_spine_19Wlabels_0_32bit_60x207x1220.raw')

    empty_indices = get_empty_indices(segmeneted_spine_brain_data)
    empty_ranges = get_empty_ranges(empty_indices)
    label_centroids, label_bboxes = get_centroids_bboxes_border_labels(empty_ranges, segmeneted_spine_brain_data)
    filled_spine_seg = fit_fill_spine_trajectory(empty_ranges, label_centroids, label_bboxes, segmeneted_spine_brain_data)

    filled_spine_seg.tofile('C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_filled_spine_seg_32bit_60x207x1220.raw')

    dims = [np.arange(v) for v in filled_spine_seg.shape]
    x,y,z = np.meshgrid(*dims)
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(x, y, z)
    #ax.set_xlabel('X axis')
    #ax.set_ylabel('Y axis')
    #ax.set_zlabel('Z axis')
    #plt.show()

def run_spine_segmentation(spine_path):
    '''
    Segments spine using circle detection and approximates
    the trajectory of spine on empty ranges with fitting polynom,
    then the approximated trajectory is used to fill the emptiness
    with neighbour spine slices.
    '''

    spine_data = open_data(spine_path)
    segmeneted_spine_brain_data = segment_spine_v2(spine_data, min_area=20.0, min_circularity=0.5)

    empty_indices = get_empty_indices(segmeneted_spine_brain_data)
    empty_ranges = get_empty_ranges(empty_indices)
    label_centroids, label_bboxes = get_centroids_bboxes_border_labels(empty_ranges, segmeneted_spine_brain_data)
    filled_spine_seg = fit_fill_spine_trajectory(empty_ranges, label_centroids, label_bboxes, segmeneted_spine_brain_data)

    return filled_spine_seg

# def main():
#     spine_path = 'C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_32bit_179x229x1667.raw'
#     spine_data = np.memmap(spine_path, dtype=np.float32, mode='r', shape=(1667,229,179))
#
#     segmeneted_spine_brain_data = segment_spine(spine_data, min_area=200.0, min_circularity=0.5)
#
#     #segmeneted_spine_brain_path = 'C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels_final_8bit_179x229x1667.raw'
#     #segmeneted_spine_brain_data = np.memmap(segmeneted_spine_brain_path, dtype=np.float32, mode='r', shape=(1667,229,179))
#
#     empty_indices = get_empty_indices(segmeneted_spine_brain_data)
#     empty_ranges = get_empty_ranges(empty_indices)
#     label_centroids, label_bboxes = get_centroids_bboxes_border_labels(empty_ranges, segmeneted_spine_brain_data)
#     filled_spine_seg = fit_fill_spine(empty_ranges, label_centroids, label_bboxes, segmeneted_spine_brain_data)
#     #filled_spine_seg.tofile('C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels_final_filled_fitted_v2_8bit_179x229x1667.raw')
#
#     filled_spine_seg_path = 'C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels_final_filled_fitted_v2_8bit_179x229x1667.raw'
#     filled_spine_seg_data = np.memmap(filled_spine_seg_path, dtype=np.float32, mode='r', shape=(1667,229,179))
#     thin_spine_seg = get_circle_roi_spine(filled_spine_seg_data)
#     thin_spine_seg.tofile('C:\\Users\\Administrator\\Documents\\AFS-Segmentation\\tests\\fish204_spine_labels_final_thinned_seg_8bit_179x229x1667.raw')
#
# if __name__ == "__main__":
#     main()

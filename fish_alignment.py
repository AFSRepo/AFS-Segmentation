import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import map_coordinates
from scipy.ndimage.morphology import binary_opening
from modules.tools.processing import binarizator
from modules.tools.morphology import object_counter, gather_statistics, extract_largest_area_data, cell_counter, stats_at_slice
from modules.segmentation.eyes import eyes_statistics
from modules.segmentation.common import flip_fish
from modules.tools.io import create_raw_stack, open_data, create_filename_with_shape, parse_filename

TMP_PATH = "C:\\Users\\Administrator\\Documents\\tmp"

def test_points_rotation():
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)

    x,y,z = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
    #x,y,z
    eye_l, eye_r = [1,3,5], [5,2,8]
    eye_c = np.array([(lv + rv)/2. for lv,rv in zip(eye_l, eye_r)])

    points = np.array([eye_l, eye_r])
    points_s = points - eye_c

    print eye_c
    print points_s
    #print points
    #print points_s

    aa_y = np.arcsin(np.sqrt(points_s[0][0]**2 + points_s[0][2]**2) / np.sqrt(points_s[0].dot(points_s[0])))
    aa_z = np.arcsin(points_s[0][0] / np.sqrt(points_s[0][0]**2 + points_s[0][2]**2))
    #aa_y = np.arccos(points_s[0][0] / np.sqrt(points_s[0][1:].dot(points_s[0][1:])))
    #aa_z = np.arccos(points_s[0][0] / np.sqrt(points_s[0][:2].dot(points_s[0][:2])))



    #l_eye_cos_angle_x = np.arccos(points_s[0][0] / np.sqrt(points_s[0][:2].dot(points_s[0][:2])))
    #l_eye_cos_angle_y = np.arccos(np.sqrt(points_s[0][:2].dot(points_s[0][:2])) / np.sqrt(points_s[0].dot(points_s[0])) )
    #l_eye_cos_angle_x = points_s[0].dot(x) / (np.sqrt(points_s[0].dot(points_s[0])) * np.sqrt(x.dot(x)))

    #l_eye_cos_angle_z = points_s[0,0] / np.sqrt(points_s[0].dot(points_s[0]))

    alpha = 0
    beta = aa_y
    phi = 0#aa_z

    Rx = np.matrix([[1., 0., 0.], [0., np.cos(alpha), -np.sin(alpha)],[0., np.sin(alpha), np.cos(alpha)]])
    Ry = np.matrix([[np.cos(beta), 0., np.sin(beta)], [0., 1., 0.], [-np.sin(beta), 0., np.cos(beta)]])
    Rz = np.matrix([[np.cos(phi), -np.sin(phi), 0.], [np.sin(phi), np.cos(phi), 0.], [0., 0., 1.]])

    R = Rz * Ry * Rx

    points_rotated = np.array((R * points_s.T).T + eye_c)
    print points_rotated


    # ax.scatter(eye_c[2], eye_c[1], eye_c[0], c='b', marker='o')
    # ax.scatter(points[:,2], points[:,1], points[:,0], c='r', marker='o')
    # ax.scatter(points_rotated[:,2], points_rotated[:,1], points_rotated[:,0], c='g', marker='o')

    #XY
    # ax.scatter(eye_c[0], eye_c[1] , c='r', marker='o')
    # ax.scatter(points[:,0], points[:,1], c='b', marker='o')
    # ax.scatter(points_rotated[:,0], points_rotated[:,1], c='g', marker='o')

    # #XZ
    ax.scatter(eye_c[0], eye_c[2] , c='r', marker='o')
    ax.scatter(points[:,0], points[:,2], c='b', marker='o')
    ax.scatter(points_rotated[:,0], points_rotated[:,2], c='g', marker='o')

    ax.set_xlim((0,10))
    ax.set_ylim((0,10))
    #ax.set_zlim((0,10))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    #ax.set_zlabel('Z')
    plt.show()

def test_points_rotation2():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax = fig.add_subplot(111, projection='3d')
    eye_l, eye_r = [1,3,5], [5,2,8]
    eye_c = np.array([(lv + rv)/2. for lv,rv in zip(eye_l, eye_r)])

    p = np.array([eye_l, eye_r])
    p_s = p - eye_c

    angle_y = get_angle(p_s[0][[0,2]], p_s[0])
    angle_x = get_angle(np.array([p_s[0][1]]), p_s[0][[0,2]])

    Ry = np.matrix([[np.cos(angle_y), 0., np.sin(angle_y)], [0., 1., 0.], [-np.sin(angle_y), 0., np.cos(angle_y)]])
    Rx = np.matrix([[1., 0., 0.], [0., np.cos(angle_x), -np.sin(angle_x)],[0., np.sin(angle_x), np.cos(angle_x)]])

    R = Ry * Rx
    p_r= np.array(( R * p_s.T).T + eye_c)

    ax.scatter(eye_c[0], eye_c[1] , c='r', marker='o')
    ax.scatter(p[:,0], p[:,1], c='b', marker='o')
    ax.scatter(p_r[:,0], p_r[:,1], c='g', marker='o')

    # ax.scatter(eye_c[0], eye_c[1], eye_c[2] , c='r', marker='o')
    #
    # ax.scatter(p[:,0], p[:,1], p[:,2], c='b', marker='o')
    # ax.plot(p[:,0], p[:,1], p[:,2])
    #
    # ax.scatter(p_r[:,0], p_r[:,1], p_r[:,2], c='g', marker='x')
    # ax.plot(p_r[:,0], p_r[:,1], p_r[:,2])

    ax.set_xlim((0,10))
    ax.set_ylim((0,10))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()

def test_points_rotation3():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax = fig.add_subplot(111, projection='3d')
    eye_r, eye_l = np.array([263,246,190]), np.array([166,242,208])
    # eye_r, eye_l = np.array([50,50,50]), np.array([150,150,50])
    # eye_r, eye_l = np.array([10,20,2]), np.array([70,5,50])

    eye_c = np.array([(lv + rv)/2. for lv,rv in zip(eye_l, eye_r)])
    p = np.array([eye_r, eye_l])

    p_s = p - eye_c

    print 'Angle and matrix creating...'
    alpha = get_angle(p_s[0][[0,1]], p_s[0])
    beta = get_angle(p_s[0][[0]], p_s[0][[0,1]])

    alpha = alpha if np.arctan2(p_s[0][2], p_s[0][0]) >= 0 else -alpha
    beta = -beta if np.arctan2(p_s[0][1], p_s[0][0]) >= 0 else 2. * np.pi + beta

    print 'alpha = %f' % np.rad2deg(get_angle(p_s[0][[0,1]], p_s[0]))
    print 'beta = %f' % np.rad2deg(2*np.pi + get_angle(p_s[0][[0]], p_s[0][[0,1]]))
    print 'atan2 = %f' % np.arctan2(p_s[0][1], p_s[0][0])
    print 'atan2 = %f' % np.arctan2(p_s[0][2], p_s[0][0])

    Ry = np.matrix([[np.cos(alpha), 0., np.sin(alpha)], [0., 1., 0.], [-np.sin(alpha), 0., np.cos(alpha)]])
    Rz = np.matrix([[np.cos(beta), -np.sin(beta), 0.], [np.sin(beta), np.cos(beta), 0.], [0., 0., 1.]])

    R = Ry * Rz

    print p_s
    print (R * p_s.T).T

    p_sr = (R * p_s.T).T + eye_c

    ax.scatter(eye_c[2], eye_c[0] , c='r', marker='o')
    ax.scatter(p[:,2], p[:,0], c='b', marker='o')
    ax.scatter(p_sr[:,2], p_sr[:,0], c='g', marker='o')
    # ax.scatter(p2[:,2], p2[:,0], c='m', marker='o')

    # ax.scatter(eye_c[0], eye_c[1] , c='r', marker='o')
    # ax.scatter(p[:,0], p[:,1], c='b', marker='o')
    # ax.scatter(p_sr[:,0], p_sr[:,1], c='g', marker='o')
    #ax.scatter(p2[:,0], p2[:,1], c='m', marker='o')

    # ax.scatter(eye_c[0], eye_c[1], eye_c[2] , c='r', marker='o')
    #
    # ax.scatter(p[:,0], p[:,1], p[:,2], c='b', marker='o')
    # ax.plot(p[:,0], p[:,1], p[:,2])
    #
    # ax.scatter(p_r[:,0], p_r[:,1], p_r[:,2], c='g', marker='x')
    # ax.plot(p_r[:,0], p_r[:,1], p_r[:,2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()

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

def get_angle(v1_adj, v2_hyp):
    return np.arccos(np.sqrt(v1_adj.dot(v1_adj)) / np.sqrt(v2_hyp.dot(v2_hyp)))

def test_fish_alignemnt():
    data = np.memmap("fish202_8bit_640x640x146.raw", dtype='uint8', shape=(146,640,640)).copy()
    dims = data.shape

    #eye_l, eye_r = np.array([166,242,208]), np.array([263,246,190])
    #eye_l, eye_r = np.array([189,351,78]), np.array([389,350,79])
    # eye_l, eye_r = np.array([196,370,90])/2., np.array([392,341,64])/2.
    eye_l, eye_r = np.array([196,370,90]), np.array([392,341,64])

    datac = data.copy()
    datac[eye_r[2], eye_r[1], eye_r[0]] = 255
    datac[eye_l[2], eye_l[1], eye_l[0]] = 255
    datac.astype(np.uint8).tofile("fish202_marked_8bit_640x640x146.raw")

    eye_c = np.array([(lv + rv)/2. for lv,rv in zip(eye_l, eye_r)])
    p = np.array([eye_l, eye_r])
    p_s = p - eye_c

    print 'Angle and matrix creating...'
    alpha = get_angle(p_s[0][[0,1]], p_s[0])
    beta = -get_angle(p_s[0][[0]], p_s[0][[0,1]])

    #Rz = np.matrix([[np.cos(alpha), -np.sin(alpha), 0.], [np.sin(alpha), np.cos(alpha), 0.], [0., 0., 1.]])
    #Ry = np.matrix([[np.cos(beta), 0., np.sin(beta)], [0., 1., 0.], [-np.sin(beta), 0., np.cos(beta)]])

    Ry = np.matrix([[np.cos(alpha), 0., np.sin(alpha)], [0., 1., 0.], [-np.sin(alpha), 0., np.cos(alpha)]])
    Rz = np.matrix([[np.cos(beta), -np.sin(beta), 0.], [np.sin(beta), np.cos(beta), 0.], [0., 0., 1.]])

    R = Rz * Ry

    print p_s
    print (R * p_s.T).T

    p_sr = (R * p_s.T).T + eye_c

    print 'Coords shifting...'
    zv, yv, xv = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing='ij')
    # coordinates_translated = np.array([np.ravel(zv - eye_c[2]).T, np.ravel(yv - eye_c[1]).T, np.ravel(xv - eye_c[0]).T])
    # coordinates_translated = np.array([np.ravel(xv - eye_c[0]).T, np.ravel(yv - eye_c[1]).T, np.ravel(zv - eye_c[2]).T])
    coordinates_translated = np.array([np.ravel(xv - eye_c[0]).T, np.ravel(yv - eye_c[1]).T, np.ravel(zv - eye_c[2]).T])

    print 'Rotating...'
    coordinates_rotated = np.array(R * coordinates_translated)

    print 'Coords back shifting...'
    # coordinates_rotated[0] = coordinates_rotated[0] + eye_c[2]
    # coordinates_rotated[1] = coordinates_rotated[1] + eye_c[1]
    # coordinates_rotated[2] = coordinates_rotated[2] + eye_c[0]
    coordinates_rotated[0] = coordinates_rotated[0] + eye_c[0]
    coordinates_rotated[1] = coordinates_rotated[1] + eye_c[1]
    coordinates_rotated[2] = coordinates_rotated[2] + eye_c[2]

    print 'Reshaping...'
    z_coordinates = np.reshape(coordinates_rotated[2,:], dims)
    y_coordinates = np.reshape(coordinates_rotated[1,:], dims)
    x_coordinates = np.reshape(coordinates_rotated[0,:], dims)

    #get the values for your new coordinates
    print 'Interpolation in 3D...'
    new_data = interp3(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), data, z_coordinates, y_coordinates, x_coordinates, order=3)
    print new_data.shape

    x1,y1,z1 = int(round(p_sr[0,0])), int(round(p_sr[0,1])), int(round(p_sr[0,2]))
    x2,y2,z2 = int(round(p_sr[1,0])), int(round(p_sr[1,1])), int(round(p_sr[1,2]))

    new_data[z1,y1,x1] = 255
    new_data[z2,y2,x2] = 255

    print 'New data saving...'
    new_data.astype(np.uint8).tofile("fish202_rotated_8bit_640x640x146.raw")

def get_rot_matrix_arbitrary_axis(ux, uy, uz, theta):
    print ux
    print uy
    print uz
    print theta
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

    print 'origin_point = %s' % str(origin_point)
    print 'rot_vector = %s' % str(rot_axis)

    R = get_rot_matrix_arbitrary_axis(rot_axis[0], rot_axis[1], rot_axis[2], angle)

    print 'Coords shifting...'
    zv, yv, xv = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing='ij')
    coordinates_translated = np.array([np.ravel(xv - origin_point[0]).T, \
                                       np.ravel(yv - origin_point[1]).T, \
                                       np.ravel(zv - origin_point[2]).T])

    print 'Rotating...'
    coordinates_rotated = np.array(R * coordinates_translated)

    print 'Coords back shifting...'
    coordinates_rotated[0] = coordinates_rotated[0] + origin_point[0]
    coordinates_rotated[1] = coordinates_rotated[1] + origin_point[1]
    coordinates_rotated[2] = coordinates_rotated[2] + origin_point[2]

    print 'Reshaping...'
    x_coordinates = np.reshape(coordinates_rotated[0,:], dims)
    y_coordinates = np.reshape(coordinates_rotated[1,:], dims)
    z_coordinates = np.reshape(coordinates_rotated[2,:], dims)

    #get the values for your new coordinates
    print 'Interpolation in 3D...'
    interp_data = interp3(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), data, \
                          z_coordinates, y_coordinates, x_coordinates, order=interp_order)

    return interp_data

def align_by_eyes_centroids(filepath, centroids):
    data = open_data(filepath)
    dims = data.shape
    eye_l, eye_r = centroids

    eye_c = np.array([(lv + rv)/2. for lv,rv in zip(eye_l, eye_r)])
    p = np.array([eye_l, eye_r])
    p_s = p - eye_c

    datac = data.copy()
    datac[eye_r[2], eye_r[1], eye_r[0]] = 255
    datac[eye_l[2], eye_l[1], eye_l[0]] = 255
    datac[eye_c[2], eye_c[1], eye_c[0]] = 255
    datac.astype(np.float32).tofile(os.path.join(TMP_PATH, create_filename_with_shape(filepath, datac.shape, prefix='marked')))

    print 'Angle and matrix creating...'
    alpha = get_angle(p_s[0][[0,1]], p_s[0])
    beta = get_angle(p_s[0][[0]], p_s[0][[0,1]])

    x1, y1 = p_s[0][[0,1]]
    x2, y2, z2 = p_s[0]

    x_axis_vec = np.array([1.,0.,0.])
    theta = np.arccos(p_s[0][0]/np.sqrt(p_s[0].dot(p_s[0])))

    alpha_axis = np.cross(p_s[0], x_axis_vec)
    alpha_axis = alpha_axis / np.linalg.norm(alpha_axis)
    print alpha_axis

    Rcross = get_rot_matrix_arbitrary_axis(alpha_axis[0], alpha_axis[1], alpha_axis[2], -theta)
    Rz = np.matrix([[np.cos(beta), -np.sin(beta), 0.], [np.sin(beta), np.cos(beta), 0.], [0., 0., 1.]])

    R = Rcross

    print R.shape
    print Rcross.shape

    print 'Old eye\'s coords:'
    print p_s
    print 'New eye\'s coords:'
    print (R * p_s.T).T

    p_sr = np.asarray((R * p_s.T).T + eye_c)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(eye_c[2], eye_c[0] , c='r', marker='o')
    # ax.scatter(p[:,2], p[:,0], c='b', marker='o')
    # ax.scatter(p_sr[:,2], p_sr[:,0], c='g', marker='o')
    # plt.show()

    print 'Coords shifting...'
    zv, yv, xv = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing='ij')
    coordinates_translated = np.array([np.ravel(xv - eye_c[0]).T, np.ravel(yv - eye_c[1]).T, np.ravel(zv - eye_c[2]).T])

    print 'Rotating...'
    coordinates_rotated = np.array(R * coordinates_translated)

    print 'Coords back shifting...'
    coordinates_rotated[0] = coordinates_rotated[0] + eye_c[0]
    coordinates_rotated[1] = coordinates_rotated[1] + eye_c[1]
    coordinates_rotated[2] = coordinates_rotated[2] + eye_c[2]

    print 'Reshaping...'
    z_coordinates = np.reshape(coordinates_rotated[2,:], dims)
    y_coordinates = np.reshape(coordinates_rotated[1,:], dims)
    x_coordinates = np.reshape(coordinates_rotated[0,:], dims)

    #get the values for your new coordinates
    print 'Interpolation in 3D...'
    interp_data = interp3(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), data, z_coordinates, y_coordinates, x_coordinates, order=3)

    print p_sr[0]
    print interp_data.shape

    interp_data[int(p_sr[0,2]), int(p_sr[0,1]), int(p_sr[0,0])] = 255
    interp_data[int(p_sr[1,2]), int(p_sr[1,1]), int(p_sr[1,0])] = 255

    return interp_data, eye_c

def get_centroid_at_slice():
    path_fish202 = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish202\\fish202_aligned_32bit_320x320x998.raw"

    largest_volume_region = None

    if not os.path.exists(os.path.join(TMP_PATH, "fish202_aligned_32bit_218x228x940.raw")):
        input_data = np.memmap(path_fish202, dtype=np.float32, shape=(998,320,320)).copy()

        binarized_stack, bbox, eyes_stats = binarizator(input_data)
        binary_stack_stats, _ = object_counter(binarized_stack)
        largest_volume_region, largest_volume_region_bbox = extract_largest_area_data(input_data, binary_stack_stats, bb_side_offset=20)
        largest_volume_region.tofile(os.path.join(TMP_PATH, create_filename_with_shape(path_fish202, largest_volume_region.shape)))

        print 'largest_volume_region_bbox:'
        print largest_volume_region_bbox

        print 'original shape:'
        print input_data.shape
    else:
        largest_volume_region = np.memmap(os.path.join(TMP_PATH, "fish202_aligned_32bit_218x228x940.raw"), dtype=np.float32, shape=(940,228,218)).copy()

    stack_statistics, _ = gather_statistics(largest_volume_region)
    eyes_stats = eyes_statistics(stack_statistics)

    eye_c = np.round([eyes_stats['com_x'].mean(), eyes_stats['com_y'].mean(), eyes_stats['com_z'].mean()]).astype(np.int32)

    print 'Eyes stats:'
    print eyes_stats

    #let's use 530 slice as tail landmark
    lmt_slice_idx = 530
    ext_vol_len = largest_volume_region.shape[0]
    print 'ext_vol_len = %f' % ext_vol_len
    eye1_idx_frac, eye2_idx_frac = eyes_stats['com_z'].values[0] / float(ext_vol_len),\
                                   eyes_stats['com_z'].values[1] / float(ext_vol_len)
    landmark_tail_idx_frac = lmt_slice_idx / float(ext_vol_len)
    landmark_tail_idx_eyes_offset = (eye1_idx_frac + eye2_idx_frac)/2.0 - landmark_tail_idx_frac

    print 'eye1_idx_frac = %f' % eye1_idx_frac
    print 'eye2_idx_frac = %f' % eye2_idx_frac
    print 'landmark_tail_idx_frac = %f' % landmark_tail_idx_frac
    print 'landmark_tail_idx_eyes_offset = %f' % landmark_tail_idx_eyes_offset
    print 'landmark_tail_idx = %f' % (float(eye1_idx_frac * ext_vol_len + eye2_idx_frac * ext_vol_len)/2.0 - landmark_tail_idx_eyes_offset * ext_vol_len)

    landmark_tail_idx = int(float(eye1_idx_frac * ext_vol_len + eye2_idx_frac * ext_vol_len)/2.0 - landmark_tail_idx_eyes_offset * ext_vol_len)

    #z -> y
    #y -> x
    lm_slice_stats, lm_labeled_slice = stats_at_slice(largest_volume_region, landmark_tail_idx)

    z_axis_vec = np.array([0., 0., -1.])
    spinal_vec = np.array([0, lm_slice_stats['com_z'] - eye_c[1], landmark_tail_idx - eye_c[2]])

    rot_axis = np.cross(z_axis_vec, spinal_vec)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    theta = np.arccos(z_axis_vec.dot(spinal_vec)/(np.sqrt(z_axis_vec.dot(z_axis_vec)) * np.sqrt(spinal_vec.dot(spinal_vec))))

    data_rotated = rotate_around_vector(largest_volume_region, eye_c, rot_axis, -theta, interp_order=3)
    data_rotated.astype(np.float32).tofile(os.path.join(TMP_PATH, create_filename_with_shape(path_fish202, data_rotated.shape, prefix='zrotated')))

    print 'z_axis_vec = %s' % str(z_axis_vec)
    print 'spinal_vec = %s' % str(spinal_vec)
    print 'theta = %f' % np.rad2deg(theta)
    print lm_slice_stats

def align_tail_part(filepath, landmark_tail_idx_frac=0.56383, spinal_angle=0.12347):
    input_data = open_data(filepath)

    binarized_stack, bbox, eyes_stats = binarizator(input_data)

    print 'Bin saved...'
    binary_stack_stats, _ = object_counter(binarized_stack)
    largest_volume_region, largest_volume_region_bbox = extract_largest_area_data(input_data, binary_stack_stats, bb_side_offset=20)
    largest_volume_region.astype(np.float32).tofile(os.path.join(TMP_PATH, create_filename_with_shape(filepath, largest_volume_region.shape, prefix='-binary')))
    binarized_stack[largest_volume_region_bbox].tofile(os.path.join(TMP_PATH, create_filename_with_shape(filepath, binarized_stack[largest_volume_region_bbox].shape, prefix='-binarymask')))

    stack_statistics, _ = gather_statistics(largest_volume_region)
    eyes_stats = eyes_statistics(stack_statistics)

    eye_c = np.round([eyes_stats['com_x'].mean(), eyes_stats['com_y'].mean(), eyes_stats['com_z'].mean()]).astype(np.int32)
    print largest_volume_region.shape

    ext_vol_len = largest_volume_region.shape[0]
    eye1_idx_frac, eye2_idx_frac = eyes_stats['com_z'].values[0] / float(ext_vol_len),\
                                   eyes_stats['com_z'].values[1] / float(ext_vol_len)

    landmark_tail_idx = int(ext_vol_len * landmark_tail_idx_frac)

    lm_slice_stats, lm_labeled_slice = stats_at_slice(largest_volume_region, landmark_tail_idx)

    z_axis_vec = np.array([0., 0., -1.])
    tail_com_y, tail_com_z = lm_slice_stats['com_z'].values[0] - eye_c[1], landmark_tail_idx - eye_c[2]
    print 'tail_com_y = %s' % str(tail_com_y)
    print 'tail_com_z = %s' % str(tail_com_z)
    spinal_vec = np.array([0, tail_com_y, tail_com_z])

    rot_axis = np.cross(z_axis_vec, spinal_vec)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    theta = np.arccos(z_axis_vec.dot(spinal_vec)/(np.sqrt(z_axis_vec.dot(z_axis_vec)) * np.sqrt(spinal_vec.dot(spinal_vec))))

    if tail_com_y < 0:
        theta = -(theta - spinal_angle) if theta > spinal_angle else (spinal_angle - theta)
    else:
        theta = theta + spinal_angle

    data_rotated = rotate_around_vector(largest_volume_region, eye_c, rot_axis, theta, interp_order=3)

    return data_rotated

def archive_eyes(eyes_list, filepath):
    eyes_dump_out = open(filepath, 'wb')
    pickle.dump(eyes_list, eyes_dump_out)
    eyes_dump_out.close()

    print 'Eyes archived in: %s' % eyes_dump_out

def dearchive_eyes(filepath):
    eyes_file = open(filepath, 'rb')
    eyes_list = pickle.load(eyes_file)
    eyes_file.close()

    return eyes_list

def create_archive_eyes_path(filepath):
    name, bits, size, ext = parse_filename(filepath)
    return os.path.join(TMP_PATH, '%s_eyes.pkl' % name)

def check_depth_orientation():
    filepath = os.path.join(TMP_PATH, "fish243_aligned-eyes_32bit_320x320x996.raw")
    data = open_data(filepath)

    stack_statistics, _ = gather_statistics(data)
    eyes_stats = eyes_statistics(stack_statistics)

    flipped_data = flip_fish(data, eyes_stats)
    flipped_data.astype(np.float32).tofile(os.path.join(TMP_PATH, create_filename_with_shape(filepath, flipped_data.shape, prefix='-flipped')))

def check_vertical_orientation():
    filepath = os.path.join(TMP_PATH, "fish243_aligned-eyes_aligned-eyes-flipped_32bit_320x320x996.raw")
    data = open_data(filepath).copy()

    stack_statistics, _ = gather_statistics(data)
    eyes_stats = eyes_statistics(stack_statistics)

    eye_c = np.round([eyes_stats['com_x'].mean(), eyes_stats['com_y'].mean(), eyes_stats['com_z'].mean()]).astype(np.int32)
    print eye_c

    avg_eye_size = np.round(np.mean([eyes_stats['bb_width'].mean(), eyes_stats['bb_height'].mean(), eyes_stats['bb_depth'].mean()]))
    print avg_eye_size

    z_shift = avg_eye_size * 2

    #z -> y
    #y -> x
    head_slice_stats, head_labeled_slice = stats_at_slice(data, eye_c[2] - z_shift)

    central_head_part = head_labeled_slice[head_slice_stats['bb_z']:head_slice_stats['bb_z']+head_slice_stats['bb_depth'],\
                                           head_slice_stats['bb_y']:head_slice_stats['bb_y']+head_slice_stats['bb_height']]
    top_part, bottom_part = central_head_part[:central_head_part.shape[0]/2,:], central_head_part[central_head_part.shape[0]/2+1:,:]
    top_part, bottom_part = binary_opening(top_part, iterations=2), binary_opening(bottom_part, iterations=2)
    non_zeros_top, non_zeros_bottom = np.count_nonzero(top_part), np.count_nonzero(bottom_part)

    print non_zeros_top
    print non_zeros_bottom

    if non_zeros_top < non_zeros_bottom:
        print 'Volume should be vertically flipped.'
        data = data[:,::-1,:]

    data.astype(np.float32).tofile(os.path.join(TMP_PATH, create_filename_with_shape(filepath, data.shape, prefix='-vertically-flipped')))

    #central_head_part = binary_opening(central_head_part, iterations=0)
    #print head_slice_stats

    #plt.imshow(central_head_part)
    #plt.show()

def flip_horizontally():
    #filepath = os.path.join(TMP_PATH, "fish243_aligned-eyes_32bit_320x320x996.raw")
    filepath = os.path.join(TMP_PATH, "fish243_aligned-eyes_32bit_320x320x996.raw")
    input_data = open_data(filepath)

    total_stats = pd.DataFrame()

    binarized_stack, _, _ = binarizator(input_data)

    for idx, slice_data in enumerate(binarized_stack):
        if idx % 100 == 0 or idx == binarized_stack.shape[0]-1:
            print 'Slice #%d' % idx

        if slice_data.any():
            stats, labels = cell_counter(slice_data, slice_index=idx)
            total_stats = total_stats.append(stats, ignore_index=True)


    print total_stats
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(total_stats['slice_idx'], total_stats['com_z'])
    plt.show()

def align_eyes_centroids():
    filepath = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish243\\fish243_32bit_320x320x996.raw"

    eyes_stats_list = None
    eyes_archive_path = create_archive_eyes_path(filepath)

    print eyes_archive_path

    if not os.path.exists(eyes_archive_path):
        input_data = np.memmap(filepath, dtype=np.float32, shape=(996,320,320)).copy()
        stack_statistics, _ = gather_statistics(input_data)
        eyes_stats = eyes_statistics(stack_statistics)

        print 'Eyes stats:'
        print eyes_stats

        print 'Aligning by eyes...'
        eye1_com, eye2_com = np.array([eyes_stats['com_x'].values[0], eyes_stats['com_y'].values[0], eyes_stats['com_z'].values[0]]), \
                             np.array([eyes_stats['com_x'].values[1], eyes_stats['com_y'].values[1], eyes_stats['com_z'].values[1]])
        eyes_coms = [eye1_com, eye2_com]

        eye1_sbbox, eye2_sbbox = np.array([eyes_stats['bb_width'].values[0], eyes_stats['bb_height'].values[0], eyes_stats['bb_depth'].values[0]]), \
                                 np.array([eyes_stats['bb_width'].values[1], eyes_stats['bb_height'].values[1], eyes_stats['bb_depth'].values[1]])
        eyes_sizes = [eye1_sbbox, eye2_sbbox]

        archive_eyes([eyes_coms, eyes_sizes], eyes_archive_path)
    else:
        eyes_stats_list = dearchive_eyes(eyes_archive_path)

    print eyes_stats_list

    aligned_data, eye_c = align_by_eyes_centroids(filepath, eyes_stats_list[0])
    print eye_c
    aligned_data.astype(np.float32).tofile(os.path.join(TMP_PATH, create_filename_with_shape(filepath, aligned_data.shape, prefix='aligned-eyes')))

if __name__ == "__main__":
    #test_points_rotation()
    #test_points_rotation2()
    #test_points_rotation3()
    #test_fish_alignemnt()
    #get_centroid_at_slice()

    #align_eyes_centroids()
    #flip_horizontally()
    #check_depth_orientation()
    #check_vertical_orientation()
    filepath = os.path.join(TMP_PATH, "fish243_aligned-eyes_aligned-eyes-flipped_-vertically-flipped_32bit_320x320x996.raw")
    aligned_data = align_tail_part(filepath)
    aligned_data.astype(np.float32).tofile(os.path.join(TMP_PATH, create_filename_with_shape(filepath, aligned_data.shape, prefix='TOTAL')))

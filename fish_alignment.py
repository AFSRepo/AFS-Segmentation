import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import map_coordinates
from modules.tools.processing import binarizator
from modules.tools.morphology import object_counter, gather_statistics, extract_largest_area_data
from modules.segmentation.eyes import eyes_statistics
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
    eye_l, eye_r = [1.,3.,5.], [5.,2.,8.]
    eye_c = np.array([(lv + rv)/2. for lv,rv in zip(eye_l, eye_r)])

    p = np.array([eye_l, eye_r])
    p_s = p - eye_c
    print p_s
    p_s_d = p_s[0] - p_s[1]

    #alpha = -np.arctan(p_s_d[1] / p_s_d[0]) #angle(X, Y)
    #beta  = -np.arctan(p_s_d[2] / np.sqrt(p_s_d[0]*p_s_d[0] + p_s_d[1]*p_s_d[1]))  # angle(Z, X)
    beta = get_angle(p_s[0][[0,1]], p_s[0])
    alpha = get_angle(p_s[0][[0]], p_s[0][[0,1]])

    Rz = np.matrix([[np.cos(alpha), -np.sin(alpha), 0.], [np.sin(alpha), np.cos(alpha), 0.], [0., 0., 1.]])
    Ry = np.matrix([[np.cos(beta), 0., np.sin(beta)], [0., 1., 0.], [-np.sin(beta), 0., np.cos(beta)]])

    R = Ry * Rz
    p_r= np.array((R * p_s.T).T + eye_c)

    print p_s
    print (R * p_s.T).T

    print 'len old = %f' % (np.sqrt((p[0] - p[1]).dot((p[0] - p[1]))))
    print 'len new = %f' % (np.sqrt((p_r[0] - p_r[1]).dot((p_r[0] - p_r[1]))))

    ax.scatter(eye_c[0], eye_c[2] , c='r', marker='o')
    ax.scatter(p[:,0], p[:,2], c='b', marker='o')
    ax.scatter(p_r[:,0], p_r[:,2], c='g', marker='o')

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

def align_by_eyes_centroids(filepath, centroids):
    data = open_data(filepath)
    dims = data.shape
    eye_l, eye_r = centroids

    datac = data.copy()
    datac[eye_r[2], eye_r[1], eye_r[0]] = 255
    datac[eye_l[2], eye_l[1], eye_l[0]] = 255
    datac.astype(np.float32).tofile(os.path.join(TMP_PATH, create_filename_with_shape(filepath, datac.shape, prefix='marked')))

    eye_c = np.array([(lv + rv)/2. for lv,rv in zip(eye_l, eye_r)])
    p = np.array([eye_l, eye_r])
    p_s = p - eye_c

    print 'Angle and matrix creating...'
    alpha = get_angle(p_s[0][[0,1]], p_s[0])
    beta = -get_angle(p_s[0][[0]], p_s[0][[0,1]])

    Ry = np.matrix([[np.cos(alpha), 0., np.sin(alpha)], [0., 1., 0.], [-np.sin(alpha), 0., np.cos(alpha)]])
    Rz = np.matrix([[np.cos(beta), -np.sin(beta), 0.], [np.sin(beta), np.cos(beta), 0.], [0., 0., 1.]])

    R = Rz * Ry

    print p_s
    print (R * p_s.T).T

    p_sr = (R * p_s.T).T + eye_c

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

    return interp_data

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

    #landmark_slice = stats_at_slice(stack_data, lmt_slice_idx)

def align_eyes_centroids():
    filepath = "C:\\Users\\Administrator\\Documents\\ProcessedMedaka\\fish243\\fish243_32bit_320x320x996.raw"

    input_data = np.memmap(filepath, dtype=np.float32, shape=(996,320,320)).copy()

    stack_statistics, _ = gather_statistics(input_data)
    eyes_stats = eyes_statistics(stack_statistics)

    print 'Eyes stats:'
    print eyes_stats

    print 'Aligning by eyes...'
    eye1_stats, eye2_stats = np.array([eyes_stats['com_x'].values[0], eyes_stats['com_y'].values[0], eyes_stats['com_z'].values[0]]), \
                             np.array([eyes_stats['com_x'].values[1], eyes_stats['com_y'].values[1], eyes_stats['com_z'].values[1]])
    aligned_data = align_by_eyes_centroids(filepath, [eye1_stats[::-1], eye2_stats[::-1]])
    aligned_data.astype(np.float32).tofile(os.path.join(TMP_PATH, create_filename_with_shape(filepath, aligned_data.shape, prefix='aligned-eyes')))


if __name__ == "__main__":
    #test_points_rotation()
    #test_points_rotation2()
    #test_points_rotation3()
    #test_fish_alignemnt()
    #get_centroid_at_slice()
    align_eyes_centroids()

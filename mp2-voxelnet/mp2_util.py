"""
EE 298-F Machine Problem 2
Miranda, Neil Jon Louie P.
2007-46489
"""

import math
from shapely.affinity import rotate
from shapely.geometry import Point, Polygon

# Detection tasks
DET_TASKS = {'Car', 'Cyclist', 'Pedestrian'}


# Ranges for point cloud dimensions
MIN_X = {'Car': 0, 'Cyclist': 0, 'Pedestrian': 0}
MAX_X = {'Car': 70.4, 'Cyclist': 48, 'Pedestrian': 48}

MIN_Y = {'Car': -3, 'Cyclist': -3, 'Pedestrian': -3}
MAX_Y = {'Car': 1, 'Cyclist': 1, 'Pedestrian': 1}

MIN_Z = {'Car': -48, 'Cyclist': -24, 'Pedestrian': -24}
MAX_Z = {'Car': 48, 'Cyclist': 24, 'Pedestrian': 24}


# Parameters for the feature learning network (adjusted to allow for dense
# implementation)
# Original parameters in paper are one-fourth those listed below
VX = {'Car': 0.8, 'Cyclist': 0.8, 'Pedestrian': 0.8}
VY = {'Car': 0.8, 'Cyclist': 0.8, 'Pedestrian': 0.8}
VZ = {'Car': 0.8, 'Cyclist': 0.8, 'Pedestrian': 0.8}

# Original parameters in paper are one-half those listed below
T = {'Car': 50, 'Cyclist': 80, 'Pedestrian': 80}

# Original parameters in paper are double those listed below
DIM_X = {'Car': 88, 'Cyclist': 88, 'Pedestrian': 88}
DIM_Y = {'Car': 5, 'Cyclist': 5, 'Pedestrian': 5}
DIM_Z = {'Car': 120, 'Cyclist': 60, 'Pedestrian': 60}


# Anchor parameters
ANCHOR_SIZES = {
    'Car': [[3.9, 1.56, 1.6]],
    'Cyclist': [[1.76, 1.73, 0.6]],
    'Pedestrian': [[0.8, 1.73, 0.6]]
}

ANCHOR_Y = {'Car': -1.0, 'Cyclist': -0.6, 'Pedestrian': -0.6}

ANCHOR_ROTATIONS = {
    'Car': [0, math.pi * 0.25],
    'Cyclist': [0, math.pi * 0.25],
    'Pedestrian': [0, math.pi * 0.25]
}


# Functions to map points to their appropriate voxels
def get_voxel_coordinates(point, task):
    x = point[0]
    y = point[1]
    z = point[2]

    if not task in DET_TASKS:
        return None

    vx = int((x - MIN_X[task]) / VX[task])
    vy = int((y - MIN_Y[task]) / VY[task])
    vz = int((z - MIN_Z[task]) / VZ[task])
    return [vx, vy, vz]

def to_voxel_index(coord, task):
    if not task in DET_TASKS:
        return -1

    return int((coord[1] * DIM_X[task] + coord[0]) * DIM_Z[task] + coord[2])


# Functions to compute the intersection-over-union (IOU) value of two 3D
# bounding boxes
def create_polygon(dim_x, dim_z, ctr_x, ctr_z, rot):
    half_dim_x = 0.5 * dim_x
    half_dim_z = 0.5 * dim_z
    pol = Polygon([(ctr_x + half_dim_x, ctr_z - half_dim_z), \
                   (ctr_x + half_dim_x, ctr_z + half_dim_z), \
                   (ctr_x - half_dim_x, ctr_z + half_dim_z), \
                   (ctr_x - half_dim_x, ctr_z - half_dim_z)])
    rot_pol = rotate(pol, rot, origin=(ctr_x, ctr_z), use_radians=True)
    return pol

def volume(box):
    # Box format: [dim_x, dim_y, dim_z, ctr_x, ctr_y, ctr_z, rot_y]
    return box[0] * box[1] * box[2]

def intersection(boxA, boxB):
    # Box format: [dim_y, dim_z, dim_x, ctr_x, ctr_y, ctr_z, rot_y]
    polA = create_polygon(boxA[2], boxA[1], boxA[3], boxA[5], boxA[6])
    polB = create_polygon(boxB[2], boxB[1], boxB[3], boxB[5], boxB[6])
    inter = polA.intersection(polB)

    half_dim_y_A = 0.5 * boxA[0]
    min_y_A = boxA[4] - half_dim_y_A
    max_y_A = boxA[4] + half_dim_y_A

    half_dim_y_B = 0.5 * boxB[0]
    min_y_B = boxB[4] - half_dim_y_B
    max_y_B = boxB[4] + half_dim_y_B

    high = min(max_y_A, max_y_B)
    low = max(min_y_A, min_y_B)
    if high > low:
        return abs(inter.area) * (high - low)
    else:
        return 0.0

def union(boxA, boxB, intersection):
    # Box format: [dim_y, dim_z, dim_x, ctr_x, ctr_y, ctr_z, rot_y]
    union = volume(boxA) + volume(boxB) - intersection
    return union

def compute_iou(boxA, boxB):
    inter_area = intersection(boxA, boxB)
    if inter_area == 0:
        return 0.0
    else:
        return float(inter_area) / \
               float(union(boxA, boxB, inter_area) + 1e-9)

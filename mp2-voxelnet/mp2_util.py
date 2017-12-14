"""
EE 298-F Machine Problem 2
Miranda, Neil Jon Louie P.
2007-46489
"""

# Detection tasks
DET_TASKS = {'Car', 'Cyclist', 'Pedestrian'}

# Ranges for point cloud dimensions
MIN_X = {'Car': 0, 'Cyclist': 0, 'Pedestrian': 0}
MAX_X = {'Car': 70.4, 'Cyclist': 48, 'Pedestrian': 48}

MIN_Y = {'Car': -48, 'Cyclist': -24, 'Pedestrian': -24}
MAX_Y = {'Car': 48, 'Cyclist': 24, 'Pedestrian': 24}

MIN_Z = {'Car': -3, 'Cyclist': -3, 'Pedestrian': -3}
MAX_Z = {'Car': 1, 'Cyclist': 1, 'Pedestrian': 1}

# Parameters for the feature learning network (adjusted to allow for dense
# implementation)
# Original parameters in paper are one-fourth those listed below
VD = {'Car': 0.8, 'Cyclist': 0.8, 'Pedestrian': 0.8}
VH = {'Car': 0.8, 'Cyclist': 0.8, 'Pedestrian': 0.8}
VW = {'Car': 0.8, 'Cyclist': 0.8, 'Pedestrian': 0.8}

# Original parameters in paper are one-half those listed below
T = {'Car': 50, 'Cyclist': 80, 'Pedestrian': 80}

# Original parameters in paper are double those listed below
DIM_D = {'Car': 5, 'Cyclist': 5, 'Pedestrian': 5}
DIM_H = {'Car': 120, 'Cyclist': 60, 'Pedestrian': 60}
DIM_W = {'Car': 88, 'Cyclist': 88, 'Pedestrian': 88}

def get_voxel_coordinates(point, task):
    x = point[0]
    y = point[1]
    z = point[2]

    if not task in DET_TASKS:
        return None

    vx = int((x - MIN_X[task]) / VW[task])
    vy = int((y - MIN_Y[task]) / VH[task])
    vz = int((z - MIN_Z[task]) / VD[task])
    return [vx, vy, vz]

def to_voxel_index(coord, task):
    if not task in DET_TASKS:
        return -1

    return int((coord[2] * DIM_D[task] + coord[1]) * DIM_H[task] + coord[0])

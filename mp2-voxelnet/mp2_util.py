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

MIN_Y = {'Car': -40, 'Cyclist': -20, 'Pedestrian': -20}
MAX_Y = {'Car': 40, 'Cyclist': 20, 'Pedestrian': 20}

MIN_Z = {'Car': -3, 'Cyclist': -3, 'Pedestrian': -3}
MAX_Z = {'Car': 1, 'Cyclist': 1, 'Pedestrian': 1}

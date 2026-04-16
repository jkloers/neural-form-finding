import numpy as np
from .core import UnitPattern

# RDQK pattern (Rigidly Deployable Quadrilateral Kirigami)
unit_RDQK_0 = UnitPattern(
    vertices=[
        [0, 0], #face 0
        [1, 0],
        [1,1],
        [0,1],
        [1,0], #face 1
        [2,0],
        [2,1],
        [1,1],
        [1,1], #face 2
        [2,1],
        [2,2],
        [1,2],
        [0,1], #face 3
        [1,1],
        [1,2],
        [0,2]
    ],
    faces=[
        [0, 1, 2, 3], # Face 0
        [4, 5, 6, 7], # Face 1
        [8, 9, 10, 11], # Face 2
        [12, 13, 14, 15]  # Face 3
    ],

    internal_hinges=[
        {'face1': 0, 'face2': 1, 'vertex1': 1, 'vertex2': 4, 'rest_angle': np.pi/4, 'stiffness': 1.0},
        {'face1': 1, 'face2': 2, 'vertex1': 7, 'vertex2': 8, 'rest_angle': np.pi/4, 'stiffness': 1.0},
        {'face1': 2, 'face2': 3, 'vertex1': 11, 'vertex2': 14, 'rest_angle': np.pi/4, 'stiffness': 1.0},
        {'face1': 3, 'face2': 0, 'vertex1': 13, 'vertex2': 2, 'rest_angle': np.pi/4, 'stiffness': 1.0}

    ],

    shift_vectors=None,

    external_hinges=[
        {'type': 'x', 'face1': 1, 'face2_offset': -1, 'vertex1': 6, 'vertex2_offset': -3},
        {'type': 'x', 'face1': 2, 'face2_offset': 1, 'vertex1': 9, 'vertex2_offset': 3},
        {'type': 'y', 'face1': 2, 'face2_offset': -1, 'vertex1': 10, 'vertex2_offset': -5},
        {'type': 'y', 'face1': 3, 'face2_offset': 1, 'vertex1': 15, 'vertex2_offset': -15}
    ]
)

unit_RDQK_D = UnitPattern(
    vertices=[[np.sqrt(2)/2 , 0], #face 0
        [np.sqrt(2), np.sqrt(2)/2],
        [np.sqrt(2)/2, np.sqrt(2)],
        [0, np.sqrt(2)/2],
        [np.sqrt(2), np.sqrt(2)/2], #face 1
        [3*np.sqrt(2)/2, 0],
        [2*np.sqrt(2), np.sqrt(2)/2],
        [3*np.sqrt(2)/2, np.sqrt(2)],
        [3*np.sqrt(2)/2, np.sqrt(2)], #face 2
        [2*np.sqrt(2), 3*np.sqrt(2)/2],
        [3*np.sqrt(2)/2, 2*np.sqrt(2)],
        [np.sqrt(2), 3*np.sqrt(2)/2],
        [0, 3*np.sqrt(2)/2], #face 3
        [np.sqrt(2)/2, np.sqrt(2)],
        [np.sqrt(2), 3*np.sqrt(2)/2],
        [np.sqrt(2)/2, 2*np.sqrt(2)]
    ],
    faces=[
        [0, 1, 2, 3], # Face 0
        [4, 5, 6, 7], # Face 1
        [8, 9, 10, 11], # Face 2
        [12, 13, 14, 15]  # Face 3
    ],
    internal_hinges=[
        {'face1': 0, 'face2': 1, 'vertex1': 1, 'vertex2': 4, 'rest_angle': np.pi/4, 'stiffness': 1.0},
        {'face1': 1, 'face2': 2, 'vertex1': 7, 'vertex2': 8, 'rest_angle': np.pi/4, 'stiffness': 1.0},
        {'face1': 2, 'face2': 3, 'vertex1': 11, 'vertex2': 14, 'rest_angle': np.pi/4, 'stiffness': 1.0},
        {'face1': 3, 'face2': 0, 'vertex1': 13, 'vertex2': 2, 'rest_angle': np.pi/4, 'stiffness': 1.0}

    ],

    shift_vectors=None,

    external_hinges=[
        {'type': 'x', 'face1': 1, 'face2_offset': -1, 'vertex1': 6, 'vertex2_offset': -3},
        {'type': 'x', 'face1': 2, 'face2_offset': 1, 'vertex1': 9, 'vertex2_offset': 3},
        {'type': 'y', 'face1': 2, 'face2_offset': -1, 'vertex1': 10, 'vertex2_offset': -5},
        {'type': 'y', 'face1': 3, 'face2_offset': 1, 'vertex1': 15, 'vertex2_offset': -15}
    ]
)





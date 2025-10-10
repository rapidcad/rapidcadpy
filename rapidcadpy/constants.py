ALL_COMMANDS = ["Line", "Arc", "Circle", "EOS", "SOS", "Ext", "Sketch", "EOL"]
LINE_IDX = ALL_COMMANDS.index("Line")  # 0
ARC_IDX = ALL_COMMANDS.index("Arc")  # 1
CIRCLE_IDX = ALL_COMMANDS.index("Circle")  # 2
EOS_IDX = ALL_COMMANDS.index("EOS")  # 3
SOL_IDX = ALL_COMMANDS.index("SOS")  # 4
EXT_IDX = ALL_COMMANDS.index("Ext")  # 5
SKETCH_IDX = ALL_COMMANDS.index("Sketch")  # 6
EOL_IDX = ALL_COMMANDS.index("EOL")  # 7

PAD_VAL = -1
UNUSED_PARAM = 257
N_COMMANDS = len(ALL_COMMANDS)
N_ARGS = 15
QUANTIZATION_SCALE = 256
N_BIT = 8

COMMAND_MASK = {
    LINE_IDX: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ARC_IDX: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    CIRCLE_IDX: [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    SKETCH_IDX: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    EXT_IDX: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    EOL_IDX: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    SOL_IDX: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    EOS_IDX: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

COMMAND_PARAMETER_COUNTS = {
    LINE_IDX + QUANTIZATION_SCALE + 1: 2,
    ARC_IDX + QUANTIZATION_SCALE + 1: 4,
    CIRCLE_IDX + QUANTIZATION_SCALE + 1: 3,
    EOS_IDX + QUANTIZATION_SCALE + 1: 0,
    SOL_IDX + QUANTIZATION_SCALE + 1: 0,
    EXT_IDX + QUANTIZATION_SCALE + 1: 1,
    SKETCH_IDX + QUANTIZATION_SCALE + 1: 6,
    EOL_IDX + QUANTIZATION_SCALE + 1: 0,
}

EXTRUDE_OPERATIONS = [
    "NewBodyFeatureOperation",
    "JoinFeatureOperation",
    "CutFeatureOperation",
    "IntersectFeatureOperation",
]
EXTENT_TYPE = [
    "OneSideFeatureExtentType",
    "SymmetricFeatureExtentType",
    "TwoSidesFeatureExtentType",
]
NODE_TYPES = ["primitive"]
CONSTRAINT_TYPES = [
    "horizontal",
    "vertical",
    "endtostartcoincidence",
    "perpendicular",
    "parallel",
]
CONSTRAINTS = [
    ("primitive", "horizontal", "primitive"),
    ("primitive", "vertical", "primitive"),
    ("primitive", "endtostartcoincidence", "primitive"),
    ("primitive", "perpendicular", "primitive"),
    ("primitive", "parallel", "primitive"),
]


PRECISION = 1e-5
NORM_FACTOR = (
    0.75  # scale factor for normalization to prevent overflow during augmentation
)

MAX_N_EXT = 10  # maximum number of extrusion
MAX_N_LOOPS = 6  # maximum number of loops per sketch
MAX_N_CURVES = 15  # maximum number of curves per loop
MAX_TOTAL_LEN = 70  # maximum cad sequence length
MAX_NODES = 200
MAX_EDGES = 800
ARGS_DIM = 256

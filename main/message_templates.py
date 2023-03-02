LEGS_ORDER = ["leg1", "leg3", "leg4", "leg2"]

BASE_MOTION_TEMPLATE = {
    "pose": {
        "position": {
            "x": None,
            "y": None,
            "z": 0.22365788448808482
        },
        "orientation": { #(z^2+w^2 = 1) !!!!!
            "x": 0.0,
            "y": 0.0,
            "z": None,
            "w": None
        }
    },
    "twist": {
        "linear": {
            "x": None,
            "y": None,
            "z": -0.00011574711882466314
        },
        "angular": {
            "x": 0.0,
            "y": 0.0,
            "z": None
        }
    },
    "accel": {
        "linear": {
            "x": 0.898957144550804,
            "y": -0.02865356300291391,
            "z": -1.7392893781453231e-15
        },
        "angular": {
            "x": 0.0,
            "y": 0.0,
            "z": -0.4473738291300291
        }
    }
}

LEG_TEMPLATE = {
    "pose": {
        "position": {
            "x": None,
            "y": None,
            "z": None
        },
        "orientation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "w": 1.0
        }
    },
    "twist": {
        "linear": {
            "x": None,
            "y": None,
            "z": None
        },
        "angular": {
            "x": 0.0,
            "y": 0.0,
            "z": None
        }
    },
    "accel": {
        "linear": {
            "x": -0.8989571445508029,
            "y": 0.028653563002905925,
            "z": 1.7392893781453231e-15
        },
        "angular": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.4473738291300291
        }
    },
    "contact": "UNSET"
}

MESSAGE_ORDER = [
    [
        "base_motion",
        "pose",
        "orientation",
        "w"
    ],
    [
        "base_motion",
        "pose",
        "orientation",
        "z"
    ],
    [
        "base_motion",
        "pose",
        "position",
        "x"
    ],
    [
        "base_motion",
        "pose",
        "position",
        "y"
    ],
    [
        "base_motion",
        "twist",
        "angular",
        "z"
    ],
    [
        "base_motion",
        "twist",
        "linear",
        "x"
    ],
    [
        "base_motion",
        "twist",
        "linear",
        "y"
    ],
    [
        "leg1",
        "pose",
        "position",
        "x"
    ],
    [
        "leg1",
        "pose",
        "position",
        "y"
    ],
    [
        "leg1",
        "pose",
        "position",
        "z"
    ],
    [
        "leg1",
        "twist",
        "angular",
        "z"
    ],
    [
        "leg1",
        "twist",
        "linear",
        "x"
    ],
    [
        "leg1",
        "twist",
        "linear",
        "y"
    ],
    [
        "leg1",
        "twist",
        "linear",
        "z"
    ],
    [
        "leg2",
        "pose",
        "position",
        "x"
    ],
    [
        "leg2",
        "pose",
        "position",
        "y"
    ],
    [
        "leg2",
        "pose",
        "position",
        "z"
    ],
    [
        "leg2",
        "twist",
        "angular",
        "z"
    ],
    [
        "leg2",
        "twist",
        "linear",
        "x"
    ],
    [
        "leg2",
        "twist",
        "linear",
        "y"
    ],
    [
        "leg2",
        "twist",
        "linear",
        "z"
    ],
    [
        "leg3",
        "pose",
        "position",
        "x"
    ],
    [
        "leg3",
        "pose",
        "position",
        "y"
    ],
    [
        "leg3",
        "pose",
        "position",
        "z"
    ],
    [
        "leg3",
        "twist",
        "angular",
        "z"
    ],
    [
        "leg3",
        "twist",
        "linear",
        "x"
    ],
    [
        "leg3",
        "twist",
        "linear",
        "y"
    ],
    [
        "leg3",
        "twist",
        "linear",
        "z"
    ],
    [
        "leg4",
        "pose",
        "position",
        "x"
    ],
    [
        "leg4",
        "pose",
        "position",
        "y"
    ],
    [
        "leg4",
        "pose",
        "position",
        "z"
    ],
    [
        "leg4",
        "twist",
        "angular",
        "z"
    ],
    [
        "leg4",
        "twist",
        "linear",
        "x"
    ],
    [
        "leg4",
        "twist",
        "linear",
        "y"
    ],
    [
        "leg4",
        "twist",
        "linear",
        "z"
    ]
]

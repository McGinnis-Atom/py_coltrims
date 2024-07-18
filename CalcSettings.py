from __future__ import annotations
from Constants import CONSTANTS
from numba.experimental import jitclass
from numba import boolean, int64, float64


@jitclass([
    ("shiftX", float64),
    ("shiftY", float64),
    ("shiftTOF", float64),
    ("stretchX", float64),
    ("stretchY", float64),
    ("stretchTotal", float64),
    ("rotateDeg", float64),
    ("mirrorX", boolean),
    ("mirrorY", boolean),
    ("shiftPX", float64),
    ("shiftPY", float64),
    ("shiftPZ", float64),
    ("stretchPX", float64),
    ("stretchPY", float64),
    ("stretchPZ", float64),
    ("stretchPTotal", float64),
    ("mirrorPX", boolean),
    ("mirrorPY", boolean),
    ("shiftThenRotate", boolean),
])
class CalcSettings:
    def __init__(self, shiftX:    float = 0.   , shiftY:    float = 0.,    shiftTOF:     float = 0.,    \
                       stretchX:  float = 1.   , stretchY:  float = 1.,    stretchTotal: float = 1.,    \
                       rotateDeg: float = 0.   , mirrorX:   bool  = False, mirrorY:      bool  = False, \
                       shiftPX:   float = 0.   , shiftPY:   float = 0.,    shiftPZ:      float = 0.,    \
                       stretchPX: float = 1.   , stretchPY: float = 1.,    stretchPZ:    float = 1., stretchPTotal: float = 1., \
                       mirrorPX:  bool  = False, mirrorPY:  bool  = False, shiftThenRotate: bool=True   \
                ) -> None:
                
        self.shiftX               = shiftX          # mm
        self.shiftY               = shiftY          # mm
        self.shiftTOF=self.shiftT = shiftTOF        # ns
        self.stretchX             = stretchX        # 1
        self.stretchY             = stretchY        # 1
        self.stretchTotal         = stretchTotal    # 1
        self.rotateDeg            = rotateDeg       # deg
        self.mirrorX              = mirrorX         # bool
        self.mirrorY              = mirrorY         # bool
        self.shiftPX              = shiftPX         # a.u.
        self.shiftPY              = shiftPY         # a.u.
        self.shiftPZ              = shiftPZ         # a.u.
        self.stretchPX            = stretchPX       # 1
        self.stretchPY            = stretchPY       # 1
        self.stretchPZ            = stretchPZ       # 1
        self.stretchPTotal        = stretchPTotal   # 1
        self.mirrorPX             = mirrorPX        # bool
        self.mirrorPY             = mirrorPY        # bool
        self.shiftThenRotate      = shiftThenRotate # bool
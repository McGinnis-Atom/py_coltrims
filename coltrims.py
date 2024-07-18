from __future__ import annotations
from Constants import CONSTANTS
from Spectrometer import Spectrometer
from CalcSettings import CalcSettings
from Particle import Particle, Ion, Electron, ParticleList
from Reaction import Reaction








class CoordinateSystem:
    from typing import overload, Union
    from numpy import ndarray
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z
    
    def __init__(self, vec1, vec2):
        """
        initalize a right-handed coordinate system with two vectors.
        vec1 defines the z-Axis
        vec1 and vec2 will be orthogonal to the y-axis
        """

        import numpy as np

        vecOne = np.asarray(vec1)
        self.vecZ = vecOne
        self.vecZ = self.vecZ / np.sqrt(self.vecZ[0]**2 + self.vecZ[1]**2 + self.vecZ[2]**2)
        
        vecTwo = np.asarray(vec2)
        if len(vecTwo.shape) == 1:
            vecTwo = np.asarray([vec2]*vecOne.shape[-1]).T
        self.vecY = np.cross(self.vecZ, vecTwo, axis=0)
        self.vecY = self.vecY / np.sqrt(self.vecY[0]**2 + self.vecY[1]**2 + self.vecY[2]**2)

        self.vecX = np.cross(self.vecY, self.vecZ, axis=0)
        self.vecX = self.vecX / np.sqrt(self.vecX[0]**2 + self.vecX[1]**2 + self.vecX[2]**2)
    
    @overload
    def projectVector(self, other: Particle) -> Particle: ...
    @overload
    def projectVector(self, other: ndarray) -> ndarray: ...
    
    def projectVector(self, other: Union[Particle, ndarray]) -> Union[Particle, ndarray]:
        from numpy import ndarray, array
        if isinstance(other, Particle): 
            newX = other * self.vecX
            newY = other * self.vecY
            newZ = other * self.vecZ
            return Particle(px=newX, py=newY, pz=newZ, recalculateMomentum=False)
        if isinstance(other, ndarray):
            newX = other * self.vecX
            newY = other * self.vecY
            newZ = other * self.vecZ
            return array(newX, newY, newZ)
        else:
            raise NotImplementedError


    @overload
    def __call__(self, other: Particle) -> Particle: ...
    @overload
    def __call__(self, other: ndarray) -> ndarray: ...
    def __call__(self, other: Union[Particle, ndarray]) -> Union[Particle, ndarray]:
        return self.projectVector(other)
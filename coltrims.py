from __future__ import annotations
from Constants import CONSTANTS
from Spectrometer import Spectrometer
from CalcSettings import CalcSettings
from Particle import Particle






class Reaction:
    from typing import Optional
    import numpy as np
    IS_ION = 1
    IS_ELECTRON = 2

    def __init__(self, ionSpectrometer: Optional[Spectrometer] = None, electronSpectrometer: Optional[Spectrometer] = None, \
                       ionCalcSettings: Optional[CalcSettings] = None, electronCalcSettings: Optional[CalcSettings] = None):
        self._ionsArr   = ParticleList()
        self._elecArr   = ParticleList()
        self._ionsSpec  = ionSpectrometer
        self._elecSpec  = electronSpectrometer
        self._ionsCalcSettings = ionCalcSettings
        self._elecCalcSettings = electronCalcSettings
    
    @property
    def numIons(self):
        return len(self._ionsArr)
    
    @property
    def numElec(self):
        return len(self._elecArr)
    
    @property
    def ionsArr(self):
        return self._ionsArr
    
    @property
    def i(self):
        return self._ionsArr
    
    @property
    def r(self):
        return self._ionsArr
    
    @property
    def elecArr(self):
        return self._elecArr
    
    @property
    def e(self):
        return self._elecArr
    
    def add_ion(self, x: np.ndarray, y: np.ndarray, tof: np.ndarray, \
                      m: np.ndarray|float|int, q: np.ndarray|float|int, \
                      tofMean:      Optional[float|int]    = None, \
                      spectrometer: Optional[Spectrometer] = None, \
                      calcSettings: Optional[CalcSettings] = None, *,\
                      dtype:  np.typing.DTypeLike           = np.double, \
                      ctype:  np.typing.DTypeLike           = np.cdouble) -> None:
        if spectrometer is None and self._ionsSpec is not None:
            spectrometer = self._ionsSpec
        if calcSettings is None:
            if self._ionsCalcSettings is not None:
                calcSettings = self._ionsCalcSettings
            else:
                calcSettings = CalcSettings()
        self._ionsArr += Ion(x=x, y=y, tof=tof, m=m, q=q, tofMean=tofMean, spectrometer=spectrometer, calcSettings=calcSettings, dtype=dtype, ctype=ctype)

    def add_elec(self, x: np.ndarray, y: np.ndarray, tof: np.ndarray, \
                       spectrometer: Optional[Spectrometer] = None, \
                       calcSettings: Optional[CalcSettings] = None, *,\
                       dtype:  np.typing.DTypeLike          = np.double, \
                       ctype:  np.typing.DTypeLike          = np.cdouble) -> None:
        if spectrometer is None and self._elecSpec is not None:
            spectrometer = self._elecSpec
        if calcSettings is None:
            if self._elecCalcSettings is not None:
                calcSettings = self._elecCalcSettings
            else:
                calcSettings = CalcSettings()
        self._elecArr += Electron(x=x, y=y, tof=tof, spectrometer=spectrometer, calcSettings=calcSettings, dtype=dtype, ctype=ctype)

    def setIonSpectrometer(self, spectrometer: Spectrometer, applyToAll: bool = True):
        self._ionsSpec = spectrometer
        if applyToAll:
            for ion in self._ionsArr:
                ion.spectrometer = spectrometer

    def setElectronSpectrometer(self, spectrometer: Spectrometer, applyToAll: bool = True):
        self._elecSpec = spectrometer
        if applyToAll:
            for electron in self._elecArr:
                electron.spectrometer = spectrometer

    def setIonSettings(self, settings: CalcSettings, applyToAll: bool = True):
        self._ionsCalcSettings = settings
        if applyToAll:
            for ion in self._ionsArr:
                ion.calcSettings = settings
    
    def setElectronSettings(self, settings: CalcSettings, applyToAll: bool = True):
        self._elecCalcSettings = settings
        if applyToAll:
            for electron in self._elecArr:
                electron.calcSettings = settings

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
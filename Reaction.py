from __future__ import annotations
from Constants import CONSTANTS
from Spectrometer import Spectrometer
from CalcSettings import CalcSettings
from Particle import ParticleList, Particle, Electron, Ion

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
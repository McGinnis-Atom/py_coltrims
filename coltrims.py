from __future__ import annotations

class CONSTANTS:
    MM_SI_TO_AU       =  5.291_772_109_03e-8
    VCM_SI_TO_AU      =  5.142_206_747_63e9
    NS_SI_TO_AU       =  2.418_884_326_585_7e-8
    EV_SI_TO_AU       = 27.211_386_245_988
    GAUSS_TO_NS = lambda x: 1e9*(2*3.14159265359*9.1093837e-31)/(1.60217733e-19*x*1e-4)

class Spectrometer:
    from typing import Optional, List
    
    def __init__(self, lengths:        Optional[List[float|int]] = None, \
                       electicFields:  Optional[List[float|int]] = None, \
                       gyrationPeriod: Optional[float|int]                    = None, \
                       magneticField:  Optional[float|int]                    = None, \
                ) -> None:
        """
        lengths:        array of spectrometer lengths       in mm
        electicFields:  array of spectrometer electicFields in V/cm
        gyrationPeriod: float of magneticField gyration periode in ns
        magneticField:  float of magneticField in Gauss (set if gyrationPeriod is not set)
        """
        self.lengths        = lengths        if lengths        is not None else list()
        self.electricFields = electicFields  if electicFields  is not None else list()
        self._gyrationPeriod = gyrationPeriod if gyrationPeriod is not None else \
                               CONSTANTS.GAUSS_TO_NS(magneticField) if magneticField is not None else None
    
    def __len__(self):
        return len(self.lengths)
    
    def __iter__(self):
        self.iterIndex = 0
        return self
    
    def __next__(self):
        if self.iterIndex < len(self):
            val = (self.lengths[self.iterIndex], self.electricFields[self.iterIndex])
            self.iterIndex += 1
            return val
        else:
            raise StopIteration
    
    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError
        if index < -len(self) or index >= len(self):
            raise IndexError
        
        return (self.lengths[index], self.electricFields[index])

    @property
    def gyrationPeriod(self) -> float|int|None:
        if self._gyrationPeriod is None:
                return None
        return self._gyrationPeriod
    @gyrationPeriod.setter
    def gyrationPeriod(self, gyrationPeriod: float|int|None):
        self._gyrationPeriod = gyrationPeriod
    @property
    def magneticField(self) -> float|int|None:
        if self._gyrationPeriod is None:
            return None
        return CONSTANTS.GAUSS_TO_NS(self._gyrationPeriod)
    @magneticField.setter
    def magneticField(self, magneticField: float|int|None):
        if magneticField is None:
            self._gyrationPeriod = None
        else:
            self._gyrationPeriod = CONSTANTS.GAUSS_TO_NS(magneticField)

    def addRegion(self, length: float|int, electricField: float|int):
        """
        Append a field region to the spectrometer.
        length:        lenght of region in mm
        electricField: field strength   in V/cm
        """
        if electricField<0:
            print("\033[93mDeaccelation Field region added.\033[0m")
        self.lengths.append(length)
        self.electricFields.append(electricField)

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


class Particle:
    import numpy as np
    from typing import Optional
    
    def __init__(self, x:            Optional[np.ndarray]           = None, \
                       y:            Optional[np.ndarray]           = None, \
                       tof:          Optional[np.ndarray]           = None, \
                       m:            Optional[np.ndarray|int|float] = None, \
                       q:            Optional[np.ndarray|int|float] = None, \
                       px:           Optional[np.ndarray]           = None, \
                       py:           Optional[np.ndarray]           = None, \
                       pz:           Optional[np.ndarray]           = None, \
                       p:            Optional[np.ndarray]           = None, \
                       energy:       Optional[np.ndarray]           = None, \
                       spectrometer: Optional[Spectrometer]         = None, \
                       calcSettings: Optional[CalcSettings]         = None, \
                       isIonSide: bool                              = True, \
                       dtype:  np.typing.DTypeLike                  = np.double, \
                       ctype:  np.typing.DTypeLike                  = np.cdouble, \
                ) -> None:
        import numpy as np
        self._x      = None if x      is None else np.array(x, dtype=dtype)      # mm
        self._y      = None if y      is None else np.array(y, dtype=dtype)      # mm
        self._tof    = None if tof    is None else np.array(tof, dtype=dtype)    # ns
        self._q      = None if q      is None else np.array(q, dtype=dtype)      # a.u.
        self._m      = None if m      is None else np.array(m, dtype=dtype)      # a.u.
                                                                    # 
        self._px     = None if px     is None else np.array(px, dtype=dtype)     # a.u.
        self._py     = None if py     is None else np.array(py, dtype=dtype)     # a.u.
        self._pz     = None if pz     is None else np.array(pz, dtype=dtype)     # a.u.
        self._p      = None if p      is None else np.array(p, dtype=dtype)      # a.u.
        self._energy = None if energy is None else np.array(energy, dtype=dtype) # eV
        
        self._spectrometer          = spectrometer
        self._recalculateMomentum   = True
        self._electricFieldPolarity = 1 if isIonSide else -1
        self._mirrorYElectron       = 1 if isIonSide else -1
        self._calcSettings          = calcSettings

        self._tofMean = None
        
        self._dtype = dtype
        self._ctype = ctype
    
    def setUpdateMomentum(self):
        self._recalculateMomentum = True
    
    @property
    def tofMean(self) -> float:
        if self._tofMean is None:
            raise ValueError("The variable 'tofMean' is not yet set!")
        return self._tofMean
    @tofMean.setter
    def tofMean(self, tofMean: float) -> None:
        self._tofMean = tofMean
        self._recalculateMomentum = True

    @property
    def x(self) -> np.ndarray:
        """
        x-position of particle in mm
        """
        if self._x is None:
            raise ValueError("The variable 'x' is not yet set!")
        return self._x
    @x.setter
    def x(self, x: np.ndarray) -> None:
        import numpy as np
        self._x = np.array(x, dtype=self._dtype)
        self._recalculateMomentum = True
    
    @property
    def y(self) -> np.ndarray:
        """
        y-position of particle in mm
        """
        if self._y is None:
            raise ValueError("The variable 'y' is not yet set!")
        return self._y
    @y.setter
    def y(self, y: np.ndarray) -> None:
        import numpy as np
        self._y = np.array(y, dtype=self._dtype)
        self._recalculateMomentum = True
    
    @property
    def tof(self) -> np.ndarray:
        """
        tof of particle in ns
        """
        if self._tof is None:
            raise ValueError("The variable 'tof' is not tofet set!")
        return self._tof
    @tof.setter
    def tof(self, tof: np.ndarray) -> None:
        import numpy as np
        self._tof = np.array(tof, dtype=self._dtype)
        self._recalculateMomentum = True
    
    @property
    def m(self) -> np.ndarray:
        """
        mass of particle in a.u.
        """
        if self._m is None:
            raise ValueError("The variable 'm' is not yet set!")
        return self._m
    @m.setter
    def m(self, m: np.ndarray) -> None:
        import numpy as np
        self._m = np.array(m, dtype=self._dtype)
        self._recalculateMomentum = True
    
    @property
    def q(self) -> np.ndarray:
        """
        electric charge of particle in a.u.
        """
        if self._q is None:
            raise ValueError("The variable 'q' is not yet set!")
        return self._q
    @q.setter
    def q(self, q: np.ndarray) -> None:
        import numpy as np
        self._q = np.array(q, dtype=self._dtype)
        self._recalculateMomentum = True
    
    def px(self) -> np.ndarray:
        """
        x-momentum of particle in a.u.
        """
        if self._px is None or self._recalculateMomentum:
            self.calcMomentum()
        return self._px
    def py(self) -> np.ndarray:
        """
        y-momentum of particle in a.u.
        """
        if self._py is None or self._recalculateMomentum:
            self.calcMomentum()
        return self._py
    def pz(self) -> np.ndarray:
        """
        z-momentum of particle in a.u.
        """
        if self._pz is None or self._recalculateMomentum:
            self.calcMomentum()
        return self._pz
    def p(self) -> np.ndarray:
        """
        absolute momentum of particle in a.u.
        """
        if self._p is None or self._recalculateMomentum:
            self.calcMomentum()
        return self._p
    def energy(self) -> np.ndarray:
        """
        kinetic energy of particle in eV
        """
        if self._energy is None or self._recalculateMomentum:
            self.calcMomentum()
        return self._energy
    
    @property
    def spectrometer(self) -> np.ndarray:
        if self._spectrometer is None:
            raise ValueError("The variable 'spectrometer' is not yet set!")
        return self._spectrometer
    @spectrometer.setter
    def spectrometer(self, spectrometer: Spectrometer) -> None:
        self._spectrometer = spectrometer
        self._recalculateMomentum = True
    @property
    def calcSettings(self) -> np.ndarray:
        if self._calcSettings is None:
            raise ValueError("The variable 'calcSettings' is not yet set!")
        return self._calcSettings
    @calcSettings.setter
    def calcSettings(self, calcSettings: calcSettings) -> None:
        self._calcSettings = calcSettings
        self._recalculateMomentum = True
    
    def calcMomentum(self, spectrometer: Optional[Spectrometer] = None, \
                           calcSettings: Optional[CalcSettings] = None  \
                    ) -> None:
        import numpy as np
        if spectrometer is None:
            spectrometer = self.spectrometer
        if calcSettings is None:
            calcSettings = self.calcSettings

        if spectrometer.gyrationPeriod is not None and spectrometer.gyrationPeriod != 0:
            omega = 2*np.pi*CONSTANTS.NS_SI_TO_AU / spectrometer.gyrationPeriod
        rotationAngle = np.deg2rad(calcSettings.rotateDeg)
        
        x = self.x   / CONSTANTS.MM_SI_TO_AU
        y = self.y   / CONSTANTS.MM_SI_TO_AU
        t = self.tof / CONSTANTS.NS_SI_TO_AU
        
        # Mirror detector
        if calcSettings.mirrorX:
            x *= -1
        if calcSettings.mirrorY:
            y *= -1
        
        # Shift and rotate detector
        if calcSettings.shiftThenRotate:
            x += calcSettings.shiftX / CONSTANTS.MM_SI_TO_AU
            y += calcSettings.shiftY / CONSTANTS.MM_SI_TO_AU
            t += calcSettings.shiftT / CONSTANTS.NS_SI_TO_AU
            
            x, y = x*np.cos(rotationAngle)-y*np.sin(rotationAngle), x*np.sin(rotationAngle)+y*np.cos(rotationAngle)
        
        else:
            x, y = x*np.cos(rotationAngle)-y*np.sin(rotationAngle), x*np.sin(rotationAngle)+y*np.cos(rotationAngle)
            
            x += calcSettings.shiftX / CONSTANTS.MM_SI_TO_AU
            y += calcSettings.shiftY / CONSTANTS.MM_SI_TO_AU
            t += calcSettings.shiftT / CONSTANTS.NS_SI_TO_AU
        
        
        # Stretch detector
        x *= calcSettings.stretchX * calcSettings.stretchTotal
        y *= calcSettings.stretchY * calcSettings.stretchTotal
        
        
        # Calculate momentum
        if spectrometer.gyrationPeriod is not None and spectrometer.gyrationPeriod != 0:
            px = omega * 0.5 * (x / np.tan(omega * t * 0.5) - y)
            py = omega * 0.5 * (y / np.tan(omega * t * 0.5) + x) * self._mirrorYElectron
        else:
            px = self.m * x / t
            py = self.m * y / t
        pz = self._calcZMomentum(tof=t, spectrometer=spectrometer, calcSettings=calcSettings)
        
        # Shift momentum
        px += calcSettings.shiftPX
        py += calcSettings.shiftPY
        pz += calcSettings.shiftPZ

        # Stretch momentum
        px *= calcSettings.stretchPX * calcSettings.stretchPTotal
        py *= calcSettings.stretchPY * calcSettings.stretchPTotal
        pz *= calcSettings.stretchPZ * calcSettings.stretchPTotal
        
        # Calculate derived values
        p2     = px**2 + py**2 + pz**2
        p      = np.sqrt(p2)
        energy = CONSTANTS.EV_SI_TO_AU * p2 / (2*self.m)
        
        # Transfer values
        self._px     = px
        self._py     = py
        self._pz     = pz
        self._p      = p
        self._energy = energy

        self._recalculateMomentum = False
    
    def _calcZMomentum(self, tof: np.ndarray,                   \
                             spectrometer: Spectrometer = None, \
                             calcSettings: CalcSettings = None, \
                       ) -> None:
        import numpy as np
        if len(spectrometer) == 0:
            raise Exception("Spectrometer not sufficiently defined")
        if len(spectrometer) == 1:
            # Linear case
            length, field = spectrometer[0]
            if length is None or length == 0:
                # Linear Approximation
                return field * (tof - self.tofMean) / 124.38
            return length * self.m / tof - 0.5 * (field * self._electricFieldPolarity) * self.q * tof
        if len(spectrometer) == 2:
            if spectrometer[1][1] == 0.:
                s_B, a = spectrometer[0] # Length and field of acceleration region
                a     *= self._electricFieldPolarity
                s_D, _ = spectrometer[1] # Length drift region
                t      = np.array(tof, dtype=self._ctype)
                m      = self.m
                s      = s_B + s_D       # Total length of spectrometer
                q      = self.q
                
                # calculate velocity
                v = (-2*a**3*q**3*t**6 - 144*a**2*q**2*t**4*s_B + 12*a**2*q**2*s*t**4 + np.sqrt((-2*a**3*q**3*t**6 - 144*a**2*q**2*t**4*s_B + 12*a**2*q**2*s*t**4 + 108*a*q*t**2*s_B**2 + 72*a*q*s*t**2*s_B + 84*a*q*s**2*t**2 + 16*s**3)**2 + 4*(24*a*q*t**2*s_B - (a*q*t**2 - 2*s)**2)**3) + 108*a*q*t**2*s_B**2 + 72*a*q*s*t**2*s_B + 84*a*q*s**2*t**2 + 16*s**3)**(1./3.)/(6*2**(1./3.)*t) - (24*a*q*t**2*s_B - (a*q*t**2 - 2*s)**2)/(3*2**(2./3.)*t*(-2*a**3*q**3*t**6 - 144*a**2*q**2*t**4*s_B + 12*a**2*q**2*s*t**4 + np.sqrt((-2*a**3*q**3*t**6 - 144*a**2*q**2*t**4*s_B + 12*a**2*q**2*s*t**4 + 108*a*q*t**2*s_B**2 + 72*a*q*s*t**2*s_B + 84*a*q*s**2*t**2 + 16*s**3)**2 + 4*(24*a*q*t**2*s_B - (a*q*t**2 - 2*s)**2)**3) + 108*a*q*t**2*s_B**2 + 72*a*q*s*t**2*s_B + 84*a*q*s**2*t**2 + 16*s**3)**(1./3.)) - (a*q*t**2 - 2*s)/(6*t)
                print(np.sum(np.imag(v)!=0), v)
                test = (-2*a**3*q**3*t**6 - 144*a**2*q**2*t**4*s_B + 12*a**2*q**2*s*t**4 + 108*a*q*t**2*s_B**2 + 72*a*q*s*t**2*s_B + 84*a*q*s**2*t**2 + 16*s**3)**2 + 4*(24*a*q*t**2*s_B - (a*q*t**2 - 2*s)**2)**3
                print(np.min(test), test)
                vReal = np.array(np.real(v), dtype=self._dtype)
                vReal[np.imag(v)!=0] = None
                return m*vReal
            else:
                raise NotImplementedError("z-Momentum calculation not implemented for 2-regions spectrometer, when the field of the second region is not zero.")
        if len(spectrometer) >= 3:
            raise NotImplementedError("z-Momentum calculation not implemented for spectrometers with three or more regions.")

class Electron(Particle):
    from typing import Optional
    import numpy as np
    def __init__(self, x:            Optional[np.ndarray]           = None, \
                       y:            Optional[np.ndarray]           = None, \
                       tof:          Optional[np.ndarray]           = None, \
                       px:           Optional[np.ndarray]           = None, \
                       py:           Optional[np.ndarray]           = None, \
                       pz:           Optional[np.ndarray]           = None, \
                       p:            Optional[np.ndarray]           = None, \
                       energy:       Optional[np.ndarray]           = None, \
                       spectrometer: Optional[Spectrometer]         = None, \
                       calcSettings: Optional[CalcSettings]         = None, \
                       dtype:  np.typing.DTypeLike                  = np.double, \
                       ctype:  np.typing.DTypeLike                  = np.cdouble, \
                ) -> None:
        super().__init__(x=x, y=y, tof=tof, m=1, q=-1, px=px, py=py, pz=py, \
                         p=p, energy=energy, spectrometer=spectrometer,     \
                         calcSettings=calcSettings, isIonSide=False,        \
                         dtype=dtype, ctype=ctype)
        
class Ion(Particle):
    from typing import Optional
    import numpy as np
    def __init__(self, x:            Optional[np.ndarray]           = None, \
                       y:            Optional[np.ndarray]           = None, \
                       tof:          Optional[np.ndarray]           = None, \
                       m:            Optional[np.ndarray|int|float] = None, \
                       q:            Optional[np.ndarray|int|float] = None, \
                       px:           Optional[np.ndarray]           = None, \
                       py:           Optional[np.ndarray]           = None, \
                       pz:           Optional[np.ndarray]           = None, \
                       p:            Optional[np.ndarray]           = None, \
                       energy:       Optional[np.ndarray]           = None, \
                       spectrometer: Optional[Spectrometer]         = None, \
                       calcSettings: Optional[CalcSettings]         = None, \
                       isIonSide:    Optional[bool]                 = True, \
                       dtype:  np.typing.DTypeLike                  = np.double, \
                       ctype:  np.typing.DTypeLike                  = np.cdouble, \
                ) -> None:
        super().__init__(x=x, y=y, tof=tof, m=m, q=q, px=px, py=py, pz=py, \
                         p=p, energy=energy, spectrometer=spectrometer,    \
                         calcSettings=calcSettings, isIonSide=isIonSide,   \
                         dtype=dtype, ctype=ctype)
               
class ParticleList:
    from typing import Optional
    def __init__(self, particles: Optional[list[Particle]] = None):
        self.particles = list() if particles is None else particles
    
    def __len__(self) -> int:
        return len(self.particles)
    
    def __iter__(self) -> ParticleList:
        self.iterIndex = 0
        return self
    
    def __next__(self) -> Particle:
        if self.iterIndex < len(self):
            val = self.particles[self.iterIndex]
            self.iterIndex += 1
            return val
        else:
            raise StopIteration
    
    def __getitem__(self, index) -> Particle:
        return self.particles[index]
    
    
    def __add__(self, other: ParticleList|Particle) -> ParticleList:
        """
        Returns a shallow copy of the particle list, with other appended.
        """
        copy = self.particle.copy()
        if   isinstance(other, ParticleList):
            copy.extend(other)
            return copy
        elif isinstance(other, Particle):
            copy.append(other)
            return copy
        else:
            raise NotImplementedError
    
    def __iadd__(self, other: ParticleList|Particle) -> ParticleList:
        """
        Appends other to the list of particles.
        """
        if   isinstance(other, ParticleList):
            self.particles.extend(other)
            return self
        elif isinstance(other, Particle):
            self.particles.append(other)
            return self
        else:
            raise NotImplementedError

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
        self._ionsArr += Ion(x=x, y=y, tof=tof, m=m, q=q, spectrometer=spectrometer, calcSettings=calcSettings, dtype=dtype, ctype=ctype)

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
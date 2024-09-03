from __future__ import annotations
import CONSTANTS
from Spectrometer import Spectrometer, Spectrometer_jit
from CalcSettings import CalcSettings
from numba.experimental import jitclass
from numba import boolean, int64, float64
from typing import Optional, List
from numpy import ndarray
import numpy as np
import warnings
from MomentumCalculation import calculateMomentum

class Particle:    
    def __init__(self, x:            Optional[np.ndarray]           = None, \
                       y:            Optional[np.ndarray]           = None, \
                       tof:          Optional[np.ndarray]           = None, \
                       m:            Optional[int|float] = None, \
                       q:            Optional[int|float] = None, \
                       tofMean:      Optional[float]                = None, \
                       px:           Optional[np.ndarray]           = None, \
                       py:           Optional[np.ndarray]           = None, \
                       pz:           Optional[np.ndarray]           = None, \
                       p:            Optional[np.ndarray]           = None, \
                       energy:       Optional[np.ndarray]           = None, \
                       spectrometer: Optional[Spectrometer]         = None, \
                       calcSettings: Optional[CalcSettings]         = None, \
                       isIonSide: bool                              = True, \
                       recalculateMomentum: bool                    = True, \
                       dtype:  np.typing.DTypeLike                  = np.double, \
                       ctype:  np.typing.DTypeLike                  = np.cdouble, \
                       name: str                                    = "Particle_(m={m}, q={q})"
                ) -> None:
        self._x      = None if x      is None else np.array(x, dtype=dtype)      # mm
        self._y      = None if y      is None else np.array(y, dtype=dtype)      # mm
        self._tof    = None if tof    is None else np.array(tof, dtype=dtype)    # ns
        self._q      = None if q      is None else float(q)                      # a.u.
        self._m      = None if m      is None else float(m)                      # a.u.
        self._px     = None if px     is None else np.array(px, dtype=dtype)     # a.u.
        self._py     = None if py     is None else np.array(py, dtype=dtype)     # a.u.
        self._pz     = None if pz     is None else np.array(pz, dtype=dtype)     # a.u.
        self._p      = None if p      is None else np.array(p, dtype=dtype)      # a.u.
        self._energy = None if energy is None else np.array(energy, dtype=dtype) # eV
        
        self._spectrometer          = spectrometer
        self._recalculateMomentum   = recalculateMomentum
        self._electricFieldPolarity = 1 if isIonSide else -1
        self._mirrorYElectron       = 1 if isIonSide else -1
        self._calcSettings          = calcSettings

        self._tofMean = tofMean
        
        self._dtype = dtype
        self._ctype = ctype

        self._name = name
    
    def setUpdateMomentum(self):
        self._recalculateMomentum = True

    @property
    def name(self) -> str:
        return self._name.format(m=self.m, q=self.q, energy=self.energy, p=self.p)
    @name.setter
    def name(self, name: str) -> None:
        self._name = name
    
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
    @property
    def px(self) -> np.ndarray:
        """
        x-momentum of particle in a.u.
        """
        if self._px is None or self._recalculateMomentum:
            self.calcMomentum()
        return self._px
    @property
    def py(self) -> np.ndarray:
        """
        y-momentum of particle in a.u.
        """
        if self._py is None or self._recalculateMomentum:
            self.calcMomentum()
        return self._py
    @property
    def pz(self) -> np.ndarray:
        """
        z-momentum of particle in a.u.
        """
        if self._pz is None or self._recalculateMomentum:
            self.calcMomentum()
        return self._pz
    @property
    def p(self) -> np.ndarray:
        """
        absolute momentum of particle in a.u.
        """

        from numpy import sqrt

        if self._recalculateMomentum:
            self.calcMomentum()
        if self._p is None:
            self._p = sqrt(self.px**2 + self.py**2 + self.pz**2)
        return self._p
    @property
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
            omega = 2*np.pi*CONSTANTS.NS_SI_TO_AU / spectrometer.gyrationPeriod / self.m
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
            px = self.m*omega * 0.5 * (x / np.tan(omega * t * 0.5) - y)
            py = self.m*omega * 0.5 * (y / np.tan(omega * t * 0.5) + x) * self._mirrorYElectron
        else:
            px = self.m * x / t
            py = self.m * y / t
        pz = self._calcZMomentum(tof=t, spectrometer=spectrometer, calcSettings=calcSettings) * self._mirrorYElectron

        # Removing nan
        indexes = np.isfinite(pz)
        px = px[indexes]
        py = py[indexes]
        pz = pz[indexes]
        indexes = None
        
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
            #print("Linear Case!")
            length, field = spectrometer[0]
            if length is None:
                #print("Linear Approximation!")
                # Linear Approximation
                #print(field*CONSTANTS.VCM_SI_TO_AU, tof* CONSTANTS.NS_SI_TO_AU, self.tofMean)
                return field*CONSTANTS.VCM_SI_TO_AU * (tof* CONSTANTS.NS_SI_TO_AU - self.tofMean)  / 124.38
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
                #print(np.sum(np.imag(v)!=0.), v, np.imag(v)!=0)
                vReal = np.array(np.real(v), dtype=self._dtype)
                vReal[np.imag(v)!=0] = None
                #print(len(v), np.sum(np.imag(v)!=0))
                return m*vReal
            else:
                raise NotImplementedError("z-Momentum calculation not implemented for 2-regions spectrometer, when the field of the second region is not zero.")
        if len(spectrometer) >= 3:
            raise NotImplementedError("z-Momentum calculation not implemented for spectrometers with three or more regions.")

    def __mul__(self, other: Particle|List[float, float, float]|np.ndarray) -> float:
        from numpy import ndarray
        if isinstance(other,Particle):
            return self.px*other.px + self.py*other.py + self.pz*other.pz
        elif isinstance(other, list) or isinstance(other, ndarray):
            return self.px*other[0] + self.py*other[1] + self.pz*other[2]
        else:
            raise NotImplementedError
    def __rmul__(self, other):
        return self.__mul__(other)
        
    
    def __add__(self, other: Particle|List[float, float, float]|np.ndarray) -> Particle:
        from numpy import ndarray
        if isinstance(other, Particle):
            return Particle(px=self.px+other.px, py=self.py+other.py, pz=self.pz+other.pz, recalculateMomentum=False)
        elif isinstance(other, list) or isinstance(other, ndarray):
            return Particle(px=self.px+other[0], py=self.py+other[1], pz=self.pz+other[2], recalculateMomentum=False)
        else:
            raise NotImplementedError
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other: Particle|List[float, float, float]|np.ndarray) -> Particle:
        from numpy import ndarray
        if isinstance(other, Particle):
            return Particle(px=self.px-other.px, py=self.py-other.py, pz=self.pz-other.pz, recalculateMomentum=False)
        elif isinstance(other, list) or isinstance(other, ndarray):
            return Particle(px=self.px-other[0], py=self.py-other[1], pz=self.pz-other[2], recalculateMomentum=False)
        else:
            raise NotImplementedError
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __array__(self):
        import numpy as np
        warnings.warn("This methode will be removed in the switch to jit class", DeprecationWarning, 2)
        return np.array([self.px, self.py, self.pz])
    
    def __len__(self):
        return len(self.px)



class Electron(Particle):
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
    def __init__(self, x:            Optional[np.ndarray]           = None, \
                       y:            Optional[np.ndarray]           = None, \
                       tof:          Optional[np.ndarray]           = None, \
                       m:            Optional[np.ndarray|int|float] = None, \
                       q:            Optional[np.ndarray|int|float] = None, \
                       tofMean:      Optional[float]                = None, \
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
        m *= CONSTANTS.U_SI_TO_KG_SI
        m /= CONSTANTS.ME_SI_TO_AU
        super().__init__(x=x, y=y, tof=tof, m=m, q=q, px=px, py=py, pz=py, \
                         p=p, energy=energy, spectrometer=spectrometer,    \
                         calcSettings=calcSettings, isIonSide=isIonSide,   \
                         dtype=dtype, ctype=ctype, tofMean=tofMean)
           
                         

class ParticleList:
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


@jitclass([
    ("_recalculateMomentum", boolean),
    ("_x", float64[:]),
    ("_y", float64[:]),
    ("_tof", float64),
    ("_m", float64),
    ("_q", float64),
    ("_tofMean", float64),
    ("_px", float64[:]),
    ("_py", float64[:]),
    ("_pz", float64[:]),
    ("_p", float64[:]),
    ("_energy", float64[:]),
    ("_spectrometer", Spectrometer_jit.class_type.instance_type),
    ("_calcSettings", CalcSettings.class_type.instance_type),
    ("_electricFieldPolarity", int64),
    ("_mirrorYElectron", int64),
    ("_recalculateMomentum", boolean)
])
class Particle_jit:
    
    def __init__(self, x:            Optional[np.ndarray]           = None, \
                       y:            Optional[np.ndarray]           = None, \
                       tof:          Optional[np.ndarray]           = None, \
                       m:            Optional[float]                = None, \
                       q:            Optional[float]                = None, \
                       tofMean:      Optional[float]                = None, \
                       px:           Optional[np.ndarray]           = None, \
                       py:           Optional[np.ndarray]           = None, \
                       pz:           Optional[np.ndarray]           = None, \
                       p:            Optional[np.ndarray]           = None, \
                       energy:       Optional[np.ndarray]           = None, \
                       spectrometer: Optional[Spectrometer]         = None, \
                       calcSettings: Optional[CalcSettings]         = None, \
                       isIonSide: bool                              = True, \
                       recalculateMomentum: bool                    = True, \
                ) -> None:
        if x is not None:
            self._x = np.array(x, dtype=np.float64) # mm
        if y is not None:
            self._y = np.array(y, dtype=np.float64) # mm   
        if tof is not None:
            self._tof = np.array(tof, dtype=np.float64)    # ns
        if q is not None:
            self._q = float(q)      # a.u.
        if m is not None:
            self._m = float(m)      # a.u.
        if px is not None:
            self._px     = np.array(px, dtype=np.float64)     # a.u.
        if py is not None:
            self._py     = np.array(py, dtype=np.float64)     # a.u.
        if pz is not None:
            self._pz     = np.array(pz, dtype=np.float64)     # a.u.
        if p is not None:
            self._p      = np.array(p, dtype=np.float64)      # a.u.
        if energy is not None:
            self._energy = np.array(energy, dtype=np.float64) # eV
        
        if spectrometer is not None:
            self._spectrometer          = spectrometer
        self._recalculateMomentum   = recalculateMomentum
        self._electricFieldPolarity = 1 if isIonSide else -1
        self._mirrorYElectron       = 1 if isIonSide else -1
        if calcSettings is not None:
            self._calcSettings          = calcSettings
        if tofMean is not None:
            self._tofMean = tofMean

        
        self._dtype = np.double
        self._ctype = np.cdouble
    
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
    @property
    def px(self) -> np.ndarray:
        """
        x-momentum of particle in a.u.
        """
        if self._px is None or self._recalculateMomentum:
            self.calcMomentum()
        return self._px
    @property
    def py(self) -> np.ndarray:
        """
        y-momentum of particle in a.u.
        """
        if self._py is None or self._recalculateMomentum:
            self.calcMomentum()
        return self._py
    @property
    def pz(self) -> np.ndarray:
        """
        z-momentum of particle in a.u.
        """
        if self._pz is None or self._recalculateMomentum:
            self.calcMomentum()
        return self._pz
    @property
    def p(self) -> np.ndarray:
        """
        absolute momentum of particle in a.u.
        """

        from numpy import sqrt

        if self._recalculateMomentum:
            self.calcMomentum()
        if self._p is None:
            self._p = sqrt(self.px**2 + self.py**2 + self.pz**2)
        return self._p
    @property
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
        if spectrometer is None:
            spectrometer = self.spectrometer
        if calcSettings is None:
            calcSettings = self.calcSettings

        if spectrometer.gyrationPeriod is not None and spectrometer.gyrationPeriod != 0:
            omega = 2*np.pi*CONSTANTS.NS_SI_TO_AU / spectrometer.gyrationPeriod / self.m
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
            px = self.m*omega * 0.5 * (x / np.tan(omega * t * 0.5) - y)
            py = self.m*omega * 0.5 * (y / np.tan(omega * t * 0.5) + x) * self._mirrorYElectron
        else:
            px = self.m * x / t
            py = self.m * y / t
        pz = self._calcZMomentum(tof=t, spectrometer=spectrometer, calcSettings=calcSettings) * self._mirrorYElectron

        # Removing nan
        indexes = np.isfinite(pz)
        px = px[indexes]
        py = py[indexes]
        pz = pz[indexes]
        indexes = None
        
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
        if len(spectrometer) == 0:
            raise Exception("Spectrometer not sufficiently defined")
        if len(spectrometer) == 1:
            # Linear case
            #print("Linear Case!")
            length, field = spectrometer[0]
            if length is None:
                #print("Linear Approximation!")
                # Linear Approximation
                #print(field*CONSTANTS.VCM_SI_TO_AU, tof* CONSTANTS.NS_SI_TO_AU, self.tofMean)
                return field*CONSTANTS.VCM_SI_TO_AU * (tof* CONSTANTS.NS_SI_TO_AU - self.tofMean)  / 124.38
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
                #print(np.sum(np.imag(v)!=0.), v, np.imag(v)!=0)
                vReal = np.array(np.real(v), dtype=self._dtype)
                vReal[np.imag(v)!=0] = None
                #print(len(v), np.sum(np.imag(v)!=0))
                return m*vReal
            else:
                raise NotImplementedError("z-Momentum calculation not implemented for 2-regions spectrometer, when the field of the second region is not zero.")
        if len(spectrometer) >= 3:
            raise NotImplementedError("z-Momentum calculation not implemented for spectrometers with three or more regions.")

    def __mul__(self, other: Particle|List[float, float, float]|np.ndarray) -> float:
        if isinstance(other,Particle):
            return self.px*other.px + self.py*other.py + self.pz*other.pz
        elif isinstance(other, list) or isinstance(other, ndarray):
            return self.px*other[0] + self.py*other[1] + self.pz*other[2]
        else:
            raise NotImplementedError
    def __rmul__(self, other):
        return self.__mul__(other)
        
    
    def __add__(self, other: Particle|List[float, float, float]|np.ndarray) -> Particle:
        if isinstance(other, Particle):
            return Particle(px=self.px+other.px, py=self.py+other.py, pz=self.pz+other.pz, recalculateMomentum=False)
        elif isinstance(other, list) or isinstance(other, ndarray):
            return Particle(px=self.px+other[0], py=self.py+other[1], pz=self.pz+other[2], recalculateMomentum=False)
        else:
            raise NotImplementedError
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other: Particle|List[float, float, float]|np.ndarray) -> Particle:
        if isinstance(other, Particle):
            return Particle(px=self.px-other.px, py=self.py-other.py, pz=self.pz-other.pz, recalculateMomentum=False)
        elif isinstance(other, list) or isinstance(other, ndarray):
            return Particle(px=self.px-other[0], py=self.py-other[1], pz=self.pz-other[2], recalculateMomentum=False)
        else:
            raise NotImplementedError
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __len__(self):
        return len(self.px)
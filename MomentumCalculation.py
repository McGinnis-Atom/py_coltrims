import numpy as np
from Spectrometer import Spectrometer, Spectrometer_jit
from CalcSettings import CalcSettings
import numba
from numba import boolean, int64, float64
import CONSTANTS

def calculateMomentum(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    m: int|float,
    q: int|float,
    spectrometer: Spectrometer,
    calcSettings: CalcSettings,
    electricFieldPolarity: int|float,
    mirrorYElectron: int|float,
    tofMean: int|float
    ) -> (np.ndarray, np.ndarray, np.ndarray):

    if spectrometer.gyrationPeriod is not None and spectrometer.gyrationPeriod != 0:
        omega = 2*np.pi*CONSTANTS.NS_SI_TO_AU / spectrometer.gyrationPeriod / m
    rotationAngle = np.deg2rad(calcSettings.rotateDeg)
        
    x = x / CONSTANTS.MM_SI_TO_AU
    y = y / CONSTANTS.MM_SI_TO_AU
    t = t / CONSTANTS.NS_SI_TO_AU
        
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
        px = m*omega * 0.5 * (x / np.tan(omega * t * 0.5) - y)
        py = m*omega * 0.5 * (y / np.tan(omega * t * 0.5) + x) * mirrorYElectron
    else:
        px = m * x / t
        py = m * y / t
    pz = calcZMomentum(
        tof=t, 
        m=m, 
        q=q, 
        spectrometer=spectrometer, 
        calcSettings=calcSettings,
        tofMean = tofMean,
        electricFieldPolarity = electricFieldPolarity
    ) * mirrorYElectron

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

    return px, py, pz

def calcZMomentum(
    tof: np.ndarray,
    m: int|float,
    q: int|float,
    spectrometer: Spectrometer, 
    calcSettings: CalcSettings,
    tofMean: float|None,
    electricFieldPolarity: int|float,
    ) -> np.ndarray:

    if len(spectrometer) == 0:
        raise Exception("Spectrometer not sufficiently defined")

    elif len(spectrometer) == 1:
        # Linear case
        #print("Linear Case!")
        length, field = spectrometer[0]
        if length is None:
            #print("Linear Approximation!")
            # Linear Approximation
            #print(field*CONSTANTS.VCM_SI_TO_AU, tof* CONSTANTS.NS_SI_TO_AU, self.tofMean)
            return field*CONSTANTS.VCM_SI_TO_AU * (tof* CONSTANTS.NS_SI_TO_AU - tofMean)  / 124.38
        return length * m / tof - 0.5 * (field * electricFieldPolarity) * q * tof
        
    elif len(spectrometer) == 2:
        if spectrometer[1][1] == 0.:
            s_B, a = spectrometer[0] # Length and field of acceleration region
            a     *= electricFieldPolarity
            s_D, _ = spectrometer[1] # Length drift region
            t      = np.array(tof, dtype=np.cdouble)
            m      = m
            s      = s_B + s_D       # Total length of spectrometer
            q      = q
                
            # calculate velocity
            v = (-2*a**3*q**3*t**6 - 144*a**2*q**2*t**4*s_B + 12*a**2*q**2*s*t**4 + np.sqrt((-2*a**3*q**3*t**6 - 144*a**2*q**2*t**4*s_B + 12*a**2*q**2*s*t**4 + 108*a*q*t**2*s_B**2 + 72*a*q*s*t**2*s_B + 84*a*q*s**2*t**2 + 16*s**3)**2 + 4*(24*a*q*t**2*s_B - (a*q*t**2 - 2*s)**2)**3) + 108*a*q*t**2*s_B**2 + 72*a*q*s*t**2*s_B + 84*a*q*s**2*t**2 + 16*s**3)**(1./3.)/(6*2**(1./3.)*t) - (24*a*q*t**2*s_B - (a*q*t**2 - 2*s)**2)/(3*2**(2./3.)*t*(-2*a**3*q**3*t**6 - 144*a**2*q**2*t**4*s_B + 12*a**2*q**2*s*t**4 + np.sqrt((-2*a**3*q**3*t**6 - 144*a**2*q**2*t**4*s_B + 12*a**2*q**2*s*t**4 + 108*a*q*t**2*s_B**2 + 72*a*q*s*t**2*s_B + 84*a*q*s**2*t**2 + 16*s**3)**2 + 4*(24*a*q*t**2*s_B - (a*q*t**2 - 2*s)**2)**3) + 108*a*q*t**2*s_B**2 + 72*a*q*s*t**2*s_B + 84*a*q*s**2*t**2 + 16*s**3)**(1./3.)) - (a*q*t**2 - 2*s)/(6*t)
            #print(np.sum(np.imag(v)!=0.), v, np.imag(v)!=0)
            vReal = np.array(np.real(v), dtype=np.double)
            vReal[np.imag(v)!=0] = None
            #print(len(v), np.sum(np.imag(v)!=0))
            return m*vReal
        else:
            raise NotImplementedError("z-Momentum calculation not implemented for 2-regions spectrometer, when the field of the second region is not zero.")
    if len(spectrometer) >= 3:
        raise NotImplementedError("z-Momentum calculation not implemented for spectrometers with three or more regions.")
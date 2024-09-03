import numpy as np
from Spectrometer import Spectrometer, Spectrometer_jit
from CalcSettings import CalcSettings
import numba
from numba import boolean, int64, float64
import CONSTANTS

def calculateMomentum(
	x: np.ndarray,
	y: np.ndarray,
	t: np.ndarray
	):
	pass
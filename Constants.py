from numba import njit, float64


MM_SI_TO_AU       =  5.291_772_109_03e-8
VCM_SI_TO_AU      =  5.142_206_747_63e9
NS_SI_TO_AU       =  2.418_884_326_585_7e-8
EV_SI_TO_AU       = 27.211_386_245_988
ME_SI_TO_AU       =  9.109_383_7e-31
U_SI_TO_KG_SI     =  1.660_539_066_60e-27


@njit()
def GAUSS_TO_NS(x):
    return 1e9*(2*3.14159265359*9.1093837e-31)/(1.60217733e-19*x*1e-4)
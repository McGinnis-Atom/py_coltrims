import coltrims


def almostEqual(a, b):
    import numpy as np
    return np.abs(a - b) < 1e-3

class TestSpectrometer:
    import numpy as np

    def test_spectrometer_creation(self):
        import coltrims
        spec = coltrims.Spectrometer()

        assert isinstance(spec, coltrims.Spectrometer)
        assert len(spec) == 0
        assert spec.gyrationPeriod is None
        assert spec.magneticField is None
        assert spec.returnAU is True

    def test_spectrometer_addRegion(self):
        spec = coltrims.Spectrometer()

        assert isinstance(spec, coltrims.Spectrometer)
        assert len(spec) == 0
        assert spec.gyrationPeriod is None
        assert spec.magneticField is None
        assert spec.returnAU is True

        valA, valB = 2, 4
        spec.addRegion(valA, valB)

        assert len(spec) == 1

        a, b = spec[0]
        assert a == valA / coltrims.CONSTANTS.MM_SI_TO_AU
        assert b == valB / coltrims.CONSTANTS.VCM_SI_TO_AU

        for a,b in spec:
            assert a == valA / coltrims.CONSTANTS.MM_SI_TO_AU
            assert b == valB / coltrims.CONSTANTS.VCM_SI_TO_AU
            break

    def test_spectrometer_addRegion_noAU(self):
        spec = coltrims.Spectrometer()
        spec.returnAU = False

        assert isinstance(spec, coltrims.Spectrometer)
        assert len(spec) == 0
        assert spec.gyrationPeriod is None
        assert spec.magneticField is None
        assert spec.returnAU == False

        valA, valB = 2, 4
        spec.addRegion(valA, valB)

        assert len(spec) == 1

        a, b = spec[0]
        assert a == valA
        assert b == valB

        for a,b in spec:
            assert a == valA
            assert b == valB
            break

    def test_spectrometer_gyration_period(self):
        spec = coltrims.Spectrometer()
        assert isinstance(spec, coltrims.Spectrometer)
        assert len(spec) == 0
        assert spec.gyrationPeriod is None
        assert spec.magneticField is None
        assert spec.returnAU is True

        val = 5
        spec.gyrationPeriod = val
        assert almostEqual(spec.gyrationPeriod,val)
        assert almostEqual(spec.magneticField,coltrims.CONSTANTS.GAUSS_TO_NS(val))

    def test_spectrometer_magnetic_field(self):
        spec = coltrims.Spectrometer()
        assert isinstance(spec, coltrims.Spectrometer)
        assert len(spec) == 0
        assert spec.gyrationPeriod is None
        assert spec.magneticField is None
        assert spec.returnAU is True

        val = 5
        spec.magneticField = val
        assert almostEqual(spec.magneticField,val)
        assert almostEqual(spec.gyrationPeriod,coltrims.CONSTANTS.GAUSS_TO_NS(val))


class TestSpectrometerJit:
    import numpy as np

    def test_spectrometer_creation(self):
        import coltrims
        spec = coltrims.Spectrometer_jit()

        assert isinstance(spec, coltrims.Spectrometer_jit)
        assert len(spec) == 0
        assert spec.gyrationPeriod is None
        assert spec.magneticField is None
        assert spec.returnAU is True

    def test_spectrometer_addRegion(self):
        spec = coltrims.Spectrometer_jit()

        assert isinstance(spec, coltrims.Spectrometer_jit)
        assert len(spec) == 0
        assert spec.gyrationPeriod is None
        assert spec.magneticField is None
        assert spec.returnAU is True

        valA, valB = 2, 4
        spec.addRegion(valA, valB)

        assert len(spec) == 1

        a, b = spec[0]
        assert a == valA / coltrims.CONSTANTS.MM_SI_TO_AU
        assert b == valB / coltrims.CONSTANTS.VCM_SI_TO_AU

        for a,b in spec:
            assert a == valA / coltrims.CONSTANTS.MM_SI_TO_AU
            assert b == valB / coltrims.CONSTANTS.VCM_SI_TO_AU

        valC, valD = 3, 5
        spec.addRegion(valC, valD)

        assert len(spec) == 2

        a, b = spec[1]
        assert a == valC / coltrims.CONSTANTS.MM_SI_TO_AU
        assert b == valD / coltrims.CONSTANTS.VCM_SI_TO_AU

        for (a,b),c,d in zip(spec,[valA, valC],[valB, valD]):
            assert a == c / coltrims.CONSTANTS.MM_SI_TO_AU
            assert b == d / coltrims.CONSTANTS.VCM_SI_TO_AU

    def test_spectrometer_addRegion_expandingArray(self):
        spec = coltrims.Spectrometer_jit(maxLength=0)

        assert isinstance(spec, coltrims.Spectrometer_jit)
        assert len(spec) == 0
        assert spec.maxLength == 0
        assert len(spec.lengths) == 0
        assert len(spec.electricFields) == 0
        assert spec.gyrationPeriod is None
        assert spec.magneticField is None
        assert spec.returnAU is True

        valA, valB = 2, 4
        spec.addRegion(valA, valB)

        assert len(spec) == 1
        assert spec.maxLength == 4
        assert len(spec.lengths) == 4
        assert len(spec.electricFields) == 4


        a, b = spec[0]
        assert a == valA / coltrims.CONSTANTS.MM_SI_TO_AU
        assert b == valB / coltrims.CONSTANTS.VCM_SI_TO_AU

        for a,b in spec:
            assert a == valA / coltrims.CONSTANTS.MM_SI_TO_AU
            assert b == valB / coltrims.CONSTANTS.VCM_SI_TO_AU

    def test_spectrometer_addRegion_noAU(self):
        spec = coltrims.Spectrometer_jit()
        spec.returnAU = False

        assert isinstance(spec, coltrims.Spectrometer_jit)
        assert len(spec) == 0
        assert spec.gyrationPeriod is None
        assert spec.magneticField is None
        assert spec.returnAU == False

        valA, valB = 2, 4
        spec.addRegion(valA, valB)

        assert len(spec) == 1

        a, b = spec[0]
        assert a == valA
        assert b == valB

        for a,b in spec:
            assert a == valA
            assert b == valB
            break

    def test_spectrometer_gyration_period(self):
        spec = coltrims.Spectrometer_jit()
        assert isinstance(spec, coltrims.Spectrometer_jit)
        assert len(spec) == 0
        assert spec.gyrationPeriod is None
        assert spec.magneticField is None
        assert spec.returnAU is True

        val = 5
        spec.gyrationPeriod = val
        assert almostEqual(spec.gyrationPeriod,val)
        assert almostEqual(spec.magneticField,coltrims.CONSTANTS.GAUSS_TO_NS(val))

    def test_spectrometer_magnetic_field(self):
        spec = coltrims.Spectrometer_jit()
        assert isinstance(spec, coltrims.Spectrometer_jit)
        assert len(spec) == 0
        assert spec.gyrationPeriod is None
        assert spec.magneticField is None
        assert spec.returnAU is True

        val = 5
        spec.magneticField = val
        assert almostEqual(spec.magneticField,val)
        assert almostEqual(spec.gyrationPeriod,coltrims.CONSTANTS.GAUSS_TO_NS(val))


"""
TODO:
    - Add test for paticle
    - Add test for Reaction
"""
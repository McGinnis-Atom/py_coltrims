import coltrims
import numpy as np
import pytest


def almostEqual(a, b):
    import numpy as np
    return np.abs(a - b) < 1e-3

class TestColtrims:
    def test_default(self):
        p = coltrims.Particle()
        assert p._x == None
        assert p._y == None
        assert p._tof == None
        assert p._m == None
        assert p._q == None
        assert p._px == None
        assert p._py == None
        assert p._pz == None
        assert p._p == None
        assert p._energy == None
        assert p._spectrometer == None
        assert p._recalculateMomentum == True
        assert p._electricFieldPolarity == 1
        assert p._mirrorYElectron == 1
        assert p._calcSettings == None
        assert p._tofMean == None
        assert p._dtype == np.double
        assert p._ctype == np.cdouble
        assert p._name == "Particle_(m={m}, q={q})"

    def test_nonDefault(self):
        spec = coltrims.Spectrometer()
        cSet = coltrims.CalcSettings()
        
        p = coltrims.Particle(
            x = np.array([0,1,2]),
            y = np.array([0,1,2]),
            tof = np.array([0,1,2]),
            m = 1,
            q = 1,
            tofMean = 1000.0,
            px = np.array([0,1,2]),
            py = np.array([0,1,2]),
            pz = np.array([0,1,2]),
            p = np.array([0,1,2]),
            energy = np.array([0,1,2]),
            spectrometer = spec,
            calcSettings = cSet,
            isIonSide = False,
            recalculateMomentum = False,
            name = "Hi"
        )
        assert all(p._x == [0,1,2])
        assert all(p._y == [0,1,2])
        assert all(p._tof == [0,1,2])
        assert p._m == 1
        assert p._q == 1
        assert all(p._px == [0,1,2])
        assert all(p._py == [0,1,2])
        assert all(p._pz == [0,1,2])
        assert all(p._p == [0,1,2])
        assert all(p._energy == [0,1,2])
        assert p._spectrometer == spec
        assert p._recalculateMomentum == False
        assert p._electricFieldPolarity == -1
        assert p._mirrorYElectron == -1
        assert p._calcSettings == cSet
        assert p._tofMean == 1000.
        assert p._dtype == np.double
        assert p._ctype == np.cdouble
        assert p._name == "Hi"
    
    def test_setUpdateMomentum(self):
        p = coltrims.Particle(recalculateMomentum = False)
        assert p._recalculateMomentum == False

        p.setUpdateMomentum()
        assert p._recalculateMomentum == True

    def test_setter_tofMean(self):
        p = coltrims.Particle(recalculateMomentum = False)
        assert p._tofMean == None
        with pytest.raises(ValueError):
            p.tofMean
        assert p._recalculateMomentum == False

        p.tofMean = 1000.0
        assert isinstance(p._tofMean, float)
        assert p._tofMean == 1000.0
        assert p.tofMean == 1000.0
        assert p._recalculateMomentum == True

        p.tofMean = 1000
        assert isinstance(p._tofMean, int)
        assert p._tofMean == 1000
        assert p.tofMean == 1000

    def test_setter_x(self):
        p = coltrims.Particle(recalculateMomentum = False)
        assert p._x == None
        with pytest.raises(ValueError):
            p.x
        assert p._recalculateMomentum == False

        p.x = [0,1,2]
        assert isinstance(p._x, np.ndarray)
        assert all(p._x == [0,1,2])
        assert all(p.x == [0,1,2])
        assert p._recalculateMomentum == True

    def test_setter_y(self):
        p = coltrims.Particle(recalculateMomentum = False)
        assert p._y == None
        with pytest.raises(ValueError):
            p.y
        assert p._recalculateMomentum == False

        p.y = [0,1,2]
        assert isinstance(p._y, np.ndarray)
        assert all(p._y == [0,1,2])
        assert all(p.y == [0,1,2])
        assert p._recalculateMomentum == True

    def test_setter_tof(self):
        p = coltrims.Particle(recalculateMomentum = False)
        assert p._tof == None
        with pytest.raises(ValueError):
            p.tof
        assert p._recalculateMomentum == False

        p.tof = [0,1,2]
        assert isinstance(p._tof, np.ndarray)
        assert all(p._tof == [0,1,2])
        assert all(p.tof == [0,1,2])
        assert p._recalculateMomentum == True

    def test_setter_m(self):
        p = coltrims.Particle(recalculateMomentum = False)
        assert p._m == None
        with pytest.raises(ValueError):
            p.m
        assert p._recalculateMomentum == False

        p.m = 1
        assert isinstance(p._m, np.ndarray)
        assert p._m == 1
        assert p.m == 1
        assert p._recalculateMomentum == True

    def test_setter_q(self):
        p = coltrims.Particle(recalculateMomentum = False)
        assert p._q == None
        with pytest.raises(ValueError):
            p.q
        assert p._recalculateMomentum == False

        p.q = 1
        assert isinstance(p._q, np.ndarray)
        assert p._q == 1
        assert p.q == 1
        assert p._recalculateMomentum == True

    def test_px(self):
        spec = coltrims.Spectrometer()
        spec.addRegion(None, 1)
        p = coltrims.Particle(
            x = [0,1,-2],
            y = [0,-1,2],
            tof = [1,2,3],
            m = 1,
            q = 1,
            tofMean = 2,
            spectrometer = spec,
            calcSettings = coltrims.CalcSettings()
        )
        assert p._px == None
        assert p._recalculateMomentum == True

        assert all(p.px == [0,0.22855144521984055,-0.3047352602931207])

        assert isinstance(p._px, np.ndarray)
        assert all(p._px == [0,0.22855144521984055,-0.3047352602931207])
        assert p._recalculateMomentum == False

    def test_py(self):
        spec = coltrims.Spectrometer()
        spec.addRegion(None, 1)
        p = coltrims.Particle(
            x = [0,1,-2],
            y = [0,-1,2],
            tof = [1,2,3],
            m = 1,
            q = 1,
            tofMean = 2,
            spectrometer = spec,
            calcSettings = coltrims.CalcSettings()
        )
        assert p._py == None
        assert p._recalculateMomentum == True

        assert all(p.py == [0,-0.22855144521984055,0.3047352602931207])

        assert isinstance(p._py, np.ndarray)
        assert all(p._py == [0,-0.22855144521984055,0.3047352602931207])
        assert p._recalculateMomentum == False

    def test_pz(self):
        spec = coltrims.Spectrometer()
        spec.addRegion(None, 1)
        p = coltrims.Particle(
            x = [0,1,-2],
            y = [0,-1,2],
            tof = [1,2,3],
            m = 1,
            q = 1,
            tofMean = 2,
            spectrometer = spec,
            calcSettings = coltrims.CalcSettings()
        )
        assert p._pz == None
        assert p._recalculateMomentum == True

        assert all(p.pz == [-0.008039877793857534,0,0.008039877793857534])

        assert isinstance(p._pz, np.ndarray)
        assert all(p._pz == [-0.008039877793857534,0,0.008039877793857534])
        assert p._recalculateMomentum == False

    def test_p(self):
        spec = coltrims.Spectrometer()
        spec.addRegion(None, 1)
        p = coltrims.Particle(
            x = [0,1,-2],
            y = [0,-1,2],
            tof = [1,2,3],
            m = 1,
            q = 1,
            tofMean = 2,
            spectrometer = spec,
            calcSettings = coltrims.CalcSettings()
        )
        assert p._p == None
        assert p._recalculateMomentum == True

        assert all(p.p == [0.008039877793857534,0.32322055352987,0.4310357263229722])

        assert isinstance(p._p, np.ndarray)
        assert all(p._p == [0.008039877793857534,0.32322055352987,0.4310357263229722])
        assert all(p.p == (p.px**2+p.py**2+p.pz**2)**0.5)
        assert p._recalculateMomentum == False

    def test_energy(self):
        spec = coltrims.Spectrometer()
        spec.addRegion(None, 1)
        p = coltrims.Particle(
            x = [0,1,-2],
            y = [0,-1,2],
            tof = [1,2,3],
            m = 1,
            q = 1,
            tofMean = 2,
            spectrometer = spec,
            calcSettings = coltrims.CalcSettings()
        )
        assert p._energy == None
        assert p._recalculateMomentum == True

        assert all(p.energy == [0.000879467036578225,1.4214075258966805,2.5278261797417874])

        assert isinstance(p._energy, np.ndarray)
        assert all(p._energy == [0.000879467036578225,1.4214075258966805,2.5278261797417874])
        assert p._recalculateMomentum == False

    def test_name(self):
        spec = coltrims.Spectrometer()
        spec.addRegion(None, 1)
        p = coltrims.Particle(
            x = [0,1,-2],
            y = [0,-1,2],
            tof = [1,2,3],
            m = 1,
            q = 1,
            tofMean = 2,
            spectrometer = spec,
            calcSettings = coltrims.CalcSettings()
        )

        assert p._name == "Particle_(m={m}, q={q})"
        assert p.name == "Particle_(m=1.0, q=1.0)"

        p.name = "Hi"
        assert p._name == "Hi"
        assert p.name == "Hi"

        p.name = "{m} {q} {energy} {p}"
        assert p._name == "{m} {q} {energy} {p}"
        assert p.name == "1.0 1.0 [8.79467037e-04 1.42140753e+00 2.52782618e+00] [0.00803988 0.32322055 0.43103573]"
class TestClass:
    import coltrims
    import numpy as np
    
    def spectrometer_creation(self):
        import coltrims
        spec = coltrims.Spectrometer()

        assert isinstance(spec, coltrims.Spectrometer)
        assert len(spec) == 0
        assert spec.gyrationPeriod is None
        assert spec.magneticField is None

from __future__ import annotations
from Constants import CONSTANTS

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
        self.returnAU = True
    
    def __len__(self):
        return len(self.lengths)
    
    def __iter__(self):
        self.iterIndex = 0
        return self
    
    def __next__(self):
        if self.iterIndex < len(self):
            if self.returnAU:
                val = (None if self.lengths[self.iterIndex] is None else self.lengths[self.iterIndex]/CONSTANTS.MM_SI_TO_AU, None if self.electricFields[self.iterIndex] is None else self.electricFields[self.iterIndex]/CONSTANTS.VCM_SI_TO_AU)
            else:
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
        
        if self.returnAU:
            return (None if self.lengths[index] is None else self.lengths[index]/CONSTANTS.MM_SI_TO_AU, None if self.electricFields[index] is None else self.electricFields[index]/CONSTANTS.VCM_SI_TO_AU)
        else:
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
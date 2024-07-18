import numba
from CalcSettings import CalcSettings


class TestCalcSettings:
	def __init__(self):
		self.setting = CalcSettings(
			shiftX    = 0., shiftY      = 0.,    shiftTOF        = 0.,    \
            stretchX  = 1., stretchY    = 1.,    stretchTotal    = 1.,    \
            rotateDeg = 0., mirrorX     = False, mirrorY         = False, \
            shiftPX   = 0., shiftPY     = 0.,    shiftPZ         = 0.,    \
            stretchPX = 1., stretchPY   = 1.,    stretchPZ       = 1., stretchPTotal = 1., \
            mirrorPX  = False, mirrorPY = False, shiftThenRotate =True   \
		)

	def test_shiftX(self):
		assert self.setting.shiftX == 0

	def test_shiftX_addOne(self):
		f = lambda x: x.shiftX + 1
		assert f(self.setting) == 1

	def test_shiftX_addOne_njit(self):
		f = lambda x: x.shiftX + 1
		f = numba.njit()(f)
		assert f(self.setting) == 1
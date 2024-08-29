import numba
from CalcSettings import CalcSettings


class TestCalcSettings:
	def inital_parameters(self):
		setting = CalcSettings()
		assert setting.shiftX == 0.
		assert setting.shiftY == 0.
		assert setting.shiftTOF==0.
		assert setting.stretchX==1.
		assert setting.stretchY==1.
		assert setting.stretchTotal==1.
		assert setting.rotateDeg==0.
		assert setting.mirrorX == False
		assert setting.mirrorY == False
		assert setting.shiftPX == 0.
		assert setting.shiftPY == 0.
		assert setting.shiftPZ == 0.
		assert setting.stretchPX==1.
		assert setting.stretchPY==1.
		assert setting.stretchPZ==1.
		assert setting.stretchPTotal==1.
		assert setting.mirrorPX == False
		assert setting.mirrorPY == False
		assert setting.shiftThenRotate == True

	def test_shiftX(self):
		setting = CalcSettings(shiftX = 1.)
		assert setting.shiftX == 1

	def test_shiftX_addOne(self):
		setting = CalcSettings(shiftX = 1.)
		f = lambda x: x.shiftX + 1
		assert f(setting) == 2.

	def test_shiftX_addOne_njit(self):
		setting = CalcSettings(shiftX = 1.)
		f = lambda x: x.shiftX + 1
		f = numba.njit()(f)
		assert f(setting) == 2.

	def test_shiftY(self):
		setting = CalcSettings(shiftY = 1.)
		assert setting.shiftY == 1

	def test_shiftY_addOne(self):
		setting = CalcSettings(shiftY = 1.)
		f = lambda x: x.shiftY + 1
		assert f(setting) == 2.

	def test_shiftY_addOne_njit(self):
		setting = CalcSettings(shiftY = 1.)
		f = lambda x: x.shiftY + 1
		f = numba.njit()(f)
		assert f(setting) == 2.

	def test_shiftTOF(self):
		setting = CalcSettings(shiftTOF = 1.)
		assert setting.shiftTOF == 1

	def test_shiftTOF_addOne(self):
		setting = CalcSettings(shiftTOF = 1.)
		f = lambda x: x.shiftTOF + 1
		assert f(setting) == 2.

	def test_shiftTOF_addOne_njit(self):
		setting = CalcSettings(shiftTOF = 1.)
		f = lambda x: x.shiftTOF + 1
		f = numba.njit()(f)
		assert f(setting) == 2.

	def test_shiftT(self):
		setting = CalcSettings(shiftTOF = 1.)
		assert setting.shiftT == 1

	def test_shiftT_addOne(self):
		setting = CalcSettings(shiftTOF = 1.)
		f = lambda x: x.shiftT + 1
		assert f(setting) == 2.

	def test_shiftT_addOne_njit(self):
		setting = CalcSettings(shiftTOF = 1.)
		f = lambda x: x.shiftT + 1
		f = numba.njit()(f)
		assert f(setting) == 2.

	def test_stretchX(self):
		setting = CalcSettings(stretchX = 1.)
		assert setting.stretchX == 1

	def test_stretchX_addOne(self):
		setting = CalcSettings(stretchX = 1.)
		f = lambda x: x.stretchX + 1
		assert f(setting) == 2.

	def test_stretchX_addOne_njit(self):
		setting = CalcSettings(stretchX = 1.)
		f = lambda x: x.stretchX + 1
		f = numba.njit()(f)
		assert f(setting) == 2.

	def test_stretchY(self):
		setting = CalcSettings(stretchY = 1.)
		assert setting.stretchY == 1

	def test_stretchY_addOne(self):
		setting = CalcSettings(stretchY = 1.)
		f = lambda x: x.stretchY + 1
		assert f(setting) == 2.

	def test_stretchY_addOne_njit(self):
		setting = CalcSettings(stretchY = 1.)
		f = lambda x: x.stretchY + 1
		f = numba.njit()(f)
		assert f(setting) == 2.

	def test_stretchTotal(self):
		setting = CalcSettings(stretchTotal = 1.)
		assert setting.stretchTotal == 1

	def test_stretchTotal_addOne(self):
		setting = CalcSettings(stretchTotal = 1.)
		f = lambda x: x.stretchTotal + 1
		assert f(setting) == 2.

	def test_stretchTotal_addOne_njit(self):
		setting = CalcSettings(stretchTotal = 1.)
		f = lambda x: x.stretchTotal + 1
		f = numba.njit()(f)
		assert f(setting) == 2.

	def test_rotateDeg(self):
		setting = CalcSettings(rotateDeg = 1.)
		assert setting.rotateDeg == 1

	def test_rotateDeg_addOne(self):
		setting = CalcSettings(rotateDeg = 1.)
		f = lambda x: x.rotateDeg + 1
		assert f(setting) == 2.

	def test_rotateDeg_addOne_njit(self):
		setting = CalcSettings(rotateDeg = 1.)
		f = lambda x: x.rotateDeg + 1
		f = numba.njit()(f)
		assert f(setting) == 2.

	def test_shiftPX(self):
		setting = CalcSettings(shiftPX = 1.)
		assert setting.shiftPX == 1

	def test_shiftPX_addOne(self):
		setting = CalcSettings(shiftPX = 1.)
		f = lambda x: x.shiftPX + 1
		assert f(setting) == 2.

	def test_shiftPX_addOne_njit(self):
		setting = CalcSettings(shiftPX = 1.)
		f = lambda x: x.shiftPX + 1
		f = numba.njit()(f)
		assert f(setting) == 2.

	def test_shiftPY(self):
		setting = CalcSettings(shiftPY = 1.)
		assert setting.shiftPY == 1

	def test_shiftPY_addOne(self):
		setting = CalcSettings(shiftPY = 1.)
		f = lambda x: x.shiftPY + 1
		assert f(setting) == 2.

	def test_shiftPY_addOne_njit(self):
		setting = CalcSettings(shiftPY = 1.)
		f = lambda x: x.shiftPY + 1
		f = numba.njit()(f)
		assert f(setting) == 2.

	def test_shiftPZ(self):
		setting = CalcSettings(shiftPZ = 1.)
		assert setting.shiftPZ == 1

	def test_shiftPZ_addOne(self):
		setting = CalcSettings(shiftPZ = 1.)
		f = lambda x: x.shiftPZ + 1
		assert f(setting) == 2.

	def test_shiftPZ_addOne_njit(self):
		setting = CalcSettings(shiftPZ = 1.)
		f = lambda x: x.shiftPZ + 1
		f = numba.njit()(f)
		assert f(setting) == 2.

	def test_stretchPTotal(self):
		setting = CalcSettings(stretchPTotal = 1.)
		assert setting.stretchPTotal == 1

	def test_stretchPTotal_addOne(self):
		setting = CalcSettings(stretchPTotal = 1.)
		f = lambda x: x.stretchPTotal + 1
		assert f(setting) == 2.

	def test_stretchPTotal_addOne_njit(self):
		setting = CalcSettings(stretchPTotal = 1.)
		f = lambda x: x.stretchPTotal + 1
		f = numba.njit()(f)
		assert f(setting) == 2.

	def test_stretchPX(self):
		setting = CalcSettings(stretchPX = 1.)
		assert setting.stretchPX == 1

	def test_stretchPX_addOne(self):
		setting = CalcSettings(stretchPX = 1.)
		f = lambda x: x.stretchPX + 1
		assert f(setting) == 2.

	def test_stretchPX_addOne_njit(self):
		setting = CalcSettings(stretchPX = 1.)
		f = lambda x: x.stretchPX + 1
		f = numba.njit()(f)
		assert f(setting) == 2.

	def test_stretchPY(self):
		setting = CalcSettings(stretchPY = 1.)
		assert setting.stretchPY == 1

	def test_stretchPY_addOne(self):
		setting = CalcSettings(stretchPY = 1.)
		f = lambda x: x.stretchPY + 1
		assert f(setting) == 2.

	def test_stretchPY_addOne_njit(self):
		setting = CalcSettings(stretchPY = 1.)
		f = lambda x: x.stretchPY + 1
		f = numba.njit()(f)
		assert f(setting) == 2.

	def test_stretchPZ(self):
		setting = CalcSettings(stretchPZ = 1.)
		assert setting.stretchPZ == 1

	def test_stretchPZ_addOne(self):
		setting = CalcSettings(stretchPZ = 1.)
		f = lambda x: x.stretchPZ + 1
		assert f(setting) == 2.

	def test_stretchPZ_addOne_njit(self):
		setting = CalcSettings(stretchPZ = 1.)
		f = lambda x: x.stretchPZ + 1
		f = numba.njit()(f)
		assert f(setting) == 2.

	def test_mirrorX(self):
		setting = CalcSettings(mirrorX = True)
		assert setting.mirrorX is True

	def test_mirrorX_inv(self):
		setting = CalcSettings(mirrorX = False)
		assert setting.mirrorX is False

	def test_mirrorX_notTrue(self):
		setting = CalcSettings(mirrorX = True)
		f = lambda x: not x.mirrorX
		assert f(setting) is False

	def test_mirrorX_notTrue_njit(self):
		setting = CalcSettings(mirrorX = True)
		f = lambda x: not x.mirrorX
		f = numba.njit()(f)
		assert f(setting) is False

	def test_mirrorX_notFalse(self):
		setting = CalcSettings(mirrorX = False)
		f = lambda x: not x.mirrorX
		assert f(setting) is True

	def test_mirrorX_notFalse_njit(self):
		setting = CalcSettings(mirrorX = False)
		f = lambda x: not x.mirrorX
		f = numba.njit()(f)
		assert f(setting) is True

	def test_mirrorY(self):
		setting = CalcSettings(mirrorY = True)
		assert setting.mirrorY is True

	def test_mirrorY_inv(self):
		setting = CalcSettings(mirrorY = False)
		assert setting.mirrorY is False

	def test_mirrorY_notTrue(self):
		setting = CalcSettings(mirrorY = True)
		f = lambda x: not x.mirrorY
		assert f(setting) is False

	def test_mirrorY_notTrue_njit(self):
		setting = CalcSettings(mirrorY = True)
		f = lambda x: not x.mirrorY
		f = numba.njit()(f)
		assert f(setting) is False

	def test_mirrorY_notFalse(self):
		setting = CalcSettings(mirrorY = False)
		f = lambda x: not x.mirrorY
		assert f(setting) is True

	def test_mirrorY_notFalse_njit(self):
		setting = CalcSettings(mirrorY = False)
		f = lambda x: not x.mirrorY
		f = numba.njit()(f)
		assert f(setting) is True

	def test_mirrorPX(self):
		setting = CalcSettings(mirrorPX = True)
		assert setting.mirrorPX is True

	def test_mirrorPX_inv(self):
		setting = CalcSettings(mirrorPX = False)
		assert setting.mirrorPX is False

	def test_mirrorPX_notTrue(self):
		setting = CalcSettings(mirrorPX = True)
		f = lambda x: not x.mirrorPX
		assert f(setting) is False

	def test_mirrorPX_notTrue_njit(self):
		setting = CalcSettings(mirrorPX = True)
		f = lambda x: not x.mirrorPX
		f = numba.njit()(f)
		assert f(setting) is False

	def test_mirrorPX_notFalse(self):
		setting = CalcSettings(mirrorPX = False)
		f = lambda x: not x.mirrorPX
		assert f(setting) is True

	def test_mirrorPX_notFalse_njit(self):
		setting = CalcSettings(mirrorPX = False)
		f = lambda x: not x.mirrorPX
		f = numba.njit()(f)
		assert f(setting) is True

	def test_mirrorPY(self):
		setting = CalcSettings(mirrorPY = True)
		assert setting.mirrorPY is True

	def test_mirrorPY_inv(self):
		setting = CalcSettings(mirrorPY = False)
		assert setting.mirrorPY is False

	def test_mirrorPY_notTrue(self):
		setting = CalcSettings(mirrorPY = True)
		f = lambda x: not x.mirrorPY
		assert f(setting) is False

	def test_mirrorPY_notTrue_njit(self):
		setting = CalcSettings(mirrorPY = True)
		f = lambda x: not x.mirrorPY
		f = numba.njit()(f)
		assert f(setting) is False

	def test_mirrorPY_notFalse(self):
		setting = CalcSettings(mirrorPY = False)
		f = lambda x: not x.mirrorPY
		assert f(setting) is True

	def test_mirrorPY_notFalse_njit(self):
		setting = CalcSettings(mirrorPY = False)
		f = lambda x: not x.mirrorPY
		f = numba.njit()(f)
		assert f(setting) is True

	def test_shiftThenRotate(self):
		setting = CalcSettings(shiftThenRotate = True)
		assert setting.shiftThenRotate is True

	def test_shiftThenRotate_inv(self):
		setting = CalcSettings(shiftThenRotate = False)
		assert setting.shiftThenRotate is False

	def test_shiftThenRotate_notTrue(self):
		setting = CalcSettings(shiftThenRotate = True)
		f = lambda x: not x.shiftThenRotate
		assert f(setting) is False

	def test_shiftThenRotate_notTrue_njit(self):
		setting = CalcSettings(shiftThenRotate = True)
		f = lambda x: not x.shiftThenRotate
		f = numba.njit()(f)
		assert f(setting) is False

	def test_shiftThenRotate_notFalse(self):
		setting = CalcSettings(shiftThenRotate = False)
		f = lambda x: not x.shiftThenRotate
		assert f(setting) is True

	def test_shiftThenRotate_notFalse_njit(self):
		setting = CalcSettings(shiftThenRotate = False)
		f = lambda x: not x.shiftThenRotate
		f = numba.njit()(f)
		assert f(setting) is True
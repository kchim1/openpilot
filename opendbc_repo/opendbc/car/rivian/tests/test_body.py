import pytest

from opendbc.car.rivian.interface import CarInterface
from opendbc.car.rivian.values import CAR, R1S_WHEELBASE

FINGERPRINT = {0: {0x321: 8}, 1: {0x131a: 8, 0x1310: 8}, 2: {}}


def params(vin):
  CP = CarInterface.get_params(CAR.RIVIAN_R1, FINGERPRINT, [], False, False, docs=False)
  CP.carVin = vin
  CarInterface.get_params_sp(CP, CAR.RIVIAN_R1, FINGERPRINT, [], False, False, docs=False)
  return CP


def test_r1s_wheelbase_from_the_vin():
  r1t, r1s = params("7FCTGAAL8NN000000"), params("7PDSGABL2NN000000")
  assert r1s.wheelbase == pytest.approx(R1S_WHEELBASE)
  assert r1t.wheelbase == pytest.approx(3.45)
  # the same weight distribution, inertia scaled with the wheelbase
  assert r1s.centerToFront / r1s.wheelbase == pytest.approx(r1t.centerToFront / r1t.wheelbase)
  assert r1s.rotationalInertia < r1t.rotationalInertia


@pytest.mark.parametrize("vin", ["0" * 17, "", "7FCTGAAL8NN000000"])
def test_unknown_or_r1t_keeps_the_r1t_wheelbase(vin):
  assert params(vin).wheelbase == pytest.approx(3.45)

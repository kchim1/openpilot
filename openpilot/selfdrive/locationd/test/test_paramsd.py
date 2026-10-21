import random
import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal import messaging
from opendbc.car.structs import car
from openpilot.selfdrive.locationd.paramsd import retrieve_initial_vehicle_params
from openpilot.selfdrive.locationd.models.car_kf import CarKalman
from openpilot.selfdrive.locationd.test.test_locationd_scenarios import TEST_ROUTE
from openpilot.selfdrive.test.process_replay.migration import migrate, migrate_carParams
from openpilot.common.params import Params
from openpilot.tools.lib.logreader import LogReader


def get_random_vehicle_parameters(CP):
  msg = messaging.new_message("vehicleParameters")
  msg.vehicleParameters.steerRatio = (random.random() + 0.5) * CP.steerRatio
  msg.vehicleParameters.stiffnessFactor = random.random()
  msg.vehicleParameters.angleOffsetAverageDeg = random.random()
  msg.vehicleParameters.debugFilterState.std = [random.random() for _ in range(CarKalman.P_initial.shape[0])]
  return msg


class TestParamsd(OpenpilotTestCase):
  def test_read_saved_params(self):
    params = Params()

    lr = migrate(LogReader(TEST_ROUTE), [migrate_carParams])
    CP = next(m for m in lr if m.which() == "carParams").carParams

    msg = get_random_vehicle_parameters(CP)
    params.put("LiveParametersV2", msg.to_bytes(), block=True)
    params.put("CarParamsPrevRoute", CP.as_builder().to_bytes(), block=True)

    sr, sf, offset, p_init = retrieve_initial_vehicle_params(params, CP, replay=True, debug=True)
    np.testing.assert_allclose(sr, msg.vehicleParameters.steerRatio)
    np.testing.assert_allclose(sf, msg.vehicleParameters.stiffnessFactor)
    np.testing.assert_allclose(offset, msg.vehicleParameters.angleOffsetAverageDeg)
    np.testing.assert_equal(p_init.shape, CarKalman.P_initial.shape)
    np.testing.assert_allclose(np.diagonal(p_init), msg.vehicleParameters.debugFilterState.std)

  def test_wheelbase_change_keeps_angle_per_curvature(self):
    # the same car with a new wheelbase (a Rivian R1S now set from the VIN): the learned steer ratio x wheelbase is kept
    params = Params()
    last_CP = car.CarParams.new_message(carFingerprint="RIVIAN_R1", steerRatio=15.2, wheelbase=3.45)
    CP = car.CarParams.new_message(carFingerprint="RIVIAN_R1", steerRatio=15.2, wheelbase=3.08)
    msg = messaging.new_message("vehicleParameters")
    msg.vehicleParameters.steerRatio = 13.88
    params.put("LiveParametersV2", msg.to_bytes(), block=True)
    params.put("CarParamsPrevRoute", last_CP.to_bytes(), block=True)
    sr, _, _, _ = retrieve_initial_vehicle_params(params, CP, replay=True, debug=False)
    np.testing.assert_allclose(sr * CP.wheelbase, 13.88 * last_CP.wheelbase, rtol=1e-6)
    # an unchanged wheelbase restores the ratio as is
    params.put("CarParamsPrevRoute", CP.to_bytes(), block=True)
    sr, _, _, _ = retrieve_initial_vehicle_params(params, CP, replay=True, debug=False)
    np.testing.assert_allclose(sr, 13.88, rtol=1e-6)

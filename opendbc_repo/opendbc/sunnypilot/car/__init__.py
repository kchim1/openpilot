"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from opendbc.car import structs

# A car port sends this button event once per Traffic Mode toggle (Rivian: the right thumb control's middle button held 1 s), and
# selfdrived flips the TrafficMode param on it. ButtonEvent.Type has no dedicated value: leftBlinker is used because
# no car port sends it as a button event and nothing in openpilot reacts to it. Not altButton2 or lkas: downstream
# Rivian branches (gerrylum fix/stg-*) already send those for the ACC stalk.
TRAFFIC_MODE_BUTTON = structs.CarState.ButtonEvent.Type.leftBlinker

# A car port whose distance control has a direction (Rivian: the right thumb wheel) sends one of these per notch, and
# selfdrived steps the Following Distance setting. ButtonEvent.Type has no values with a direction (and car.capnp can't
# change under the prebuilt): farther is gapAdjustCruise, closer is rightBlinker, which no car port sends as a button
# event and nothing reacts to. Other brands' distance buttons send gapAdjustCruise and keep cycling the personality:
# selfdrived only treats these as steps on a car with the wheel
FOLLOW_DISTANCE_FARTHER_BUTTON = structs.CarState.ButtonEvent.Type.gapAdjustCruise
FOLLOW_DISTANCE_CLOSER_BUTTON = structs.CarState.ButtonEvent.Type.rightBlinker


def crc8_pedal(data):
  crc = 0xFF    # standard init value
  poly = 0xD5   # standard crc8: x8+x7+x6+x4+x2+1
  size = len(data)
  for i in range(size - 1, -1, -1):
    crc ^= data[i]
    for _ in range(8):
      if (crc & 0x80) != 0:
        crc = ((crc << 1) ^ poly) & 0xFF
      else:
        crc <<= 1
  return crc


def create_gas_interceptor_command(packer, gas_amount, idx):
  # Common gas pedal msg generator
  enable = gas_amount > 0.001

  values = {
    "ENABLE": enable,
    "PEDAL_COUNTER": idx & 0xF,
  }

  if enable:
    values["GAS_COMMAND"] = gas_amount * 255.
    values["GAS_COMMAND2"] = gas_amount * 255.

  dat = packer.make_can_msg("GAS_COMMAND", 0, values)[1]

  checksum = crc8_pedal(dat[:-1])
  values["PEDAL_CHECKSUM"] = checksum

  return packer.make_can_msg("GAS_COMMAND", 0, values)

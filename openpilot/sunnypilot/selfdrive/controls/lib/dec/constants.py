class WMACConstants:
  # Lead detection parameters
  LEAD_WINDOW_SIZE = 6  # Stable detection window
  LEAD_PROB = 0.45  # Balanced threshold for lead detection

  # Slow down detection parameters
  SLOW_DOWN_WINDOW_SIZE = 5  # Responsive but stable
  SLOW_DOWN_PROB = 0.3  # Balanced threshold for slow down scenarios

  # Optimized slow down distance curve - smooth and progressive
  SLOW_DOWN_BP = [0., 10., 20., 30., 40., 50., 55., 60.]
  SLOW_DOWN_DIST = [32., 46., 64., 86., 108., 130., 145., 165.]
  # Brake Sooner for Stops (BrakeSoonerForStops): the curve above holds 165 m from 60 km/h up, while the truck covers
  # 194-278 m in the model's 10 s plan at 70-100 km/h, so there DEC only switched once the model already wanted about
  # -1 m/s^2 (route 539e9b3496be3ce9/00000017). Keep it at 92% of the 10 s distance up to 100 km/h, then hold it.
  SLOW_DOWN_BP_BRAKE_SOONER = SLOW_DOWN_BP + [70., 80., 90., 100.]
  SLOW_DOWN_DIST_BRAKE_SOONER = SLOW_DOWN_DIST + [178.9, 204.4, 230., 255.6]

  # Slowness detection parameters
  SLOWNESS_WINDOW_SIZE = 10  # Stable slowness detection
  SLOWNESS_PROB = 0.55  # Clear threshold for slowness
  SLOWNESS_CRUISE_OFFSET = 1.025  # Conservative cruise speed offset
